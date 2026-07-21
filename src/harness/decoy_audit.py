"""Pre-docking quality control for active and presumed-decoy intake sets.

This audit asks whether docking could distinguish the two cohorts by simple
chemistry rather than target recognition.  It never treats presumed decoys as
experimentally proven inactive and never combines the checks into a molecular
quality score.

This is basically a report card for the decoys that were selected, there is no rejection anywhere
the 60,950 decoys are locked
The audit reads them, measures them, and stamps a verdict on the whole set at once. 

Usage:
    python -m src.harness.decoy_audit \
        --active-intake ACTIVE_INTAKE/molecules.csv \
        --decoy-intake DECOY_INTAKE/molecules.csv \
        --outdir RESULTS/decoy_audit
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import ks_2samp

from src.analysis.chemistry import _load_intake
from . import runtime


AUDIT_SCHEMA_VERSION = 1
PASS_LIMIT = 0.10
WARN_LIMIT = 0.20
DEFAULT_TOPOLOGY_THRESHOLD = 0.50

MATCHING_PROPERTIES = (
    ("molecular_weight", "MW (Da)"),
    ("clogp", "cLogP"),
    ("hbond_donors", "H-bond donors"),
    ("hbond_acceptors", "H-bond acceptors"),
    ("rotatable_bonds", "Rotatable bonds"),
    ("formal_charge", "Formal charge"),
)


class DecoyAuditError(ValueError):
    """Raised when a decoy audit input or output is ambiguous."""





######## wraps every number and raises if it's NaN/inf ###########

def _finite(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise DecoyAuditError("A decoy-audit calculation produced a non-finite value")
    return value


######## summary stats (n, mean, std, median, p05/p95, min/max) for one property ###########

def _describe(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": _finite(np.mean(values)),
        "std": _finite(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": _finite(np.median(values)),
        "p05": _finite(np.quantile(values, 0.05)),
        "p95": _finite(np.quantile(values, 0.95)),
        "min": _finite(np.min(values)),
        "max": _finite(np.max(values)),
    }





















#####   Computes SMD  ######

def _standardized_mean_difference(
    active_values: np.ndarray,
    decoy_values: np.ndarray,
) -> float:
    active_variance = (
        float(np.var(active_values, ddof=1)) if active_values.size > 1 else 0.0
    )
    decoy_variance = (
        float(np.var(decoy_values, ddof=1)) if decoy_values.size > 1 else 0.0
    )
    denominator = active_values.size + decoy_values.size - 2
    pooled_variance = (
        (
            (active_values.size - 1) * active_variance
            + (decoy_values.size - 1) * decoy_variance
        )
        / denominator
        if denominator > 0
        else 0.0
    )
    mean_difference = float(np.mean(active_values) - np.mean(decoy_values))
    if pooled_variance <= 0.0:
        return 0.0 if mean_difference == 0.0 else float("inf")
    return mean_difference / float(np.sqrt(pooled_variance))








######## THE GATE ###########

def _balance_status(abs_smd: float, ks_statistic: float) -> str:
    if abs_smd <= PASS_LIMIT and ks_statistic <= PASS_LIMIT:       # pass if both |SMD| and KS ≤ 0.10
        return "pass"                                              # warn if both ≤ 0.20
    if abs_smd <= WARN_LIMIT and ks_statistic <= WARN_LIMIT:       # else fail
        return "warn"
    return "fail"


def _property_balance(actives: pd.DataFrame, decoys: pd.DataFrame) -> dict:     
    result = {}
    for column, label in MATCHING_PROPERTIES:
        active_values = actives[column].to_numpy(float)                         # this one  runs SMD + KS for all six properties
        decoy_values = decoys[column].to_numpy(float)                           # and returns per-property verdicts
        smd = _standardized_mean_difference(active_values, decoy_values)
        abs_smd = abs(smd)
        ks_statistic = _finite(
            ks_2samp(active_values, decoy_values, method="auto").statistic
        )
        result[column] = {
            "label": label,
            "actives": _describe(active_values),
            "decoys": _describe(decoy_values),
            "active_minus_decoy_mean": _finite(
                np.mean(active_values) - np.mean(decoy_values)
            ),
            "standardized_mean_difference": (
                _finite(smd) if np.isfinite(smd) else None
            ),
            "absolute_standardized_mean_difference": (
                _finite(abs_smd) if np.isfinite(abs_smd) else None
            ),
            "ks_statistic": ks_statistic,
            "status": (
                _balance_status(abs_smd, ks_statistic)
                if np.isfinite(abs_smd)
                else "fail"
            ),
        }
    return result














#####  second, independent fingerprint definition  ######

def _fingerprints(frame: pd.DataFrame):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    )
    fingerprints = []
    for row in frame.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.parent_smiles))
        if molecule is None:
            raise DecoyAuditError(
                f"Previously accepted parent no longer parses: {row.molecule_id}"
            )
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def _topology_audit(
    actives: pd.DataFrame,
    decoys: pd.DataFrame,
    threshold: float,
) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise DecoyAuditError("topology threshold must be between 0 and 1")

    active_fingerprints = _fingerprints(actives)
    decoy_fingerprints = _fingerprints(decoys)
    maxima = []
    examples = []
    for row, fingerprint in zip(
        decoys[["molecule_id", "parent_smiles"]].itertuples(index=False),
        decoy_fingerprints,
    ):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            active_fingerprints,
        )
        best_index = int(np.argmax(similarities))
        best_value = float(similarities[best_index])
        maxima.append(best_value)
        examples.append(
            {
                "decoy_id": str(row.molecule_id),
                "decoy_parent_smiles": str(row.parent_smiles),
                "nearest_active_id": str(
                    actives.iloc[best_index]["molecule_id"]
                ),
                "max_active_tanimoto": _finite(best_value),
            }
        )

    values = np.asarray(maxima, dtype=float)
    examples.sort(
        key=lambda item: (-item["max_active_tanimoto"], item["decoy_id"])
    )
    return {
        "method": (
            "RDKit Morgan radius 2, 2048 bits, chirality excluded; "
            "Tanimoto against every accepted active parent"
        ),
        "threshold": threshold,
        "n_decoys": int(values.size),
        "median_max_active_tanimoto": _finite(np.median(values)),
        "p95_max_active_tanimoto": _finite(np.quantile(values, 0.95)),
        "p99_max_active_tanimoto": _finite(np.quantile(values, 0.99)),
        "largest_max_active_tanimoto": _finite(np.max(values)),
        "n_above_threshold": int(np.sum(values > threshold)),
        "highest_similarity_examples": examples[:10],
    }


def build_decoy_audit(
    active_intake_path: Path,
    decoy_intake_path: Path,
    *,
    compute_topology: bool = True,
    topology_threshold: float = DEFAULT_TOPOLOGY_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Return the two accepted cohorts and a JSON-safe pre-docking audit."""

    actives, active_audit = _load_intake(active_intake_path, "actives")
    decoys, decoy_audit = _load_intake(decoy_intake_path, "decoys")
    combined = pd.concat([actives, decoys], ignore_index=True)

    active_parents = set(actives["parent_smiles"].astype(str))
    overlap = decoys.loc[
        decoys["parent_smiles"].astype(str).isin(active_parents),
        ["molecule_id", "parent_smiles"],
    ]
    property_balance = _property_balance(actives, decoys)
    property_statuses = [
        record["status"] for record in property_balance.values()
    ]

    topology = (
        _topology_audit(actives, decoys, topology_threshold)
        if compute_topology
        else None
    )
    status = "pass"
    if "warn" in property_statuses:
        status = "warn"
    if (
        "fail" in property_statuses
        or not overlap.empty
        or (
            topology is not None
            and topology["n_above_threshold"] > 0
        )
    ):
        status = "fail"

    record = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": "pre_docking_decoy_audit",
        "interpretation": {
            "decoy_label": (
                "presumed inactive for benchmarking; not experimental evidence "
                "of inactivity unless separately documented"
            ),
            "purpose": (
                "detect simple-property and topology leakage before docking"
            ),
            "no_composite_molecular_score": True,
        },
        "thresholds": {
            "property_pass_max_abs_smd_and_ks": PASS_LIMIT,
            "property_warn_max_abs_smd_and_ks": WARN_LIMIT,
            "topology_max_tanimoto": (
                topology_threshold if compute_topology else None
            ),
            "note": (
                "Project QA thresholds; they are diagnostics rather than "
                "universal chemical laws"
            ),
        },
        "status": status,
        "counts": {
            "actives": int(len(actives)),
            "decoys": int(len(decoys)),
            "exact_parent_overlaps": int(len(overlap)),
        },
        "exact_parent_overlap_examples": [
            {
                "decoy_id": str(row.molecule_id),
                "parent_smiles": str(row.parent_smiles),
            }
            for row in overlap.head(10).itertuples(index=False)
        ],
        "property_balance": property_balance,
        "topology": topology,
        "intake_audit": {
            "actives": active_audit,
            "decoys": decoy_audit,
        },
        "inputs": {
            "active_intake": runtime.file_record(Path(active_intake_path)),
            "decoy_intake": runtime.file_record(Path(decoy_intake_path)),
            "active_intake_summary": runtime.file_record(
                Path(active_intake_path).parent / "summary.json"
            ),
            "decoy_intake_summary": runtime.file_record(
                Path(decoy_intake_path).parent / "summary.json"
            ),
        },
    }
    return combined, record















###########    six overlaid active-vs-decoy histograms in Plotly. Eye-candy for the report    ############


def _matching_figure(frame: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[label for _, label in MATCHING_PROPERTIES],
    )
    colors = {"actives": "#b91c1c", "decoys": "#2563eb"}
    for index, (column, _) in enumerate(MATCHING_PROPERTIES):
        row = index // 3 + 1
        column_index = index % 3 + 1
        for cohort in ("actives", "decoys"):
            values = frame.loc[frame["cohort"].eq(cohort), column]
            figure.add_trace(
                go.Histogram(
                    x=values,
                    name=cohort,
                    legendgroup=cohort,
                    showlegend=index == 0,
                    histnorm="probability",
                    opacity=0.55,
                    marker_color=colors[cohort],
                    nbinsx=45,
                    hovertemplate="%{x}<br>fraction=%{y:.4f}<extra>"
                    + cohort
                    + "</extra>",
                ),
                row=row,
                col=column_index,
            )
    figure.update_layout(
        barmode="overlay",
        height=720,
        title="Pre-docking active–decoy property matching",
        template="plotly_white",
    )
    return figure




### builds the HTML ####

def _render_report(frame: pd.DataFrame, audit: dict) -> str:
    rows = []
    for record in audit["property_balance"].values():
        abs_smd = record["absolute_standardized_mean_difference"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['label'])}</td>"
            f"<td>{record['actives']['mean']:.4g}</td>"
            f"<td>{record['decoys']['mean']:.4g}</td>"
            f"<td>{abs_smd:.4f}</td>"
            f"<td>{record['ks_statistic']:.4f}</td>"
            f"<td class='{record['status']}'>{record['status'].upper()}</td>"
            "</tr>"
        )
    topology = audit["topology"]
    if topology is None:
        topology_html = "<p>Topology comparison was disabled.</p>"
    else:
        topology_html = (
            "<p>Maximum active similarity uses Morgan radius 2 fingerprints. "
            f"Median: {topology['median_max_active_tanimoto']:.3f}; "
            f"95th percentile: {topology['p95_max_active_tanimoto']:.3f}; "
            f"above {topology['threshold']:.2f}: "
            f"{topology['n_above_threshold']:,}.</p>"
        )
    figure_div = _matching_figure(frame).to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pre-docking decoy audit</title>
<script>{get_plotlyjs()}</script>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1180px;margin:0 auto;padding:28px 20px 60px;color:#1f2937;background:#f8fafc}}
.card{{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:16px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #e5e7eb;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.pass{{color:#166534;font-weight:700}}.warn{{color:#a16207;font-weight:700}}.fail{{color:#b91c1c;font-weight:700}}
</style></head><body>
<h1>Pre-docking decoy audit</h1>
<p>Status: <strong class="{audit['status']}">{audit['status'].upper()}</strong>.
This is a leakage check, not a composite molecule score.</p>
<div class="card"><table><thead><tr><th>Property</th><th>Active mean</th>
<th>Decoy mean</th><th>|SMD|</th><th>KS</th><th>Gate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="card">{topology_html}
<p>Exact active–decoy parent overlaps: {audit['counts']['exact_parent_overlaps']:,}.</p>
</div>
<div class="card">{figure_div}</div>
<p>Decoys are presumed negatives unless independent experimental provenance says otherwise.</p>
</body></html>"""






######  writes atomically via a .partial dir then replace  ######

def write_decoy_audit(
    active_intake_path: Path,
    decoy_intake_path: Path,
    outdir: Path,
    *,
    compute_topology: bool = True,
    topology_threshold: float = DEFAULT_TOPOLOGY_THRESHOLD,
) -> Path:
    """Write one immutable pre-docking audit directory."""

    outdir = Path(outdir)
    partial = outdir.with_name(f".{outdir.name}.partial")
    if outdir.exists() or partial.exists():
        raise DecoyAuditError(f"Audit output path already exists: {outdir}")
    partial.mkdir(parents=True)

    frame, audit = build_decoy_audit(
        active_intake_path,
        decoy_intake_path,
        compute_topology=compute_topology,
        topology_threshold=topology_threshold,
    )
    runtime.write_json_atomic(partial / "quality.json", audit)
    report_path = partial / "report.html"
    report_path.write_text(_render_report(frame, audit), encoding="utf-8")
    partial.replace(outdir)
    return outdir / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit active/decoy property matching before docking"
    )
    parser.add_argument("--active-intake", type=Path, required=True)
    parser.add_argument("--decoy-intake", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--topology-threshold",
        type=float,
        default=DEFAULT_TOPOLOGY_THRESHOLD,
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="skip fingerprint comparison (property audit still runs)",
    )
    args = parser.parse_args()
    report_path = write_decoy_audit(
        args.active_intake,
        args.decoy_intake,
        args.outdir,
        compute_topology=not args.skip_topology,
        topology_threshold=args.topology_threshold,
    )
    audit = json.loads(
        (report_path.parent / "quality.json").read_text(encoding="utf-8")
    )
    print(
        f"Pre-docking decoy audit: {audit['status'].upper()} | "
        f"{audit['counts']['actives']} actives | "
        f"{audit['counts']['decoys']} presumed decoys"
    )
    for record in audit["property_balance"].values():
        print(
            f"  {record['label']}: |SMD|="
            f"{record['absolute_standardized_mean_difference']:.3f}, "
            f"KS={record['ks_statistic']:.3f} [{record['status']}]"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
