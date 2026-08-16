"""Analyze one unlabeled generated-candidate cohort.

Candidate mode is intentionally separate from active-versus-decoy benchmark
validation.  It reports intake quality, docking-score coverage/distribution,
chemical properties, and computational cost.  It never computes AUC, BEDROC,
or enrichment because a candidate cohort has no known binary activity labels.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from src.harness import runtime

from . import provenance
from .chemistry import (
    PROPERTY_COLUMNS,
    PROPERTY_LABELS,
    PROPERTY_SPECS,
    _load_intake,
    _property_statistics,
)
from .dataset import AnalysisInputError, load_score_table
from .interpretation import candidate_interpretation
from .molecule_gallery import render_top_bottom_galleries
from .report_theme import plotly_config, polish_plotly_figure, report_css, report_toolbar


CANDIDATE_SCHEMA_VERSION = 1
VALID_ROLES = ("model", "naive_baseline", "pilot", "unassigned")   # Roles for each cohort






#### Convert NaN and +/- inf to None for JSON file hygiene ####

def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None




#######  Twelve summary numbers over the observed scores  #######

def _score_distribution(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": _finite_or_none(np.mean(values)),
        "std": (
            _finite_or_none(np.std(values, ddof=1))
            if values.size > 1
            else None
        ),
        "minimum_best": _finite_or_none(np.min(values)),        # More negative is better
        "p05": _finite_or_none(np.quantile(values, 0.05)),      # docking convention
        "p10": _finite_or_none(np.quantile(values, 0.10)),      # these are the statistics for the
        "p25": _finite_or_none(np.quantile(values, 0.25)),      # docking scores only
        "median": _finite_or_none(np.median(values)),
        "p75": _finite_or_none(np.quantile(values, 0.75)),
        "p90": _finite_or_none(np.quantile(values, 0.90)),
        "p95": _finite_or_none(np.quantile(values, 0.95)),
        "maximum_worst": _finite_or_none(np.max(values)),
    }


def _load_docking_cost(scores_path: Path) -> tuple[dict | None, Path | None]:
    summary_path = Path(scores_path).parent / "_dock_summary.json"      # Looks for the JSON file
    if not summary_path.is_file():                                      # silent if JSON missing
        return None, None                                               # Loud if missmatch or missing hash
    try:                                                                # Hashing for extra verification
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisInputError(
            f"Cannot read docking cost summary {summary_path}: {error}"
        ) from error
    if record.get("stage") != "smina_docking":
        raise AnalysisInputError(f"{summary_path} is not a Smina docking summary")
    recorded_hash = (
        record.get("outputs", {}).get("scores_csv", {}).get("sha256")
    )
    actual_hash = runtime.sha256_file(scores_path)
    if not recorded_hash or recorded_hash != actual_hash:
        raise AnalysisInputError(
            f"{summary_path} does not match candidate scores {scores_path}"
        )
    return record, summary_path


def _screening_protocol(cost: dict | None) -> dict | None:
    if cost is None:
        return None
    receptor = cost.get("inputs", {}).get("receptor_pdbqt")
    receptor_identity = None
    if isinstance(receptor, dict):
        receptor_identity = {
            "sha256": receptor.get("sha256"),
            "missing": receptor.get("missing", False),
        }
    return {
        "engine_stage": cost.get("stage"),
        "receptor": receptor_identity,
        "parameters": cost.get("parameters", {}),
    }









########## For each of the 12 properties, we do a spearman ρ agains the docking score ############


def _score_property_correlations(joined: pd.DataFrame) -> dict:
    observed = joined.loc[joined["score"].notna()]
    values = {}
    for column in PROPERTY_COLUMNS:
        if (
            len(observed) < 3
            or observed["score"].nunique() < 2          # Guardrails for low sample size
            or observed[column].nunique() < 2           # They force None for the JSON stuff
        ):
            values[column] = None
        else:
            rho = spearmanr(observed["score"], observed[column]).statistic
            values[column] = _finite_or_none(rho)
    return {
        "n_observed_scores": int(len(observed)),
        "rho": values,
        "interpretation": (
            "descriptive association only; correlation does not establish "
            "target-specific binding or causation"
        ),
    }





























def build_candidate_profile(
    scores_path: Path,
    intake_path: Path,
    *,
    name: str,
    role: str,
    source_description: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Validate and summarize one candidate cohort."""

    if role not in VALID_ROLES:
        raise AnalysisInputError(
            f"Unknown candidate role {role!r}; choose one of {VALID_ROLES}"            #### 6 GATES EXECUTED IN ORDER
        )
    properties, intake_audit = _load_intake(intake_path, "candidates")             # 1 # Role gate - fail loud on an unknown role        
    scores, score_audit = load_score_table(scores_path, "candidates")              # 2 # Load Both Tables gate - or else
    accepted_ids = set(properties["molecule_id"].astype(str))                      # 3 # Referential integrity check
    score_ids = set(scores["molecule_id"].astype(str))                                      # Score IDs must be a subset of Accepted IDs
    unknown = sorted(score_ids - accepted_ids)                                              # ALSO, the intake table is the authority
    if unknown:                                                                             # on what was legitimately evaluated
        raise AnalysisInputError(
            "Candidate score IDs are absent from accepted intake parents: "        # 4 # The Join. Left join, intake on the left.
            + ", ".join(unknown[:8])                                               # 5 # The emptyt check
        )                                                                          # 6 # Profile Asse,bly

    joined = properties.merge(
        scores,
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )
    if "status" in joined:
        joined["status"] = joined["status"].fillna("absent_from_scores")
    else:
        joined["status"] = np.where(
            joined["molecule_id"].astype(str).isin(score_ids),
            "",
            "absent_from_scores",
        )
    if "reason" not in joined:
        joined["reason"] = ""
    joined["reason"] = joined["reason"].fillna("")
    observed = joined.loc[joined["score"].notna()].copy()
    if observed.empty:
        raise AnalysisInputError("Candidate cohort contains no observed docking scores")

    cost, cost_path = _load_docking_cost(Path(scores_path))
    profile = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "stage": "candidate_analysis",
        "name": name,
        "role": role,
        "source_description": source_description,
        "interpretation": {
            "mode": "unlabeled candidate cohort",
            "score_direction": "lower docking score is better",
            "binary_ranking_metrics_applicable": False,
            "auc_bedroc_enrichment_computed": False,
            "no_composite_quality_score": True,
            "docking_is_a_computational_prioritization_signal_not_activity_proof": True,
        },
        "intake": intake_audit,
        "docking": {
            **score_audit,
            "n_accepted_parents": int(len(properties)),
            "n_with_observed_score": int(len(observed)),
            "n_accepted_absent_from_score_table": int(len(accepted_ids - score_ids)),
            "coverage_over_accepted_parents": float(len(observed) / len(properties)),   #    "did my docking work"
            "successful_per_submitted": float(                                          # vs "what fraction of what my model produced survived the whole pipeline."
                len(observed) / intake_audit["submitted_rows"]
            ),
            "score_distribution_kcal_mol": _score_distribution(
                observed["score"].to_numpy(float)
            ),
        },
        "chemistry": {
            "scope": "all evaluated parents accepted by molecule intake",
            "property_definitions": [
                {"key": key, "label": label, "method": method}
                for key, label, method in PROPERTY_SPECS
            ],
            "properties": _property_statistics(properties),
            "score_property_spearman": _score_property_correlations(joined),
        },
        "computational_cost": cost,
        "screening_protocol": _screening_protocol(cost),
        "inputs": {
            "scores": runtime.file_record(scores_path),
            "intake_molecules": runtime.file_record(intake_path),
            "docking_summary": (
                runtime.file_record(cost_path) if cost_path is not None else None
            ),
        },
    }
    profile["interpretation"]["neutral_summary"] = candidate_interpretation(
        profile
    )
    return joined, profile












def _run_paths(outdir: Path, name: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise AnalysisInputError(
            "Run name must be 1-80 characters using letters, digits, '.', '_', or '-'"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    final = Path(outdir) / f"{name}_{timestamp}"
    working = final.with_name(f".{final.name}.partial")
    if final.exists() or working.exists():
        raise AnalysisInputError(f"Candidate output path already exists: {final}")
    working.mkdir(parents=True)
    return working, final






####     Charts     ####

def _candidate_figures(joined: pd.DataFrame) -> tuple[str, str]:
    observed = joined.loc[joined["score"].notna()].sort_values("score")
    overview = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Docking-score distribution", "Ranked observed scores"),
    )
    overview.add_histogram(
        x=observed["score"],
        nbinsx=40,
        name="Candidates",
        marker_color="#2563eb",
        row=1,
        col=1,
    )
    overview.add_scatter(
        x=np.arange(1, len(observed) + 1),
        y=observed["score"],
        mode="markers",
        name="Candidates",
        marker={"size": 6, "color": observed["clogp"], "colorscale": "Viridis"},
        row=1,
        col=2,
    )
    overview.update_xaxes(title_text="Score (kcal/mol)", row=1, col=1)
    overview.update_xaxes(title_text="Rank", row=1, col=2)
    overview.update_yaxes(title_text="Count", row=1, col=1)
    overview.update_yaxes(title_text="Score (kcal/mol; lower is better)", row=1, col=2)
    overview.update_layout(height=430, showlegend=False, margin={"t": 55})
    polish_plotly_figure(overview, height=430)

    properties = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=[PROPERTY_LABELS[column] for column in PROPERTY_COLUMNS],
    )
    for position, column in enumerate(PROPERTY_COLUMNS):
        row = position // 4 + 1
        col = position % 4 + 1
        properties.add_histogram(
            x=joined[column],
            nbinsx=35,
            marker_color="#0f766e",
            showlegend=False,
            row=row,
            col=col,
        )
    properties.update_layout(height=840, margin={"t": 60})
    polish_plotly_figure(properties, height=840)
    return (
        overview.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="candidate-score-overview",
            config=plotly_config(),
        ),
        properties.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="candidate-property-distributions",
            config=plotly_config(),
        ),
    )


def _render_report(joined: pd.DataFrame, profile: dict) -> str:
    overview_div, property_div = _candidate_figures(joined)
    docking = profile["docking"]
    intake = profile["intake"]
    distribution = docking["score_distribution_kcal_mol"]
    property_rows = []
    for column in PROPERTY_COLUMNS:
        values = profile["chemistry"]["properties"][column]
        property_rows.append(
            {
                "Property": PROPERTY_LABELS[column],
                "Mean": values["mean"],
                "Median": values["median"],
                "Minimum": values["min"],
                "Maximum": values["max"],
            }
        )
    property_table = pd.DataFrame(property_rows).to_html(
        index=False,
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    ranked = joined.loc[joined["score"].notna()].sort_values(
        ["score", "molecule_id"]
    )
    top_table = ranked[
        ["molecule_id", "score", "qed", "sa_score", "molecular_weight", "clogp", "tpsa_a2"]
    ].head(20).rename(
        columns={
            "molecule_id": "Molecule",
            "score": "Docking score",
            "qed": "QED",
            "sa_score": "SA score",
            "molecular_weight": "MW",
            "clogp": "cLogP",
            "tpsa_a2": "TPSA",
        }
    ).to_html(
        index=False,
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    cost_note = (
        "Authenticated sibling docking-cost summary loaded."
        if profile["computational_cost"] is not None
        else "No authenticated sibling docking-cost summary was available."
    )
    neutral_summary = profile.get("interpretation", {}).get(
        "neutral_summary", candidate_interpretation(profile)
    )
    gallery = render_top_bottom_galleries(joined, n=10)
    toolbar = report_toolbar(f"{profile['name']} candidate analysis")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(profile['name'])} candidate analysis</title>
<script>{get_plotlyjs()}</script>
<style>{report_css()}</style></head>
<body>{toolbar}<h1>{html.escape(profile['name'])}: candidate-only analysis</h1>
<p>Role: <strong>{html.escape(profile['role'])}</strong>. This cohort has no active/decoy labels, so AUC, BEDROC, and enrichment are deliberately not computed. Docking prioritizes candidates; it does not prove biochemical activity.</p>
<div class="metrics"><div class="metric">Submitted<strong>{intake['submitted_rows']:,}</strong></div><div class="metric">Validity<strong>{100*intake['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100*intake['parent_uniqueness']:.2f}%</strong></div><div class="metric">Accepted<strong>{intake['accepted_for_preparation']:,}</strong></div><div class="metric">Docking coverage<strong>{100*docking['coverage_over_accepted_parents']:.2f}%</strong></div><div class="metric">Median score<strong>{distribution['median']:.3f}</strong>kcal/mol</div></div>
<div class="card"><strong>Neutral interpretation</strong><p>{html.escape(str(neutral_summary))}</p></div>
<div class="card">{overview_div}</div>
{gallery}
<h2>Evaluated-parent chemistry</h2><p>All properties remain separate; no composite molecular-quality score is created.</p><div class="card">{property_table}</div><div class="card">{property_div}</div>
<h2>Top 20 observed docking scores</h2><p>Lower is better. QED and SA are shown beside docking, not folded into it.</p><div class="card">{top_table}</div>
<h2>Computational cost</h2><p>{html.escape(cost_note)}</p>
</body></html>"""


def run_candidate_analysis(
    scores_path: Path,
    intake_path: Path,
    name: str,
    outdir: Path,
    *,
    role: str = "unassigned",
    source_description: str = "",
) -> Path:
    """Write an immutable candidate report, metrics, ranking, and provenance."""

    joined, profile = build_candidate_profile(
        scores_path,
        intake_path,
        name=name,
        role=role,
        source_description=source_description,
    )
    working, final = _run_paths(outdir, name)
    ranked = joined.loc[joined["score"].notna()].sort_values(
        ["score", "molecule_id"]
    ).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked_path = working / "ranked_candidates.csv"
    ranked.to_csv(ranked_path, index=False)
    runtime.write_json_atomic(working / "metrics.json", profile)
    (working / "report.html").write_text(
        _render_report(joined, profile),
        encoding="utf-8",
    )
    def final_output_record(path: Path) -> dict:
        record = runtime.file_record(path)
        record["path"] = str((final / path.name).resolve())
        return record

    runtime.write_json_atomic(
        working / "run_log.json",
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "stage": "candidate_analysis_run_log",
            "name": name,
            "role": role,
            "provenance": {
                "git": provenance.git_state(Path(__file__).resolve().parents[2]),
                "software": provenance.software_versions(),
                "analysis_time_harness_snapshot": provenance.harness_snapshot(),
            },
            "outputs": {
                "metrics": final_output_record(working / "metrics.json"),
                "ranked_candidates": final_output_record(ranked_path),
                "report": final_output_record(working / "report.html"),
            },
        },
    )
    working.replace(final)
    return final / "report.html"













def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one unlabeled generated-candidate cohort"
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/candidates"))
    parser.add_argument("--role", choices=VALID_ROLES, default="unassigned")
    parser.add_argument("--source-description", default="")
    args = parser.parse_args()
    try:
        report = run_candidate_analysis(
            args.scores,
            args.intake,
            args.name,
            args.outdir,
            role=args.role,
            source_description=args.source_description,
        )
    except AnalysisInputError as error:
        parser.error(str(error))
    print(f"Candidate report: {report}")


if __name__ == "__main__":
    main()
