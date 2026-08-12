"""Audit all raw Gen1 samples and write the Gen1-specific report section.

The audit is descriptive.  It never filters, ranks, or selects molecules for
docking.  The registered 1,000-molecule docking branch is sampled independently from the
raw 10,000 by :mod:`src.generation.run_candidate_pipeline`.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.analysis.chemistry import (
    PROPERTY_COLUMNS,
    PROPERTY_LABELS,
    PROPERTY_SPECS,
    _property_statistics,
)
from src.analysis.report_theme import report_css, report_toolbar
from src.generation.gen1_guacamol import (
    MODEL_NAME,
    MOLECULES_NAME,
    RAW_SAMPLES_NAME,
    SAMPLING_NAME,
)
from src.generation.naive_baseline import _load_actives
from src.harness import runtime
from src.harness.intake import _largest_fragment_parent


GEN1_ANALYSIS_SCHEMA_VERSION = 1
REFERENCE_INDEX_SCHEMA_VERSION = 1
REFERENCE_MANIFEST_NAME = "reference_index.json"
REFERENCE_PARENT_KEYS_NAME = "parent_inchikeys.txt.gz"
REFERENCE_SCAFFOLDS_NAME = "scaffolds.smi.gz"


class Gen1AnalysisError(ValueError):
    """Raised when Gen1 sample, intake, or reference provenance is broken."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen1AnalysisError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen1AnalysisError(f"{path} must contain one JSON object")
    return value


def _verify_record(path: Path, record: dict, label: str) -> None:
    if not path.is_file():
        raise Gen1AnalysisError(f"{label} does not exist: {path}")
    recorded_hash = record.get("sha256") if isinstance(record, dict) else None
    if not recorded_hash or runtime.sha256_file(path) != recorded_hash:
        raise Gen1AnalysisError(f"{label} does not match its recorded SHA-256: {path}")


def _parent_key(parent_smiles: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(parent_smiles)
    if molecule is None:
        raise Gen1AnalysisError(
            f"Accepted Gen1 parent no longer parses: {parent_smiles}"
        )
    try:
        with rdBase.BlockLogs():
            key = str(inchi.MolToInchiKey(molecule) or "")
    except Exception as error:
        raise Gen1AnalysisError(
            f"Could not compute an InChIKey for accepted parent {parent_smiles}"
        ) from error
    if not key:
        raise Gen1AnalysisError(
            f"RDKit returned an empty InChIKey for accepted parent {parent_smiles}"
        )
    return key


def _scaffold_smiles(parent_smiles: str) -> str:
    molecule = Chem.MolFromSmiles(parent_smiles)
    if molecule is None:
        raise Gen1AnalysisError(
            f"Accepted Gen1 parent no longer parses: {parent_smiles}"
        )
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _load_reference_index(index_dir: Path | None) -> tuple[set[str], set[str], dict] | None:
    if index_dir is None:
        return None
    index_dir = Path(index_dir)
    manifest_path = index_dir / REFERENCE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != REFERENCE_INDEX_SCHEMA_VERSION:
        raise Gen1AnalysisError("Unsupported Gen1 reference-index schema version")
    if manifest.get("stage") != "gen1_guacamol_reference_index":
        raise Gen1AnalysisError(f"{manifest_path} is not a Gen1 GuacaMol reference index")
    files = manifest.get("outputs", {})
    keys_path = index_dir / REFERENCE_PARENT_KEYS_NAME
    scaffolds_path = index_dir / REFERENCE_SCAFFOLDS_NAME
    _verify_record(keys_path, files.get("parent_inchikeys", {}), "Reference parent keys")
    _verify_record(scaffolds_path, files.get("scaffolds", {}), "Reference scaffolds")
    with gzip.open(keys_path, "rt", encoding="utf-8") as handle:
        parent_keys = {line.strip() for line in handle if line.strip()}
    with gzip.open(scaffolds_path, "rt", encoding="utf-8") as handle:
        scaffolds = {line.strip() for line in handle if line.strip()}
    if len(parent_keys) != manifest.get("counts", {}).get("unique_parent_inchikeys"):
        raise Gen1AnalysisError("Reference parent-key count does not match its manifest")
    if len(scaffolds) != manifest.get("counts", {}).get("unique_nonempty_scaffolds"):
        raise Gen1AnalysisError("Reference scaffold count does not match its manifest")
    return parent_keys, scaffolds, manifest


def build_reference_index(
    dataset_smiles: Path,
    outdir: Path,
    *,
    source_description: str,
) -> dict:
    """Build the one-time parent/scaffold index for GuacaMol training novelty."""

    started = time.perf_counter()
    dataset_smiles = Path(dataset_smiles)
    outdir = Path(outdir)
    if not dataset_smiles.is_file():
        raise Gen1AnalysisError(
            f"GuacaMol training SMILES do not exist: {dataset_smiles}"
        )
    if outdir.exists():
        raise Gen1AnalysisError(f"Reference-index output already exists: {outdir}")
    if not source_description.strip():
        raise Gen1AnalysisError("Reference index requires a source description")

    parent_keys: set[str] = set()
    scaffolds: set[str] = set()
    training_rows = 0
    acyclic_parents = 0
    opener = gzip.open if dataset_smiles.suffix.lower() == ".gz" else open
    with opener(dataset_smiles, "rt", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            training_rows += 1
            raw_smiles = fields[0]
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(raw_smiles)
            if molecule is None:
                raise Gen1AnalysisError(
                    f"GuacaMol training SMILES does not parse at row {row_number}"
                )
            parent, parent_smiles, _ = _largest_fragment_parent(molecule)
            try:
                with rdBase.BlockLogs():
                    parent_key = str(inchi.MolToInchiKey(parent) or "")
            except Exception as error:
                raise Gen1AnalysisError(
                    f"GuacaMol training parent has no InChIKey at row {row_number}"
                ) from error
            if not parent_key:
                raise Gen1AnalysisError(
                    "GuacaMol training parent has an empty InChIKey at row "
                    f"{row_number}"
                )
            parent_keys.add(parent_key)
            scaffold = _scaffold_smiles(parent_smiles)
            if scaffold:
                scaffolds.add(scaffold)
            else:
                acyclic_parents += 1

    if training_rows == 0:
        raise Gen1AnalysisError("GuacaMol training file contains no SMILES rows")
    outdir.mkdir(parents=True)
    keys_path = outdir / REFERENCE_PARENT_KEYS_NAME
    scaffolds_path = outdir / REFERENCE_SCAFFOLDS_NAME
    with gzip.open(keys_path, "wt", encoding="utf-8", newline="\n") as handle:
        for key in sorted(parent_keys):
            handle.write(f"{key}\n")
    with gzip.open(scaffolds_path, "wt", encoding="utf-8", newline="\n") as handle:
        for scaffold in sorted(scaffolds):
            handle.write(f"{scaffold}\n")
    summary = {
        "schema_version": REFERENCE_INDEX_SCHEMA_VERSION,
        "stage": "gen1_guacamol_reference_index",
        "source_description": source_description,
        "input": runtime.file_record(dataset_smiles),
        "selection": "every non-blank row; SMILES is the first whitespace field",
        "normalization": (
            "RDKit parse; deterministic largest-fragment parent; RDKit parent "
            "InChIKey; canonical isomeric Bemis-Murcko scaffold"
        ),
        "counts": {
            "training_rows": training_rows,
            "unique_parent_inchikeys": len(parent_keys),
            "unique_nonempty_scaffolds": len(scaffolds),
            "acyclic_training_parents": acyclic_parents,
        },
        "outputs": {
            "parent_inchikeys": runtime.file_record(keys_path),
            "scaffolds": runtime.file_record(scaffolds_path),
        },
        "software": {"rdkit": rdkit.__version__},
        "hardware": runtime.hardware_record(),
        "timing": runtime.timing_record(
            started,
            attempted_tasks=training_rows,
            workers=1,
        ),
    }
    runtime.write_json_atomic(outdir / REFERENCE_MANIFEST_NAME, summary)
    return summary



def _sampling_inputs(sample_dir: Path) -> tuple[dict, pd.DataFrame]:
    sampling_path = sample_dir / SAMPLING_NAME
    raw_path = sample_dir / RAW_SAMPLES_NAME
    molecules_path = sample_dir / MOLECULES_NAME
    sampling = _read_json(sampling_path)
    if sampling.get("stage") != "gen1_sampling" or sampling.get("model_name") != MODEL_NAME:
        raise Gen1AnalysisError(f"{sampling_path} is not a Gen1 sampling record")
    outputs = sampling.get("outputs", {})
    _verify_record(raw_path, outputs.get("raw_samples_csv", {}), "Raw Gen1 samples")
    _verify_record(molecules_path, outputs.get("molecules_smi", {}), "Gen1 transport SMILES")
    raw = pd.read_csv(
        raw_path,
        dtype={"molecule_id": "string", "raw_smiles": "string"},
        keep_default_na=False,
    )
    required = {
        "sample_index",
        "molecule_id",
        "raw_smiles",
        "terminated_by_eos",
        "hit_max_length",
    }
    missing = required - set(raw.columns)
    if missing:
        raise Gen1AnalysisError(f"Raw Gen1 samples are missing columns: {sorted(missing)}")
    if len(raw) != sampling.get("counts", {}).get("raw_samples"):
        raise Gen1AnalysisError("Raw Gen1 sample count does not match sampling.json")
    expected_indices = np.arange(1, len(raw) + 1)
    if not np.array_equal(pd.to_numeric(raw["sample_index"]).to_numpy(), expected_indices):
        raise Gen1AnalysisError("Raw Gen1 sample_index is not the exact 1..N sequence")
    if raw["molecule_id"].duplicated().any():
        raise Gen1AnalysisError("Raw Gen1 molecule IDs are not unique")
    return sampling, raw


def _intake_inputs(sample_dir: Path, intake_dir: Path, expected_rows: int) -> tuple[dict, pd.DataFrame]:
    summary_path = intake_dir / "summary.json"
    molecules_csv = intake_dir / "molecules.csv"
    summary = _read_json(summary_path)
    if summary.get("stage") != "molecule_intake":
        raise Gen1AnalysisError(f"{summary_path} is not a molecule-intake record")
    if summary.get("input", {}).get("sha256") != runtime.sha256_file(
        sample_dir / MOLECULES_NAME
    ):
        raise Gen1AnalysisError("Full-cohort intake is not linked to the Gen1 sample file")
    if summary.get("counts", {}).get("submitted_rows") != expected_rows:
        raise Gen1AnalysisError("Full-cohort intake did not receive every raw Gen1 draw")
    recorded_hash = summary.get("outputs", {}).get("molecules_csv_sha256")
    if not molecules_csv.is_file() or runtime.sha256_file(molecules_csv) != recorded_hash:
        raise Gen1AnalysisError("Full-cohort molecules.csv fails its intake hash")
    frame = pd.read_csv(
        molecules_csv,
        dtype={"molecule_id": "string", "parent_smiles": "string"},
    )
    return summary, frame


def build_gen1_profile(
    sample_dir: Path,
    intake_dir: Path,
    actives_path: Path,
    *,
    reference_index_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the full-10,000 Gen1 profile without selecting any molecule."""

    sample_dir = Path(sample_dir)
    intake_dir = Path(intake_dir)
    sampling, raw = _sampling_inputs(sample_dir)
    intake, intake_frame = _intake_inputs(sample_dir, intake_dir, len(raw))
    accepted = intake_frame.loc[
        intake_frame["status"].fillna("").astype(str).str.lower().eq("accepted")
    ].copy()
    if accepted.empty:
        raise Gen1AnalysisError("Gen1 full-cohort intake accepted no parent structures")
    for column in PROPERTY_COLUMNS:
        accepted[column] = pd.to_numeric(accepted[column], errors="raise")

    active_keys = {active.inchikey for active in _load_actives(Path(actives_path))}
    accepted["parent_inchikey"] = accepted["parent_smiles"].map(_parent_key)
    accepted["bemis_murcko_scaffold"] = accepted["parent_smiles"].map(
        _scaffold_smiles
    )
    accepted["exact_known_active_parent"] = accepted["parent_inchikey"].isin(active_keys)

    reference = _load_reference_index(reference_index_dir)
    if reference is None:
        novelty = {
            "available": False,
            "reason": (
                "No authenticated GuacaMol training reference index was supplied; "
                "training-set and scaffold novelty are not guessed."
            ),
        }
    else:
        training_parent_keys, training_scaffolds, reference_manifest = reference
        accepted["novel_parent_vs_training"] = ~accepted["parent_inchikey"].isin(
            training_parent_keys
        )
        scaffold_bearing = accepted.loc[accepted["bemis_murcko_scaffold"].ne("")].copy()
        scaffold_bearing["novel_scaffold_vs_training"] = ~scaffold_bearing[
            "bemis_murcko_scaffold"
        ].isin(training_scaffolds)
        novelty = {
            "available": True,
            "reference_index": reference_manifest,
            "novel_unique_parents": int(accepted["novel_parent_vs_training"].sum()),
            "novel_unique_parent_fraction": float(
                accepted["novel_parent_vs_training"].mean()
            ),
            "novel_nonempty_scaffolds": int(
                scaffold_bearing.loc[
                    scaffold_bearing["novel_scaffold_vs_training"],
                    "bemis_murcko_scaffold",
                ].nunique()
            ),
            "unique_nonempty_generated_scaffolds": int(
                scaffold_bearing["bemis_murcko_scaffold"].nunique()
            ),
        }

    counts = intake["counts"]
    metrics = intake["aggregate_metrics"]
    nonempty_scaffolds = accepted.loc[
        accepted["bemis_murcko_scaffold"].ne(""), "bemis_murcko_scaffold"
    ]
    profile = {
        "schema_version": GEN1_ANALYSIS_SCHEMA_VERSION,
        "stage": "gen1_full_cohort_analysis",
        "model_name": MODEL_NAME,
        "scope": "all raw Gen1 draws; no docking subsample selection is used here",
        "interpretation": {
            "target_aware": False,
            "target_activity_labels_available": False,
            "auc_bedroc_enrichment_computed": False,
            "no_composite_quality_score": True,
            "known_active_overlap_is_reported_not_filtered": True,
        },
        "sampling": {
            "seed": sampling["parameters"]["seed"],
            "raw_samples": len(raw),
            "terminated_by_eos": sampling["counts"]["terminated_by_eos"],
            "hit_max_length": sampling["counts"]["hit_max_length"],
            "empty_raw_smiles": sampling["counts"]["empty_raw_smiles"],
            "raw_character_length": sampling["raw_character_length"],
            "checkpoint": sampling["checkpoint"],
        },
        "generation_quality": {
            "valid_structures": int(counts["valid_structures"]),
            "validity": float(metrics["validity"]),
            "unique_valid_parents": int(counts["unique_valid_parents"]),
            "uniqueness_among_valid_parents": float(
                metrics["uniqueness_among_valid_parents"]
            ),
            "accepted_unique_parents": int(counts["accepted_for_preparation"]),
            "accepted_fraction": float(metrics["accepted_fraction"]),
            "duplicate_or_other_rejected_rows": int(counts["rejected_rows"]),
            "multifragment_valid_structures": int(
                counts["multifragment_valid_structures"]
            ),
            "parent_extractions": int(counts["parent_extractions"]),
            "rejection_reasons": counts["rejection_reasons"],
            "exact_known_active_parent_rediscoveries": int(
                accepted["exact_known_active_parent"].sum()
            ),
            "unique_nonempty_bemis_murcko_scaffolds": int(
                nonempty_scaffolds.nunique()
            ),
            "acyclic_accepted_parents": int(
                accepted["bemis_murcko_scaffold"].eq("").sum()
            ),
            "training_novelty": novelty,
        },
        "chemistry": {
            "scope": "all unique parents accepted by full-cohort intake",
            "property_definitions": [
                {"key": key, "label": label, "method": method}
                for key, label, method in PROPERTY_SPECS
            ],
            "properties": _property_statistics(accepted),
        },
        "inputs": {
            "sampling_json": runtime.file_record(sample_dir / SAMPLING_NAME),
            "raw_samples_csv": runtime.file_record(sample_dir / RAW_SAMPLES_NAME),
            "intake_summary": runtime.file_record(intake_dir / "summary.json"),
            "intake_molecules": runtime.file_record(intake_dir / "molecules.csv"),
            "actives": runtime.file_record(Path(actives_path)),
        },
        "software": {"rdkit": rdkit.__version__},
    }
    return accepted, profile


def _render_full_cohort_report(profile: dict) -> str:
    quality = profile["generation_quality"]
    sampling = profile["sampling"]
    novelty = quality["training_novelty"]
    property_rows = []
    for column in PROPERTY_COLUMNS:
        values = profile["chemistry"]["properties"][column]
        property_rows.append(
            {
                "Property": PROPERTY_LABELS[column],
                "Mean": values["mean"],
                "Median": values["median"],
                "SD": values["std"],
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
    if novelty["available"]:
        novelty_text = (
            f"{novelty['novel_unique_parents']:,} unique parents "
            f"({100*novelty['novel_unique_parent_fraction']:.2f}%) were absent "
            "from the authenticated GuacaMol training-parent index; "
            f"{novelty['novel_nonempty_scaffolds']:,}/"
            f"{novelty['unique_nonempty_generated_scaffolds']:,} generated "
            "non-empty scaffolds were absent from its scaffold index."
        )
    else:
        novelty_text = str(novelty["reason"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gen1 full-cohort analysis</title>
<style>{report_css()}</style></head>
<body>{report_toolbar("Gen1 full-cohort audit")}<h1>Gen1: target-unaware GuacaMol SMILES-LSTM</h1>
<p>This section covers all {sampling['raw_samples']:,} raw draws. It is generated before the random docking branch and does not filter or rank any molecule. Because the cohort is unlabelled, AUC, BEDROC, and enrichment are not applicable.</p>
<div class="metrics"><div class="metric">Raw draws<strong>{sampling['raw_samples']:,}</strong></div><div class="metric">Validity<strong>{100*quality['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100*quality['uniqueness_among_valid_parents']:.2f}%</strong></div><div class="metric">Accepted / raw<strong>{100*quality['accepted_fraction']:.2f}%</strong></div><div class="metric">Unique scaffolds<strong>{quality['unique_nonempty_bemis_murcko_scaffolds']:,}</strong></div><div class="metric">Known-active rediscoveries<strong>{quality['exact_known_active_parent_rediscoveries']:,}</strong></div></div>
<div class="card"><h2>Decoder behavior</h2><p>EOS terminated: {sampling['terminated_by_eos']:,}; hit the 100-character ceiling: {sampling['hit_max_length']:,}; empty raw SMILES: {sampling['empty_raw_smiles']:,}. No invalid or duplicate draw was replaced.</p></div>
<div class="card"><h2>Training and scaffold novelty</h2><p>{html.escape(novelty_text)}</p></div>
<h2>Accepted-parent chemistry</h2><p>These are descriptive outcomes on all unique valid parents. No property is used as a post-generation filter and no composite score is created.</p><div class="card">{property_table}</div>
</body></html>"""


def write_gen1_analysis(
    sample_dir: Path,
    intake_dir: Path,
    actives_path: Path,
    outdir: Path,
    *,
    reference_index_dir: Path | None = None,
) -> Path:
    accepted, profile = build_gen1_profile(
        sample_dir,
        intake_dir,
        actives_path,
        reference_index_dir=reference_index_dir,
    )
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen1AnalysisError(f"Gen1 analysis output already exists: {outdir}")
    outdir.mkdir(parents=True)
    accepted[
        [
            "molecule_id",
            "parent_smiles",
            "parent_inchikey",
            "bemis_murcko_scaffold",
            "exact_known_active_parent",
        ]
    ].to_csv(outdir / "evaluated_parents.csv", index=False)
    runtime.write_json_atomic(outdir / "metrics.json", profile)
    report_path = outdir / "report.html"
    report_path.write_text(_render_full_cohort_report(profile), encoding="utf-8")
    runtime.write_json_atomic(
        outdir / "run_log.json",
        {
            "schema_version": GEN1_ANALYSIS_SCHEMA_VERSION,
            "stage": "gen1_full_cohort_analysis_run_log",
            "outputs": {
                "metrics": runtime.file_record(outdir / "metrics.json"),
                "evaluated_parents": runtime.file_record(
                    outdir / "evaluated_parents.csv"
                ),
                "report": runtime.file_record(report_path),
            },
        },
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gen1 full-cohort audit utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("build-reference-index")
    index_parser.add_argument("dataset_smiles", type=Path)
    index_parser.add_argument("outdir", type=Path)
    index_parser.add_argument("--source-description", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("sample_dir", type=Path)
    audit_parser.add_argument("intake_dir", type=Path)
    audit_parser.add_argument("outdir", type=Path)
    audit_parser.add_argument(
        "--actives",
        type=Path,
        default=Path("data/reference/dyrk1a_actives_chembl.csv"),
    )
    audit_parser.add_argument("--reference-index", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build-reference-index":
            summary = build_reference_index(
                args.dataset_smiles,
                args.outdir,
                source_description=args.source_description,
            )
            print(
                f"Gen1 reference index: {summary['counts']['training_rows']:,} training rows"
            )
        else:
            report = write_gen1_analysis(
                args.sample_dir,
                args.intake_dir,
                args.actives,
                args.outdir,
                reference_index_dir=args.reference_index,
            )
            print(f"Gen1 full-cohort report: {report}")
    except Gen1AnalysisError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
