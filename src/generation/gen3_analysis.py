"""


Audit every raw pocket-conditioned Molexar draw before screening selection.




2 HALVES, VERIFY, THEN DESCRIBE:





"""







from __future__ import annotations

import html
import json
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
from src.generation.gen3_molexar import (
    MODEL_NAME,
    MOLECULES_NAME,
    RAW_SAMPLES_NAME,
    SAMPLING_NAME,
)
from src.generation.naive_baseline import _load_actives
from src.harness import runtime


GEN3_ANALYSIS_SCHEMA_VERSION = 1


class Gen3AnalysisError(ValueError):
    """Raised when Gen3 sample, intake, or provenance records disagree."""














#############    VERIFY    ###############



def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen3AnalysisError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen3AnalysisError(f"{path} must contain one JSON object")
    return value


def _verify_record(path: Path, record: dict, label: str) -> None:
    if not path.is_file():
        raise Gen3AnalysisError(f"{label} does not exist: {path}")
    recorded_hash = record.get("sha256") if isinstance(record, dict) else None
    if not recorded_hash or runtime.sha256_file(path) != recorded_hash:
        raise Gen3AnalysisError(f"{label} does not match recorded SHA-256: {path}")


def _parent_key(parent_smiles: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(parent_smiles)
    if molecule is None:
        raise Gen3AnalysisError(f"Accepted Gen3 parent no longer parses: {parent_smiles}")
    try:
        with rdBase.BlockLogs():
            value = str(inchi.MolToInchiKey(molecule) or "")
    except Exception as error:
        raise Gen3AnalysisError(
            f"Could not compute InChIKey for accepted Gen3 parent {parent_smiles}"
        ) from error
    if not value:
        raise Gen3AnalysisError(
            f"RDKit returned empty InChIKey for accepted Gen3 parent {parent_smiles}"
        )
    return value


def _scaffold_smiles(parent_smiles: str) -> str:
    molecule = Chem.MolFromSmiles(parent_smiles)
    if molecule is None:
        raise Gen3AnalysisError(f"Accepted Gen3 parent no longer parses: {parent_smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)





#### Basically a chain of custody check ####

def _sampling_inputs(sample_dir: Path) -> tuple[dict, pd.DataFrame]:
    sampling_path = sample_dir / SAMPLING_NAME
    raw_path = sample_dir / RAW_SAMPLES_NAME
    molecules_path = sample_dir / MOLECULES_NAME
    sampling = _read_json(sampling_path)
    if sampling.get("stage") != "gen3_sampling" or sampling.get("model_name") != MODEL_NAME:
        raise Gen3AnalysisError(f"{sampling_path} is not a Gen3 sampling record")
    outputs = sampling.get("outputs", {})
    _verify_record(raw_path, outputs.get("raw_samples_csv", {}), "Raw Gen3 samples")
    _verify_record(molecules_path, outputs.get("molecules_smi", {}), "Gen3 transport SMILES")
    raw = pd.read_csv(
        raw_path,
        dtype={
            "molecule_id": "string",
            "fragment_selfies": "string",
            "raw_smiles": "string",
        },
        keep_default_na=False,
    )
    required = {
        "sample_index",
        "molecule_id",
        "fragment_selfies",
        "raw_smiles",
        "conversion_success",
    }
    missing = required - set(raw.columns)
    if missing:
        raise Gen3AnalysisError(f"Raw Gen3 samples are missing columns: {sorted(missing)}")
    if len(raw) != sampling.get("counts", {}).get("raw_samples"):
        raise Gen3AnalysisError("Raw Gen3 sample count does not match sampling.json")
    observed_indices = pd.to_numeric(raw["sample_index"]).to_numpy()
    if not np.array_equal(observed_indices, np.arange(1, len(raw) + 1)):
        raise Gen3AnalysisError("Raw Gen3 sample_index is not the exact 1..N sequence")
    if raw["molecule_id"].duplicated().any():
        raise Gen3AnalysisError("Raw Gen3 molecule IDs are not unique")
    return sampling, raw


def _intake_inputs(
    sample_dir: Path,
    intake_dir: Path,
    expected_rows: int,
) -> tuple[dict, pd.DataFrame]:
    summary_path = intake_dir / "summary.json"
    molecules_csv = intake_dir / "molecules.csv"
    summary = _read_json(summary_path)
    if summary.get("stage") != "molecule_intake":
        raise Gen3AnalysisError(f"{summary_path} is not a molecule-intake record")
    if summary.get("input", {}).get("sha256") != runtime.sha256_file(
        sample_dir / MOLECULES_NAME
    ):
        raise Gen3AnalysisError("Full-cohort intake is not linked to Gen3 sample file")
    if summary.get("counts", {}).get("submitted_rows") != expected_rows:
        raise Gen3AnalysisError("Full-cohort intake did not receive every raw Gen3 draw")
    recorded_hash = summary.get("outputs", {}).get("molecules_csv_sha256")
    if not molecules_csv.is_file() or runtime.sha256_file(molecules_csv) != recorded_hash:
        raise Gen3AnalysisError("Full-cohort molecules.csv fails its intake hash")
    frame = pd.read_csv(
        molecules_csv,
        dtype={"molecule_id": "string", "parent_smiles": "string"},
    )
    return summary, frame

























###########    THEN DESCRIBE    #############


def build_gen3_profile(
    sample_dir: Path,
    intake_dir: Path,
    actives_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Build the all-draw Gen3 profile without selecting/ranking candidates."""

    sample_dir = Path(sample_dir)
    intake_dir = Path(intake_dir)
    sampling, raw = _sampling_inputs(sample_dir)
    intake, intake_frame = _intake_inputs(sample_dir, intake_dir, len(raw))
    accepted = intake_frame.loc[
        intake_frame["status"].fillna("").astype(str).str.lower().eq("accepted")
    ].copy()
    if accepted.empty:
        raise Gen3AnalysisError("Gen3 full-cohort intake accepted no parent structures")
    for column in PROPERTY_COLUMNS:
        accepted[column] = pd.to_numeric(accepted[column], errors="raise")

    active_keys = {active.inchikey for active in _load_actives(Path(actives_path))}
    accepted["parent_inchikey"] = accepted["parent_smiles"].map(_parent_key)
    accepted["bemis_murcko_scaffold"] = accepted["parent_smiles"].map(_scaffold_smiles)
    accepted["exact_known_active_parent"] = accepted["parent_inchikey"].isin(active_keys)

    counts = intake["counts"]
    aggregate = intake["aggregate_metrics"]
    nonempty_scaffolds = accepted.loc[
        accepted["bemis_murcko_scaffold"].ne(""), "bemis_murcko_scaffold"
    ]
    profile = {
        "schema_version": GEN3_ANALYSIS_SCHEMA_VERSION,
        "stage": "gen3_full_cohort_analysis",
        "model_name": MODEL_NAME,
        "scope": "all raw Gen3 draws; no docking-subsample selection is used here",
        "interpretation": {
            "target_aware": True,
            "conditioning": "locked 7O7K protein-pocket coordinates only",
            "target_activity_labels_available": False,
            "auc_bedroc_enrichment_computed": False,
            "no_composite_quality_score": True,
            "known_active_overlap_is_reported_not_filtered": True,
            "target_disjoint_training_claimed": False,
            "docking_reward_or_active_ligand_used": False,
        },
        "sampling": {
            "seed": sampling["parameters"]["seed"],
            "raw_samples": len(raw),
            "fragment_selfies_conversion_successes": sampling["counts"][
                "fragment_selfies_conversion_successes"
            ],
            "fragment_selfies_conversion_failures": sampling["counts"][
                "fragment_selfies_conversion_failures"
            ],
            "empty_raw_smiles": sampling["counts"]["empty_raw_smiles"],
            "raw_fragment_selfies_character_length": sampling[
                "raw_fragment_selfies_character_length"
            ],
            "converted_smiles_character_length": sampling[
                "converted_smiles_character_length"
            ],
            "checkpoint": sampling["checkpoint"],
            "target": sampling["target"],
            "pocket_graph": sampling["pocket_graph"],
        },
        "generation_quality": {
            "valid_structures": int(counts["valid_structures"]),
            "validity": float(aggregate["validity"]),
            "unique_valid_parents": int(counts["unique_valid_parents"]),
            "uniqueness_among_valid_parents": float(
                aggregate["uniqueness_among_valid_parents"]
            ),
            "accepted_unique_parents": int(counts["accepted_for_preparation"]),
            "accepted_fraction": float(aggregate["accepted_fraction"]),
            "duplicate_or_other_rejected_rows": int(counts["rejected_rows"]),
            "multifragment_valid_structures": int(
                counts["multifragment_valid_structures"]
            ),
            "parent_extractions": int(counts["parent_extractions"]),
            "rejection_reasons": counts["rejection_reasons"],
            "exact_known_active_parent_rediscoveries": int(
                accepted["exact_known_active_parent"].sum()
            ),
            "unique_nonempty_bemis_murcko_scaffolds": int(nonempty_scaffolds.nunique()),
            "acyclic_accepted_parents": int(
                accepted["bemis_murcko_scaffold"].eq("").sum()
            ),
            "training_novelty": {
                "available": False,
                "reason": (
                    "No authenticated molecule/scaffold index for the Molexar base "
                    "and SAIR/PLINDER SFT training rows is available; training-set "
                    "novelty and exact DYRK1A target disjointness are not guessed."
                ),
            },
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


def _render_report(profile: dict) -> str:
    quality = profile["generation_quality"]
    sampling = profile["sampling"]
    pocket = sampling["pocket_graph"]
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
    novelty_text = html.escape(quality["training_novelty"]["reason"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gen3 full-cohort analysis</title>
<style>{report_css()}</style></head>
<body>{report_toolbar("Gen3 full-cohort audit")}<h1>Gen3: pocket-aware Molexar</h1>
<p>This section covers all {sampling['raw_samples']:,} raw draws before screening selection. The frozen model received only the 7O7K protein pocket centered on the locked docking site: {pocket['atoms_used']} of {pocket['atoms_within_radius_before_truncation']} in-radius non-hydrogen atoms under the official 425-atom limit. No ligand, sequence embedding, molecular property, gate result, or docking score was supplied.</p>
<div class="metrics"><div class="metric">Raw draws<strong>{sampling['raw_samples']:,}</strong></div><div class="metric">Converted<strong>{sampling['fragment_selfies_conversion_successes']:,}</strong></div><div class="metric">Validity<strong>{100*quality['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100*quality['uniqueness_among_valid_parents']:.2f}%</strong></div><div class="metric">Accepted / raw<strong>{100*quality['accepted_fraction']:.2f}%</strong></div><div class="metric">Unique scaffolds<strong>{quality['unique_nonempty_bemis_murcko_scaffolds']:,}</strong></div><div class="metric">Known-active rediscoveries<strong>{quality['exact_known_active_parent_rediscoveries']:,}</strong></div></div>
<div class="card"><h2>Representation conversion</h2><p>Fragment-SELFIES conversion failures: {sampling['fragment_selfies_conversion_failures']:,}; empty transported outputs: {sampling['empty_raw_smiles']:,}. No invalid or duplicate draw was replaced.</p></div>
<div class="card"><h2>Training-novelty boundary</h2><p>{novelty_text}</p></div>
<h2>Accepted-parent chemistry</h2><p>These are descriptive outcomes on every unique valid parent. No property is used as a post-generation filter and no composite score is created.</p><div class="card">{property_table}</div>
</body></html>"""


def write_gen3_analysis(
    sample_dir: Path,
    intake_dir: Path,
    actives_path: Path,
    outdir: Path,
) -> Path:
    accepted, profile = build_gen3_profile(sample_dir, intake_dir, actives_path)
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen3AnalysisError(f"Gen3 analysis output already exists: {outdir}")
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
    report_path.write_text(_render_report(profile), encoding="utf-8")
    runtime.write_json_atomic(
        outdir / "run_log.json",
        {
            "schema_version": GEN3_ANALYSIS_SCHEMA_VERSION,
            "stage": "gen3_full_cohort_analysis_run_log",
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
