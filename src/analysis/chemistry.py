"""Validate intake descriptors and summarize chemical-property cohorts.

Docking discrimination and molecular properties answer different questions. 
This module keeps them separate while placing both in one report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .dataset import AnalysisInputError


PROPERTY_SPECS = (
    ("molecular_weight", "MW (Da)", "RDKit average molecular weight"),          # THERE ARE 12 DESCRITPORS
    ("clogp", "cLogP", "RDKit Crippen MolLogP"),                                # each line represents a descriptor, split into 3 columns
    ("tpsa_a2", "TPSA (Å²)", "RDKit topological polar surface area"),                       # 1st column is the key,  
    ("hbond_donors", "H-bond donors", "RDKit Lipinski hydrogen-bond donors"),               # 2nd column is the label,
    ("hbond_acceptors", "H-bond acceptors", "RDKit Lipinski hydrogen-bond acceptors"),      # 3rd column is the method   
    ("rotatable_bonds", "Rotatable bonds", "RDKit Lipinski rotatable bonds"),
    ("ring_count", "Rings", "RDKit ring count"),
    ("aromatic_ring_count", "Aromatic rings", "RDKit aromatic ring count"),                 # property whitelist basically
    ("fraction_csp3", "Fraction Csp³", "RDKit fraction of sp3 carbons"),
    ("formal_charge", "Formal charge", "Sum of RDKit atom formal charges"),
    ("qed", "QED", "RDKit default weighted QED (descriptive; no cutoff)"),
    ("sa_score", "SA score", "RDKit Contrib SA_Score (descriptive; no cutoff)"),
)
PROPERTY_COLUMNS = tuple(spec[0] for spec in PROPERTY_SPECS)
PROPERTY_LABELS = {key: label for key, label, _ in PROPERTY_SPECS}



#### checks intake identity #####

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


##### produces a short list of bad IDs for readable errors #####

def _sample(values: pd.Series, limit: int = 8) -> str:
    return ", ".join(values.astype(str).head(limit).tolist())



#####    converts invalid numerical results to None, which becomes valid JSON null    #####

def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None












######## prove the intake file is authentic #########

def _load_intake(path: Path, cohort: str) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.is_file():
        raise AnalysisInputError(f"{cohort} intake table does not exist: {path}")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise AnalysisInputError(
            f"{cohort} intake table requires its sibling summary.json: {summary_path}"      # For each intake CSV:
        )                                                                                       # Confirm the CSV exists.
                                                                                                # Require its sibling summary.json.
    try:                                                                                        # Confirm that summary says stage=molecule_intake.
        summary = json.loads(summary_path.read_text(encoding="utf-8"))                          # Calculate the CSV’s current hash.
    except (json.JSONDecodeError, OSError) as error:                                            # Compare it with the recorded hash.
        raise AnalysisInputError(
            f"Cannot read intake summary {summary_path}: {error}"
        ) from error
    if summary.get("stage") != "molecule_intake":
        raise AnalysisInputError(f"{summary_path} is not a molecule-intake summary")

    recorded_hash = summary.get("outputs", {}).get("molecules_csv_sha256")
    if not recorded_hash or recorded_hash != _sha256(path):
        raise AnalysisInputError(
            f"{path} does not match the SHA-256 recorded in {summary_path}"
        )

    header = pd.read_csv(path, nrows=0)
    required = {
        "molecule_id",
        "parent_smiles",
        "status",
        "parent_was_extracted",
        *PROPERTY_COLUMNS,
    }
    missing = sorted(required - set(header.columns))
    if missing:
        raise AnalysisInputError(f"{path} is missing intake column(s): {missing}")

    raw = pd.read_csv(
        path, dtype={"molecule_id": "string", "parent_smiles": "string"}
    )
    status = raw["status"].fillna("").astype(str).str.strip().str.lower()
    accepted = raw.loc[status.eq("accepted")].copy()
    if accepted.empty:
        raise AnalysisInputError(f"{path} contains no rows accepted for preparation")

    accepted["molecule_id"] = accepted["molecule_id"].astype("string").str.strip()
    accepted["parent_smiles"] = (
        accepted["parent_smiles"].astype("string").str.strip()
    )
    missing_identity = (
        accepted["molecule_id"].isna()
        | accepted["molecule_id"].eq("")
        | accepted["parent_smiles"].isna()
        | accepted["parent_smiles"].eq("")
    )
    if missing_identity.any():
        raise AnalysisInputError(f"{path} has accepted rows without an ID or parent SMILES")
    duplicates = accepted.loc[
        accepted["molecule_id"].duplicated(keep=False), "molecule_id"
    ]
    if not duplicates.empty:
        raise AnalysisInputError(
            f"{path} has duplicate accepted molecule IDs: {_sample(duplicates)}"
        )

    for column in PROPERTY_COLUMNS:
        raw_values = accepted[column]
        numeric = pd.to_numeric(raw_values, errors="coerce")
        malformed = numeric.isna() | ~np.isfinite(numeric.to_numpy(float))
        if malformed.any():
            bad_ids = accepted.loc[malformed, "molecule_id"]
            raise AnalysisInputError(
                f"{path} has missing or non-finite {column} for: {_sample(bad_ids)}"
            )
        accepted[column] = numeric.astype(float)

    recorded_count = summary.get("counts", {}).get("accepted_for_preparation")
    if recorded_count != len(accepted):
        raise AnalysisInputError(
            f"{path} has {len(accepted)} accepted rows but summary.json records "
            f"{recorded_count}"
        )

    accepted["cohort"] = cohort
    accepted = accepted[
        ["molecule_id", "parent_smiles", "cohort", *PROPERTY_COLUMNS]
    ]
    counts = summary.get("counts", {})
    metrics = summary.get("aggregate_metrics", {})
    audit = {
        "molecules_csv": str(path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "submitted_rows": int(counts.get("submitted_rows", 0)),
        "valid_structures": int(counts.get("valid_structures", 0)),
        "unique_valid_parents": int(counts.get("unique_valid_parents", 0)),
        "accepted_for_preparation": int(recorded_count),
        "parent_extractions": int(counts.get("parent_extractions", 0)),
        "validity": float(metrics.get("validity", 0.0)),
        "parent_uniqueness": float(
            metrics.get("uniqueness_among_valid_parents", 0.0)
        ),
        "wall_seconds": float(summary.get("timing", {}).get("wall_seconds", 0.0)),
        "evaluation_policy": summary.get("evaluation_policy", {}),
    }
    return accepted, audit


def _property_statistics(frame: pd.DataFrame) -> dict:
    result = {}
    for column in PROPERTY_COLUMNS:
        values = frame[column].to_numpy(float)
        result[column] = {
            "n": int(values.size),
            "mean": _finite_or_none(np.mean(values)),
            "median": _finite_or_none(np.median(values)),
            "std": (
                _finite_or_none(np.std(values, ddof=1))
                if values.size > 1
                else None
            ),
            "min": _finite_or_none(np.min(values)),
            "max": _finite_or_none(np.max(values)),
        }
    return result













###### This function computes the correlation between docking scores and chemical properties for a given cohort.
###### It filters the data to include only observed scores 
###### and then calculates the Spearman correlation coefficient for each property.




def _score_correlations(joined: pd.DataFrame, cohort: str) -> dict:
    subset = joined.loc[
        joined["cohort"].eq(cohort)
        & ~joined["score_imputed"]
        & joined["score"].notna()
    ]
    rho = {}
    for column in PROPERTY_COLUMNS:
        if (
            len(subset) < 3
            or subset["score"].nunique() < 2
            or subset[column].nunique() < 2
        ):
            rho[column] = None
        else:
            value = spearmanr(subset["score"], subset[column]).statistic
            rho[column] = _finite_or_none(value)
    return {"n_observed_scores": int(len(subset)), "rho": rho}




##### this function builds a chemical profile by loading intake data for actives and decoys, 
##### merging it with docking scores, and summarizing properties and correlations. 
##### It checks for missing data and raises errors if necessary.
##### The output includes a DataFrame of properties and a JSON-safe summary of the chemical profile.


def build_chemical_profile(
    active_intake_path: Path,
    decoy_intake_path: Path,
    docking_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return accepted-parent properties and a JSON-safe chemical summary."""

    actives, active_audit = _load_intake(active_intake_path, "actives")
    decoys, decoy_audit = _load_intake(decoy_intake_path, "decoys")
    properties = pd.concat([actives, decoys], ignore_index=True)

    docking = docking_frame[["molecule_id", "label", "score"]].copy()
    docking["cohort"] = docking["label"].map({1: "actives", 0: "decoys"})
    if "score_imputed" in docking_frame:
        docking["score_imputed"] = (
            docking_frame["score_imputed"].astype(bool).to_numpy()
        )
    else:
        docking["score_imputed"] = False

    expected = set(zip(docking["cohort"], docking["molecule_id"].astype(str)))
    available = set(
        zip(properties["cohort"], properties["molecule_id"].astype(str))
    )
    missing = sorted(expected - available)
    if missing:
        examples = ", ".join(
            f"{cohort}:{mol_id}" for cohort, mol_id in missing[:8]
        )
        raise AnalysisInputError(
            "Docking rows are absent from the matching accepted intake parents: "
            + examples
        )

    joined = properties.merge(
        docking[["molecule_id", "cohort", "score", "score_imputed"]],
        on=["molecule_id", "cohort"],
        how="left",
        validate="one_to_one",
    )
    joined["score_imputed"] = joined["score_imputed"].fillna(False).astype(bool)
    cohort_frames = {
        name: properties.loc[properties["cohort"].eq(name)]
        for name in ("actives", "decoys")
    }
    cohort_summaries = {}
    for name, frame in cohort_frames.items():
        observed_scores = joined.loc[
            joined["cohort"].eq(name) & ~joined["score_imputed"], "score"
        ].notna()
        cohort_summaries[name] = {
            "n_accepted_parents": int(len(frame)),
            "n_with_observed_docking_score": int(observed_scores.sum()),
            "properties": _property_statistics(frame),
        }

    differences = {
        column: (
            cohort_summaries["actives"]["properties"][column]["mean"]
            - cohort_summaries["decoys"]["properties"][column]["mean"]
        )
        for column in PROPERTY_COLUMNS
    }
    profile = {
        "schema_version": 1,
        "scope": (
            "all intake rows accepted for preparation; descriptors are for "
            "evaluated parents"
        ),
        "property_definitions": [
            {"key": key, "label": label, "method": method}
            for key, label, method in PROPERTY_SPECS
        ],
        "intake_audit": {"actives": active_audit, "decoys": decoy_audit},
        "cohorts": cohort_summaries,
        "mean_difference_actives_minus_decoys": differences,
        "score_property_spearman": {
            name: _score_correlations(joined, name)
            for name in ("actives", "decoys")
        },
    }
    return properties, profile
