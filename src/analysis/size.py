"""Score-versus-size diagnostic for one validated ranking analysis.

A docking score that tracks heavy atom count is measuring how much molecule
fits in the box, not how well it binds. This module recomputes heavy atom
count from the same .smi inputs that were prepared and docked, then reports
the correlation between score and size. It answers a question about the
scoring function; it does not modify, reweight, or override any metric.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr

from .dataset import AnalysisInputError

RDLogger.DisableLog("rdApp.*")


COHORTS = ("actives", "decoys")

# src.harness.prepare_ligands.safe_name() replaces every character outside this
# class with '_' when it writes <ID>.pdbqt, and dock.py takes the molecule ID
# back from that filename. The same substitution is mirrored here so a scored
# ID can be matched to its input SMILES, without importing the harness (and its
# Meeko/OpenBabel stack) into the analysis layer.
_UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")

# Enough points for a rank correlation to mean anything at all.
_MINIMUM_PAIRS = 3




#####  Input SMILES  #####

def _safe_name(molecule_id: str) -> str:
    return _UNSAFE_ID_CHARACTERS.sub("_", molecule_id.strip())


def load_smiles(path: Path, cohort: str) -> dict[str, str]:
    """Return {molecule_id: SMILES} from a two-column "<SMILES> <ID>" file."""

    path = Path(path)
    if not path.is_file():
        raise AnalysisInputError(f"{cohort} SMILES input does not exist: {path}")

    mapping: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise AnalysisInputError(
                f"{path} line {lineno} has no molecule ID: {line!r}"
            )
        smiles, molecule_id = parts[0], parts[1]
        if molecule_id in mapping:
            raise AnalysisInputError(
                f"{path} repeats molecule ID {molecule_id} on line {lineno}"
            )
        mapping[molecule_id] = smiles
    if not mapping:
        raise AnalysisInputError(f"{path} contains no molecules")
    return mapping


def _sanitized_index(mapping: dict[str, str], path: Path) -> dict[str, str]:
    """Second lookup keyed by the filename-safe form of each input ID."""

    index: dict[str, str] = {}
    for molecule_id, smiles in mapping.items():
        key = _safe_name(molecule_id)
        if key in index and index[key] != smiles:
            raise AnalysisInputError(
                f"{path} contains IDs that collapse to the same prepared-file "
                f"name {key!r} with different SMILES"
            )
        index[key] = smiles
    return index


def _name_ids(ids: list[str], limit: int = 20) -> str:
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", ... ({len(ids)} total)"
    return shown




#####  Heavy atom counts for every analyzed molecule  #####

def heavy_atom_frame(
    docking_frame: pd.DataFrame,
    active_smi_path: Path,
    decoy_smi_path: Path,
) -> pd.DataFrame:
    """Attach an RDKit heavy atom count to every analyzed row.

    Any scored ID without a SMILES, and any SMILES RDKit cannot parse, is a
    hard error: a silently dropped molecule would change which compounds the
    correlation is computed over without saying so.
    """

    sources = {
        "actives": Path(active_smi_path),
        "decoys": Path(decoy_smi_path),
    }
    exact = {name: load_smiles(path, name) for name, path in sources.items()}
    sanitized = {
        name: _sanitized_index(exact[name], sources[name]) for name in COHORTS
    }

    frame = docking_frame.copy()
    frame["cohort"] = frame["label"].map({1: "actives", 0: "decoys"})

    smiles_column: list[str | None] = []
    unmapped: list[str] = []
    for cohort, molecule_id in zip(frame["cohort"], frame["molecule_id"].astype(str)):
        smiles = exact[cohort].get(molecule_id)
        if smiles is None:
            smiles = sanitized[cohort].get(molecule_id)
        if smiles is None:
            unmapped.append(f"{cohort}:{molecule_id}")
        smiles_column.append(smiles)
    if unmapped:
        raise AnalysisInputError(
            "Scored molecule IDs have no SMILES in the supplied .smi inputs: "
            + _name_ids(unmapped)
        )

    counts: list[int] = []
    unparsed: list[str] = []
    for cohort, molecule_id, smiles in zip(
        frame["cohort"], frame["molecule_id"].astype(str), smiles_column
    ):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            unparsed.append(f"{cohort}:{molecule_id}")
            counts.append(-1)
            continue
        counts.append(int(Descriptors.HeavyAtomCount(molecule)))
    if unparsed:
        raise AnalysisInputError(
            "RDKit cannot parse the input SMILES for: " + _name_ids(unparsed)
        )

    frame["smiles"] = smiles_column
    frame["heavy_atoms"] = counts
    if "score_imputed" not in frame.columns:
        frame["score_imputed"] = False
    frame["score_imputed"] = frame["score_imputed"].astype(bool)
    frame["score_observed"] = ~frame["score_imputed"] & frame["score"].notna()
    return frame




#####  Statistics  #####

def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _correlations(scores: np.ndarray, sizes: np.ndarray) -> dict:
    """Spearman (primary) and Pearson (secondary), each two-sided."""

    record = {
        "n": int(scores.size),
        "spearman_rho": None,
        "spearman_p_value": None,
        "pearson_r": None,
        "pearson_p_value": None,
    }
    if (
        scores.size < _MINIMUM_PAIRS
        or np.unique(scores).size < 2
        or np.unique(sizes).size < 2
    ):
        return record
    spearman = spearmanr(scores, sizes, alternative="two-sided")
    pearson = pearsonr(scores, sizes, alternative="two-sided")
    record["spearman_rho"] = _finite_or_none(spearman.statistic)
    record["spearman_p_value"] = _finite_or_none(spearman.pvalue)
    record["pearson_r"] = _finite_or_none(pearson.statistic)
    record["pearson_p_value"] = _finite_or_none(pearson.pvalue)
    return record


def _size_statistics(sizes: np.ndarray) -> dict:
    return {
        "n": int(sizes.size),
        "mean": _finite_or_none(np.mean(sizes)) if sizes.size else None,
        "median": _finite_or_none(np.median(sizes)) if sizes.size else None,
        "std": _finite_or_none(np.std(sizes, ddof=1)) if sizes.size > 1 else None,
    }


def _standardized_mean_difference(
    active_sizes: np.ndarray, decoy_sizes: np.ndarray
) -> float | None:
    """(mean_actives - mean_decoys) / pooled SD, or None when undefined."""

    n_active, n_decoy = active_sizes.size, decoy_sizes.size
    if n_active < 2 or n_decoy < 2:
        return None
    active_variance = float(np.var(active_sizes, ddof=1))
    decoy_variance = float(np.var(decoy_sizes, ddof=1))
    pooled = np.sqrt(
        ((n_active - 1) * active_variance + (n_decoy - 1) * decoy_variance)
        / (n_active + n_decoy - 2)
    )
    if not np.isfinite(pooled) or pooled == 0:
        return None
    return _finite_or_none(
        (float(np.mean(active_sizes)) - float(np.mean(decoy_sizes))) / pooled
    )




#####  Profile  #####

def build_size_profile(
    docking_frame: pd.DataFrame,
    active_smi_path: Path,
    decoy_smi_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Return per-molecule heavy atom counts and a JSON-safe size summary."""

    frame = heavy_atom_frame(docking_frame, active_smi_path, decoy_smi_path)
    observed = frame.loc[frame["score_observed"]]

    correlations = {
        "pooled": _correlations(
            observed["score"].to_numpy(float),
            observed["heavy_atoms"].to_numpy(float),
        )
    }
    for cohort in COHORTS:
        subset = observed.loc[observed["cohort"].eq(cohort)]
        correlations[cohort] = _correlations(
            subset["score"].to_numpy(float),
            subset["heavy_atoms"].to_numpy(float),
        )

    cohort_sizes = {
        cohort: frame.loc[frame["cohort"].eq(cohort), "heavy_atoms"].to_numpy(float)
        for cohort in COHORTS
    }
    profile = {
        "schema_version": 1,
        "descriptor": "RDKit Descriptors.HeavyAtomCount on the input SMILES",
        "scope": (
            "heavy atom counts cover every analyzed molecule; correlations use "
            "only molecules with an observed docking score"
        ),
        "inputs": {
            "actives_smi": str(Path(active_smi_path).resolve()),
            "decoys_smi": str(Path(decoy_smi_path).resolve()),
        },
        "n_analyzed": int(len(frame)),
        "n_correlated": int(len(observed)),
        "n_excluded_missing_score": int(len(frame) - len(observed)),
        "correlations": correlations,
        "heavy_atom_counts": {
            cohort: _size_statistics(sizes) for cohort, sizes in cohort_sizes.items()
        },
        "standardized_mean_difference_actives_minus_decoys": (
            _standardized_mean_difference(
                cohort_sizes["actives"], cohort_sizes["decoys"]
            )
        ),
    }
    return frame, profile
