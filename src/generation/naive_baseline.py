"""Generate reproducible uniform or property-matched random baselines.

The ``uniform`` variant samples unique, RDKit-valid parent structures without
property or drug-likeness filtering. The ``property_matched`` variant matches
the DYRK1A actives on molecular weight, cLogP, hydrogen-bond donors,
hydrogen-bond acceptors, and rotatable bonds, with exact formal-charge
matching. It intentionally contains no fingerprint, similarity, or Tanimoto
screen.

The source is caller-supplied so a future training-set holdout can replace
ChEMBL without changing this module. Versioned ChEMBL ``chemreps`` filenames
are recognized for provenance, but resolving and downloading the current
``latest/`` release remains a separate acquisition step.

Usage:
    python -m src.generation.naive_baseline \
      --mode uniform \
      --source-file chembl_<version>_chemreps.txt.gz \
      --source-description "ChEMBL <version> bulk chemreps, full set, no filtering" \
      --actives data/reference/dyrk1a_actives_chembl.csv \
      --n 10000 \
      --seed 20260801 \
      --outdir data/generated/naive_uniform_seed20260801
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, inchi
from scipy.spatial import cKDTree

from src.harness import runtime
from src.harness.intake import _largest_fragment_parent


BASELINE_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_SIZE = 10_000
PROPERTY_NEIGHBORHOOD_SIZE = 500
PROPERTY_RETRY_CEILING = 100_000
DECLARED_SEEDS = (20260801, 20260802, 20260803)

MOLECULES_NAME = "molecules.smi"
MATCHED_PAIRS_NAME = "matched_pairs.csv"
SELECTION_NAME = "selection.json"

CONTINUOUS_MATCH_COLUMNS = (
    "molecular_weight",
    "clogp",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
)
MATCH_COLUMNS = CONTINUOUS_MATCH_COLUMNS + ("formal_charge",)
VALID_MODES = ("uniform", "property_matched")

CHEMBL_CITATION = {
    "resource": "ChEMBL",
    "title": (
        "The ChEMBL Database in 2023: a drug discovery platform spanning "
        "multiple bioactivity data types and time periods"
    ),
    "doi": "10.1093/nar/gkad1004",
    "url": "https://www.ebi.ac.uk/chembl/",
}

INTERPRETATION_CAVEAT = (
    "This baseline is a chemical-space control, not a training-set holdout. "
    "REINVENT's prior saw ChEMBL; MOSES-family gen1 models saw ZINC. A true "
    "holdout library unavailable to all three models remains an open project "
    "decision."
)

_CHEMBL_CHEMREPS = re.compile(
    r"^chembl_(?P<version>[0-9]+)_chemreps[.]txt(?:[.]gz)?$",
    re.IGNORECASE,
)


class NaiveBaselineError(ValueError):
    """Raised when an immutable baseline run cannot be completed exactly."""


@dataclass(frozen=True)
class Active:
    molecule_id: str
    parent_smiles: str
    inchikey: str
    properties: tuple[float, float, int, int, int, int]


@dataclass(frozen=True)
class Candidate:
    source_id: str
    parent_smiles: str
    inchikey: str
    source_path: str
    source_line: int
    properties: tuple[float, float, int, int, int, int] | None


@dataclass(frozen=True)
class Match:
    candidate_index: int
    active_index: int | None
    scaled_property_distance: float | None


def _parse_parent(smiles: str) -> tuple[Chem.Mol, str] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    parent, parent_smiles, _ = _largest_fragment_parent(molecule)
    return parent, parent_smiles


def _inchikey(molecule: Chem.Mol) -> str:
    try:
        with rdBase.BlockLogs():
            return str(inchi.MolToInchiKey(molecule) or "")
    except Exception:
        return ""


def _match_properties(
    molecule: Chem.Mol,
) -> tuple[float, float, int, int, int, int]:
    return (
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        int(Lipinski.NumHDonors(molecule)),
        int(Lipinski.NumHAcceptors(molecule)),
        int(Lipinski.NumRotatableBonds(molecule)),
        int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
    )


def _load_actives(path: Path) -> list[Active]:
    path = Path(path)
    if not path.is_file():
        raise NaiveBaselineError(f"Actives CSV does not exist: {path}")

    actives = []
    seen_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"molecule_chembl_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise NaiveBaselineError(
                "Actives CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, 2):
            molecule_id = str(row["molecule_chembl_id"]).strip()
            smiles = str(row["canonical_smiles"]).strip()
            if not molecule_id or molecule_id in seen_ids:
                raise NaiveBaselineError(
                    f"Actives CSV has a blank or duplicate ID at row {row_number}"
                )
            parsed = _parse_parent(smiles)
            if parsed is None:
                raise NaiveBaselineError(
                    f"Active {molecule_id} does not parse and sanitize"
                )
            parent, parent_smiles = parsed
            active_inchikey = _inchikey(parent)
            if not active_inchikey:
                raise NaiveBaselineError(
                    f"Could not compute parent InChIKey for active {molecule_id}"
                )
            seen_ids.add(molecule_id)
            actives.append(
                Active(
                    molecule_id=molecule_id,
                    parent_smiles=parent_smiles,
                    inchikey=active_inchikey,
                    properties=_match_properties(parent),
                )
            )
    if not actives:
        raise NaiveBaselineError("Actives CSV contains no active molecules")
    return actives


def _open_source(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(encoding="utf-8-sig", newline="")


def _iter_source_rows(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield line number, SMILES, and source ID from chemreps TSV or .smi."""

    with _open_source(path) as handle:
        header_smiles_index = None
        header_id_index = None
        format_decided = False

        for line_number, raw_line in enumerate(handle, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if not format_decided:
                tab_fields = raw_line.rstrip("\r\n").split("\t")
                normalized_header = [field.strip().lower() for field in tab_fields]
                if "canonical_smiles" in normalized_header:
                    header_smiles_index = normalized_header.index("canonical_smiles")
                    header_id_index = (
                        normalized_header.index("chembl_id")
                        if "chembl_id" in normalized_header
                        else None
                    )
                    format_decided = True
                    continue
                format_decided = True

            if header_smiles_index is not None:
                fields = raw_line.rstrip("\r\n").split("\t")
                if len(fields) <= header_smiles_index:
                    raise NaiveBaselineError(
                        f"{path}:{line_number} is missing canonical_smiles"
                    )
                smiles = fields[header_smiles_index].strip()
                if header_id_index is not None and len(fields) > header_id_index:
                    source_id = fields[header_id_index].strip()
                else:
                    source_id = f"SOURCE_L{line_number}"
            else:
                fields = stripped.split()
                if len(fields) not in (1, 2):
                    raise NaiveBaselineError(
                        f"{path}:{line_number} is not SMILES[<space>ID]"
                    )
                smiles = fields[0]
                source_id = (
                    fields[1] if len(fields) == 2 else f"SOURCE_L{line_number}"
                )

            if not smiles:
                raise NaiveBaselineError(
                    f"{path}:{line_number} contains a blank SMILES field"
                )
            if not source_id:
                source_id = f"SOURCE_L{line_number}"
            yield line_number, smiles, source_id


def _read_candidates(
    source_path: Path,
    *,
    active_inchikeys: set[str],
    compute_properties: bool,
) -> tuple[list[Candidate], dict]:
    source_path = Path(source_path)
    if not source_path.is_file():
        raise NaiveBaselineError(f"Source file does not exist: {source_path}")

    candidates = []
    seen_parents = set()
    counts = {
        "source_rows": 0,
        "rdkit_parse_or_sanitize_failed": 0,
        "duplicate_parent": 0,
        "excluded_active_inchikey": 0,
        "missing_inchikey": 0,
        "deduplicated_eligible_pool": 0,
    }

    for line_number, smiles, source_id in _iter_source_rows(source_path):
        counts["source_rows"] += 1
        if counts["source_rows"] % 100_000 == 0:
            print(
                f"  source: {counts['source_rows']:,} rows | "
                f"{len(candidates):,} eligible unique parents",
                flush=True,
            )
        parsed = _parse_parent(smiles)
        if parsed is None:
            counts["rdkit_parse_or_sanitize_failed"] += 1
            continue
        parent, parent_smiles = parsed
        if parent_smiles in seen_parents:
            counts["duplicate_parent"] += 1
            continue
        seen_parents.add(parent_smiles)

        candidate_inchikey = _inchikey(parent)
        if not candidate_inchikey:
            counts["missing_inchikey"] += 1
        if candidate_inchikey and candidate_inchikey in active_inchikeys:
            counts["excluded_active_inchikey"] += 1
            continue

        candidates.append(
            Candidate(
                source_id=source_id,
                parent_smiles=parent_smiles,
                inchikey=candidate_inchikey,
                source_path=str(source_path.resolve()),
                source_line=line_number,
                properties=(
                    _match_properties(parent) if compute_properties else None
                ),
            )
        )

    counts["deduplicated_eligible_pool"] = len(candidates)
    return candidates, counts


def _active_feature_array(actives: list[Active]) -> np.ndarray:
    return np.asarray([active.properties for active in actives], dtype=float)


def _matching_scales(active_features: np.ndarray) -> np.ndarray:
    """Return raw sample SDs for the five continuous active properties."""

    if active_features.ndim != 2 or active_features.shape[1] != len(MATCH_COLUMNS):
        raise NaiveBaselineError(
            f"Expected active feature matrix with {len(MATCH_COLUMNS)} columns"
        )
    if len(active_features) < 2:
        raise NaiveBaselineError(
            "Property matching requires at least two active molecules"
        )
    scales = np.std(active_features[:, :-1], axis=0, ddof=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise NaiveBaselineError(
            "Every continuous active property must have a finite, positive "
            "sample standard deviation; no minimum-scale floor is applied"
        )
    return scales


def _uniform_selection(
    candidates: list[Candidate],
    *,
    n: int,
    rng: np.random.Generator,
) -> list[Match]:
    indices = np.asarray(
        rng.choice(len(candidates), size=n, replace=False),
        dtype=int,
    ).reshape(-1)
    return [
        Match(
            candidate_index=int(candidate_index),
            active_index=None,
            scaled_property_distance=None,
        )
        for candidate_index in indices
    ]


def _property_matched_selection(
    actives: list[Active],
    candidates: list[Candidate],
    *,
    n: int,
    rng: np.random.Generator,
    scales: np.ndarray,
    neighborhood_size: int = PROPERTY_NEIGHBORHOOD_SIZE,
    retry_ceiling: int = PROPERTY_RETRY_CEILING,
) -> tuple[list[Match], dict]:
    if neighborhood_size < 1:
        raise NaiveBaselineError("Property neighborhood size must be positive")
    if retry_ceiling < 1:
        raise NaiveBaselineError("Property retry ceiling must be positive")

    active_features = _active_feature_array(actives)
    candidate_features = np.asarray(
        [candidate.properties for candidate in candidates],
        dtype=float,
    )
    active_charges = {int(value) for value in active_features[:, -1]}
    same_charge_indices = np.flatnonzero(
        np.isin(candidate_features[:, -1], list(active_charges))
    )
    if len(same_charge_indices) < n:
        raise NaiveBaselineError(
            f"Requested {n:,} property-matched molecules but only "
            f"{len(same_charge_indices):,} of {len(candidates):,} "
            "deduplicated eligible pool parents have a formal charge present "
            f"among {len(actives):,} actives"
        )

    class_indices: dict[int, np.ndarray] = {}
    trees: dict[int, cKDTree] = {}
    for charge in sorted(active_charges):
        indices = np.flatnonzero(candidate_features[:, -1] == charge)
        if not len(indices):
            continue
        class_indices[charge] = indices
        trees[charge] = cKDTree(candidate_features[indices, :-1] / scales)

    used = set()
    matches = []
    consecutive_retries = 0
    total_retries = 0

    while len(matches) < n:
        active_index = int(rng.integers(len(actives)))
        charge = int(active_features[active_index, -1])
        indices = class_indices.get(charge)
        tree = trees.get(charge)
        if indices is None or tree is None:
            consecutive_retries += 1
            total_retries += 1
        else:
            k = min(neighborhood_size, len(indices))
            distances, local_indices = tree.query(
                active_features[active_index, :-1] / scales,
                k=k,
            )
            distances = np.asarray(distances, dtype=float).reshape(-1)
            local_indices = np.asarray(local_indices, dtype=int).reshape(-1)
            unused_options = [
                (int(indices[local_index]), float(distance))
                for local_index, distance in zip(local_indices, distances)
                if int(indices[local_index]) not in used
            ]
            if unused_options:
                selected_option = int(rng.integers(len(unused_options)))
                candidate_index, distance = unused_options[selected_option]
                used.add(candidate_index)
                matches.append(
                    Match(
                        candidate_index=candidate_index,
                        active_index=active_index,
                        scaled_property_distance=distance,
                    )
                )
                consecutive_retries = 0
            else:
                consecutive_retries += 1
                total_retries += 1

        if consecutive_retries >= retry_ceiling:
            raise NaiveBaselineError(
                "Property-matched selection reached the retry ceiling with "
                f"{len(matches):,}/{n:,} selected, {len(used):,} candidates "
                f"used, {len(same_charge_indices):,} same-charge candidates, "
                f"and retry_ceiling={retry_ceiling:,}"
            )

    return matches, {
        "same_charge_eligible_pool": int(len(same_charge_indices)),
        "total_retries": total_retries,
        "retry_ceiling": retry_ceiling,
    }


def _source_metadata(source_path: Path) -> dict:
    match = _CHEMBL_CHEMREPS.fullmatch(source_path.name)
    if match is None:
        return {
            "resolved_filename": source_path.name,
            "version_string": None,
            "download_url": None,
            "resolution_note": (
                "Caller-supplied swappable source; no ChEMBL release was "
                "inferred from the filename."
            ),
        }

    version = match.group("version")
    filename = f"chembl_{version}_chemreps.txt.gz"
    return {
        "resolved_filename": source_path.name,
        "version_string": f"ChEMBL {version}",
        "download_url": (
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{version}/{filename}"
        ),
        "resolution_note": (
            "Release inferred from the supplied versioned chemreps filename. "
            "The acquisition step must resolve ChEMBL latest before invoking "
            "this module."
        ),
    }


def _molecule_id(mode: str, seed: int, ordinal: int) -> str:
    variant = "U" if mode == "uniform" else "PM"
    return f"NAIVE_{variant}_{seed}_{ordinal:06d}"


def _write_molecules(
    path: Path,
    *,
    mode: str,
    seed: int,
    matches: list[Match],
    candidates: list[Candidate],
) -> list[dict]:
    rows = []
    for ordinal, match in enumerate(matches, 1):
        candidate = candidates[match.candidate_index]
        rows.append(
            {
                "molecule_id": _molecule_id(mode, seed, ordinal),
                "parent_smiles": candidate.parent_smiles,
                "candidate": candidate,
                "match": match,
            }
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            f"{row['parent_smiles']} {row['molecule_id']}\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return rows


def _write_matched_pairs(
    path: Path,
    *,
    rows: list[dict],
    actives: list[Active],
) -> None:
    columns = (
        "molecule_id",
        "parent_smiles",
        "source_id",
        "source_path",
        "source_line",
        "matched_active_id",
        "scaled_property_distance",
        *MATCH_COLUMNS,
        "matched_active_formal_charge",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            candidate = row["candidate"]
            match = row["match"]
            if candidate.properties is None or match.active_index is None:
                raise NaiveBaselineError(
                    "Property-matched audit row lacks properties or active"
                )
            active = actives[match.active_index]
            writer.writerow(
                {
                    "molecule_id": row["molecule_id"],
                    "parent_smiles": row["parent_smiles"],
                    "source_id": candidate.source_id,
                    "source_path": candidate.source_path,
                    "source_line": candidate.source_line,
                    "matched_active_id": active.molecule_id,
                    "scaled_property_distance": match.scaled_property_distance,
                    **{
                        column: candidate.properties[index]
                        for index, column in enumerate(MATCH_COLUMNS)
                    },
                    "matched_active_formal_charge": active.properties[-1],
                }
            )
    temporary.replace(path)


def generate_naive_baseline(
    *,
    mode: str,
    source_file: Path,
    source_description: str,
    actives_path: Path,
    n: int,
    seed: int,
    outdir: Path,
) -> dict:
    """Select one immutable baseline cohort and write its audit artifacts."""

    started = time.perf_counter()
    if mode not in VALID_MODES:
        raise NaiveBaselineError(
            f"mode must be one of {', '.join(VALID_MODES)}"
        )
    source_description = str(source_description).strip()
    if not source_description:
        raise NaiveBaselineError("source_description must be non-blank")
    if n < 1:
        raise NaiveBaselineError("n must be at least 1")
    if seed < 0:
        raise NaiveBaselineError("seed must be non-negative")

    source_file = Path(source_file)
    actives_path = Path(actives_path)
    outdir = Path(outdir)
    if outdir.exists():
        raise NaiveBaselineError(
            f"Baseline output directory already exists: {outdir}"
        )

    actives = _load_actives(actives_path)
    active_inchikeys = {active.inchikey for active in actives}
    print("Reading and canonicalizing baseline source...", flush=True)
    candidates, source_counts = _read_candidates(
        source_file,
        active_inchikeys=active_inchikeys,
        compute_properties=mode == "property_matched",
    )
    if len(candidates) < n:
        raise NaiveBaselineError(
            f"Requested {n:,} molecules but the deduplicated eligible pool "
            f"contains {len(candidates):,}"
        )

    rng = np.random.default_rng(seed)
    scales = None
    matching_counts = None
    if mode == "uniform":
        matches = _uniform_selection(candidates, n=n, rng=rng)
    else:
        active_features = _active_feature_array(actives)
        scales = _matching_scales(active_features)
        matches, matching_counts = _property_matched_selection(
            actives,
            candidates,
            n=n,
            rng=rng,
            scales=scales,
        )

    outdir.mkdir(parents=True)
    molecules_path = outdir / MOLECULES_NAME
    output_rows = _write_molecules(
        molecules_path,
        mode=mode,
        seed=seed,
        matches=matches,
        candidates=candidates,
    )

    output_records = {
        "molecules_smi": runtime.file_record(molecules_path),
    }
    if mode == "property_matched":
        pairs_path = outdir / MATCHED_PAIRS_NAME
        _write_matched_pairs(pairs_path, rows=output_rows, actives=actives)
        output_records["matched_pairs_csv"] = runtime.file_record(pairs_path)

    source_metadata = _source_metadata(source_file)
    source_citation = (
        CHEMBL_CITATION
        if source_metadata["version_string"] is not None
        else {
            "resource": "caller-supplied swappable source",
            "description": source_description,
            "structured_citation_available": False,
        }
    )
    matching_scales = (
        {
            column: float(scale)
            for column, scale in zip(CONTINUOUS_MATCH_COLUMNS, scales)
        }
        if scales is not None
        else None
    )
    active_charges = {active.properties[-1] for active in actives}
    charge_absent = (
        sum(
            candidate.properties is not None
            and candidate.properties[-1] not in active_charges
            for candidate in candidates
        )
        if mode == "property_matched"
        else None
    )

    selection = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "stage": "naive_baseline_selection",
        "mode": mode,
        "parameters": {
            "n": n,
            "seed": seed,
            "declared_benchmark_seeds": list(DECLARED_SEEDS),
            "sampling_without_replacement": True,
            "parent_rule": (
                "largest fragment by heavy-atom count, then molecular weight, "
                "then canonical isomeric SMILES"
            ),
            "active_exclusion_identity": "parent InChIKey",
            "matching_columns": (
                list(MATCH_COLUMNS) if mode == "property_matched" else None
            ),
            "matching_scales": matching_scales,
            "minimum_scale_floor": None,
            "formal_charge_matching": (
                "exact" if mode == "property_matched" else None
            ),
            "property_neighborhood_size": (
                PROPERTY_NEIGHBORHOOD_SIZE
                if mode == "property_matched"
                else None
            ),
            "active_sampling": (
                "uniform with replacement"
                if mode == "property_matched"
                else None
            ),
            "candidate_sampling": (
                "uniform unused candidate within the active's k-nearest "
                "same-charge property neighborhood"
                if mode == "property_matched"
                else "uniform without replacement over eligible unique parents"
            ),
            "topology_or_similarity_filter": None,
        },
        "source": {
            "description": source_description,
            **source_metadata,
            "file": runtime.file_record(source_file),
            "citation": source_citation,
        },
        "actives": {
            "file": runtime.file_record(actives_path),
            "count": len(actives),
            "unique_parent_inchikeys": len(active_inchikeys),
        },
        "counts": {
            "requested": n,
            "selected_unique_parents": len(output_rows),
            "source_funnel": source_counts,
            "property_matching": matching_counts,
            "charge_absent_from_actives": charge_absent,
        },
        "exclusions": {
            "rdkit_parse_or_sanitize_failed": source_counts[
                "rdkit_parse_or_sanitize_failed"
            ],
            "duplicate_parent": source_counts["duplicate_parent"],
            "exact_active_inchikey": source_counts["excluded_active_inchikey"],
        },
        "outputs": output_records,
        "timing": runtime.timing_record(
            started,
            attempted_tasks=source_counts["source_rows"],
            workers=1,
        ),
        "hardware": runtime.hardware_record(),
        "versions": {
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
        },
        "interpretation": {
            "caveat": INTERPRETATION_CAVEAT,
            "uniform": (
                "An unfiltered random draw after RDKit validity, deterministic "
                "parent extraction, deduplication, and exact-active exclusion."
            ),
            "property_matched": (
                "A random same-charge physicochemical control with no topology "
                "or active-similarity exclusion."
            ),
        },
    }
    runtime.write_json_atomic(outdir / SELECTION_NAME, selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a uniform or property-matched naive baseline"
    )
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--source-description", required=True)
    parser.add_argument("--actives", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    selection = generate_naive_baseline(
        mode=args.mode,
        source_file=args.source_file,
        source_description=args.source_description,
        actives_path=args.actives,
        n=args.n,
        seed=args.seed,
        outdir=args.outdir,
    )
    print(
        f"Naive {selection['mode']} baseline complete: "
        f"{selection['counts']['selected_unique_parents']:,} molecules"
    )
    print(f"Outputs: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
