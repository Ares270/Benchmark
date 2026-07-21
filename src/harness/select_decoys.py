"""Select unique, property-matched presumed decoys from local SMILES pools.

This is a reproducible local alternative when a remote decoy-generation job is
unavailable.  It is inspired by DUD-E's separation of physicochemical matching
from 2D-topology dissimilarity, but its output is a project-derived set and must
not be represented as official DUD-E or DUDE-Z output.

The selector uses accepted active parents, exact formal-charge matching,
nearest-neighbour matching on MW/cLogP/HBD/HBA/rotatable bonds, and Morgan
radius-2 topology screening against every active.  It then runs molecule intake
and the pre-docking audit automatically.

BTW, for the INPUT,    you get the active CSV that has already gone through the intake pipeline
     for the POOL,     you get a  plain .smi file of normalized SMILES<space>ID (this is the 1.4M rows)
     for the EXCLUDE,  you get a plain .smi file of known actives (this is the 1.4K rows)



Usage:
    python -m src.harness.select_decoys \
        --active-intake ACTIVE_INTAKE/molecules.csv \
        --pool DUDE_SOURCE/pool.smi \
        --exclude DUDE_SOURCE/known_dude_actives.smi \
        --outdir data/external/dyrk1a_decoys_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from scipy.spatial import cKDTree

from src.analysis.chemistry import _load_intake
from . import runtime
from .decoy_audit import DEFAULT_TOPOLOGY_THRESHOLD, write_decoy_audit
from .intake import _largest_fragment_parent, run_intake


SELECTION_SCHEMA_VERSION = 1
DEFAULT_PER_ACTIVE = 50
DEFAULT_NEIGHBORS_PER_ACTIVE = 5000
DEFAULT_DISSIMILAR_FRACTION = 1.0

MATCH_COLUMNS = (
    "molecular_weight",             # These are the constants
    "clogp",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "formal_charge",
)
CONTINUOUS_MATCH_COLUMNS = MATCH_COLUMNS[:-1]
MINIMUM_SCALES = np.asarray([25.0, 0.50, 1.0, 1.0, 1.0], dtype=float)


class DecoySelectionError(ValueError):
    """Raised when a reproducible decoy selection cannot be completed."""


@dataclass(frozen=True)
class Candidate:
    source_id: str                  # a frozen record of the original pool
    parent_smiles: str              # so that the id's cant drift
    source_path: str
    source_line: int
    properties: tuple[float, float, int, int, int, int]






#################          PIPELINE FUNCTIONS          #################


#### Smiles in, RDKit mol out, with parent extraction ####

def _parse_parent(smiles: str) -> tuple[Chem.Mol, str] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    parent, parent_smiles, _ = _largest_fragment_parent(molecule)
    return parent, parent_smiles



#######   all the properties, computed once   ########

def _match_properties(molecule: Chem.Mol) -> tuple[float, float, int, int, int, int]:
    return (
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        int(Lipinski.NumHDonors(molecule)),
        int(Lipinski.NumHAcceptors(molecule)),
        int(Lipinski.NumRotatableBonds(molecule)),
        int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
    )



######    Reads known-DUD-E-actives files into a set of canonical parent SMILES    ###########

def _read_exclusions(paths: list[Path]) -> tuple[set[str], dict]:       # its a loop that fills a set
    parents = set()                                                     # de salts exclusions before adding them
    counts = {                                                                                                              
        "files": len(paths),
        "nonblank_rows": 0,
        "invalid_rows": 0,
        "unique_parent_exclusions": 0,
    }
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise DecoySelectionError(f"Exclusion SMILES file does not exist: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            1,
        ):
            stripped = line.strip()
            if not stripped:
                continue
            counts["nonblank_rows"] += 1
            parsed = _parse_parent(stripped.split()[0])
            if parsed is None:
                counts["invalid_rows"] += 1
                continue
            _, parent_smiles = parsed
            parents.add(parent_smiles)
    counts["unique_parent_exclusions"] = len(parents)
    return parents, counts























#############     INTAKE FUNNEL     #############


def _read_pool(                         # Basically streams every source row and applies cheap gates in sequence
    paths: list[Path],                  # cheapest, most decisive gates first
    *,                                                      
    excluded_parents: set[str],             # GATE 1: format. Not exactly SMILES ID? The run dies
    active_charges: set[int],               # GATE 2: parseability. A well-formed line whose SMILES is chemically nonsense gets skipped
) -> tuple[list[Candidate], dict]:                      # So the module raises on a structural defect (wrong number of columns = wrong format)
    candidates = []                                     # but tolerates a content defect (this one molecule is shit)
    seen_parents = set()                    # GATE 3: is the parent in the known-actives set?  If so, skip it
    counts = {                              # GATE 4: duplicate
        "files": len(paths),                # GATE 5: charge
        "nonblank_rows": 0,                             
        "invalid_rows": 0,                              # Every survivor becomes a Candidate
        "malformed_rows": 0,                            # and the counters tally why everything else died
        "excluded_known_parent": 0,                     # Those counts land in selection.json
        "duplicate_parent": 0,
        "charge_absent_from_actives": 0,
        "eligible_unique_parents": 0,
    }
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise DecoySelectionError(f"Pool SMILES file does not exist: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            1,
        ):
            stripped = line.strip()
            if not stripped:
                continue
            counts["nonblank_rows"] += 1
            if counts["nonblank_rows"] % 100000 == 0:
                print(
                    f"  pool: {counts['nonblank_rows']:,} source rows read | "
                    f"{len(candidates):,} eligible unique parents",
                    flush=True,
                )
            parts = stripped.split()
            if len(parts) != 2:
                counts["malformed_rows"] += 1
                raise DecoySelectionError(
                    f"{path}:{line_number} is not normalized SMILES<space>ID"
                )
            smiles, source_id = parts
            parsed = _parse_parent(smiles)
            if parsed is None:
                counts["invalid_rows"] += 1
                continue
            parent, parent_smiles = parsed
            if parent_smiles in excluded_parents:
                counts["excluded_known_parent"] += 1
                continue
            if parent_smiles in seen_parents:
                counts["duplicate_parent"] += 1
                continue
            properties = _match_properties(parent)
            if properties[-1] not in active_charges:
                counts["charge_absent_from_actives"] += 1
                continue
            seen_parents.add(parent_smiles)
            candidates.append(
                Candidate(
                    source_id=source_id,
                    parent_smiles=parent_smiles,
                    source_path=str(path.resolve()),
                    source_line=line_number,
                    properties=properties,
                )
            )
    counts["eligible_unique_parents"] = len(candidates)
    return candidates, counts

















#########   actives' six properties into a float numpy array   ##########

def _active_feature_table(actives: pd.DataFrame) -> np.ndarray:
    return actives[list(MATCH_COLUMNS)].to_numpy(float)




######  computes each continuous property's standard deviation across your actives   #######

def _matching_scales(active_features: np.ndarray) -> np.ndarray:
    if len(active_features) > 1:
        observed = np.std(active_features[:, :-1], axis=0, ddof=1)          # every value in observed is compared 
    else:                                                                   # to its corresponding minimum allowed limit
        observed = np.zeros(len(CONTINUOUS_MATCH_COLUMNS), dtype=float)
    return np.maximum(observed, MINIMUM_SCALES)                          ##### distance molecules in property space by 
                                                                         ##### scaling each property by its observed standard deviation 
                                                                         ##### (or a minimum threshold if the standard deviation is too small).
def _property_neighborhoods(                                             ##### This ensures that each property contributes appropriately 
    active_features: np.ndarray,                                         ##### to the distance metric, 
    candidates: list[Candidate],                                         ##### preventing any single property from 
    *,                                                                   ##### dominating the distance calculation due to scale differences.
    scales: np.ndarray,
    neighbors_per_active: int,                              ##  Per active, find its k nearest candidates 
) -> list[list[tuple[int, float]]]:                         ##  in scaled property space,
    candidate_features = np.asarray(                        ##  segregated by charge
        [candidate.properties for candidate in candidates], ##  only ever compares actives to candidates of the same formal charge
        dtype=float,
    )
    neighborhoods: list[list[tuple[int, float]]] = [
        [] for _ in range(len(active_features))
    ]
    for charge in sorted({int(value) for value in active_features[:, -1]}):     # Builds the spatial index on scaled candidate properties
        active_indices = np.flatnonzero(active_features[:, -1] == charge)       # then queries the k nearest for each active
        candidate_indices = np.flatnonzero(candidate_features[:, -1] == charge) # A KD-tree for turning an all pair comparison to smth more efficient
        if not len(candidate_indices):                                          # asks for up to 5,000 nearest candidates per active     
            continue
        tree = cKDTree(candidate_features[candidate_indices, :-1] / scales)
        k = min(neighbors_per_active, len(candidate_indices))
        distances, local_indices = tree.query(
            active_features[active_indices, :-1] / scales,
            k=k,
        )   
        distances = np.asarray(distances)                                        # right not properties are a soft preference
        local_indices = np.asarray(local_indices)                                # hard rejections are the known actives,
        if k == 1:                                                               # the formal charge and the tanimoto (higher = more similar)
            distances = distances.reshape(-1, 1)
            local_indices = local_indices.reshape(-1, 1)
        for row, active_index in enumerate(active_indices):
            neighborhoods[int(active_index)] = [
                (
                    int(candidate_indices[int(local_index)]),
                    float(distance),
                )
                for local_index, distance in zip(
                    local_indices[row],
                    distances[row],
                )
            ]
    return neighborhoods


























#############     TOPOLOGY FILTERING     #############

def _fingerprint_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    )


def _maximum_active_similarities(                           # max over mean
    actives: pd.DataFrame,                                  # cause we don't care whether a candidate is on-average-ish similar to the actives 
    candidates: list[Candidate],                            # if its similar to even one then its over
    candidate_indices: set[int],
) -> dict[int, float]:
    generator = _fingerprint_generator()
    active_fingerprints = []
    for row in actives.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.parent_smiles))
        if molecule is None:
            raise DecoySelectionError(
                f"Accepted active parent no longer parses: {row.molecule_id}"
            )
        active_fingerprints.append(generator.GetFingerprint(molecule))

    maxima = {}
    ordered_indices = sorted(candidate_indices)
    for position, candidate_index in enumerate(ordered_indices, 1):
        molecule = Chem.MolFromSmiles(candidates[candidate_index].parent_smiles)
        if molecule is None:
            raise DecoySelectionError(
                "Canonical candidate parent unexpectedly failed to parse"
            )
        fingerprint = generator.GetFingerprint(molecule)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            active_fingerprints,
        )
        maxima[candidate_index] = float(max(similarities))
        if position % 10000 == 0 or position == len(ordered_indices):
            print(
                f"  topology: {position:,}/{len(ordered_indices):,} "
                "shortlisted unique candidates",
                flush=True,
            )
    return maxima








def _topology_filtered_options(
    neighborhoods: list[list[tuple[int, float]]],
    candidates: list[Candidate],
    maxima: dict[int, float],
    *,
    per_active: int,
    dissimilar_fraction: float,
    max_tanimoto: float,
) -> list[list[tuple[int, float]]]:
    options = []
    for neighborhood in neighborhoods:                              # The gate: drop every candidate whose max-similarity exceeds 0.5
        below_cap = [                                               # Survivors are property-near and topology-far      
            (candidate_index, distance)
            for candidate_index, distance in neighborhood
            if maxima[candidate_index] <= max_tanimoto
        ]
        below_cap.sort(
            key=lambda item: (
                maxima[item[0]],
                item[1],
                candidates[item[0]].source_id,
            )
        )
        keep = min(
            len(below_cap),
            max(per_active, int(math.ceil(len(below_cap) * dissimilar_fraction))),
        )
        retained = below_cap[:keep]
        retained.sort(                              # there is a sort A and a sort B, 
            key=lambda item: (                      # but the first sort is by maxima, the second is by distance. 
                item[1],                            # So the final order is by property distance, then maxima similarity, then source_id
                maxima[item[0]],                    # dont change them, under construction.
                candidates[item[0]].source_id,
            )
        )
        options.append(retained)
    return options












##########   maximum bipartite matching run 50 times   ########### (for filling per active quota)

def _balanced_unique_assignment(
    actives: pd.DataFrame,
    options: list[list[tuple[int, float]]],                                        # Here, greedy naive algorithms fail to match 1:50
    candidates: list[Candidate],                                                   # The fix is to use an augmenting-path algorithm 
    *,                                                                             # When active B wants candidate X and X is taken by A
    per_active: int,                                                               # B doesn't give up — it asks A, "can you move somewhere else?"
) -> list[tuple[int, int, float]]:
    # Allocate one decoy to every active before starting the next round.  Each
    # active's options are already ordered by property distance.  The small
    # augmenting-path matcher resolves collisions without changing the rule
    # that the closest available chemistry is preferred.
    used_candidates: set[int] = set()
    assignments: list[tuple[int, int, float]] = []

    for quota_round in range(per_active):
        round_by_candidate: dict[int, tuple[int, float]] = {}
        round_by_active: dict[int, tuple[int, float]] = {}

        def assign(                                                 # Also, the round structure for balance
            active_index: int,                                      # It doesnt hand active A all its decoys before moving to B 
            visited_actives: set[int],                              # It does 50 rounds, each giving every active exactly one new decoy
            visited_candidates: set[int],
        ) -> bool:                                                      # SCARCITY SORTING   
            if active_index in visited_actives:                               # within each round, actives are processed fewest-available-options-first
                return False                                                  # Basically, serve the actives with the least room before the pool thins further
            visited_actives.add(active_index)
            for candidate_index, distance in options[active_index]:
                if (
                    candidate_index in used_candidates
                    or candidate_index in visited_candidates
                ):
                    continue
                visited_candidates.add(candidate_index)
                previous = round_by_candidate.get(candidate_index)      # ON FAILURE 
                if previous is None or assign(                                 # either increase --neighbors-per-active 
                    previous[0],                                               # or widen the pool
                    visited_actives,
                    visited_candidates,
                ):
                    round_by_candidate[candidate_index] = (
                        active_index,
                        distance,
                    )
                    round_by_active[active_index] = (
                        candidate_index,
                        distance,
                    )
                    return True
            return False

        active_order = sorted(
            range(len(actives)),
            key=lambda active_index: (
                sum(
                    candidate_index not in used_candidates
                    for candidate_index, _ in options[active_index]
                ),
                str(actives.iloc[active_index]["molecule_id"]),
            ),
        )
        for active_index in active_order:
            if not assign(active_index, set(), set()):
                active_id = str(actives.iloc[active_index]["molecule_id"])
                eligible = sum(
                    candidate_index not in used_candidates
                    for candidate_index, _ in options[active_index]
                )
                raise DecoySelectionError(
                    "Could not fill the balanced unique-decoy quota. "
                    f"Active {active_id} failed in allocation round "
                    f"{quota_round + 1}/{per_active} with {eligible} unused "
                    "eligible options. Add a broader pool or increase "
                    "--neighbors-per-active."
                )

        for active_index in range(len(actives)):
            candidate_index, distance = round_by_active[active_index]
            assignments.append((active_index, candidate_index, distance))
            used_candidates.add(candidate_index)

    return assignments
























#######    This function writes the final assignments to a CSV file                              #########
#######    and returns a list of dictionaries representing each assignment                       #########
#######    It sorts the assignments by active index, distance, and source ID,                    #########
#######    and includes relevant information about the decoy and matched active in the output    #########




def _write_assignments(
    path: Path,
    assignments: list[tuple[int, int, float]],
    actives: pd.DataFrame,
    candidates: list[Candidate],
    maxima: dict[int, float],
) -> list[dict]:
    ordered = sorted(
        assignments,
        key=lambda item: (
            item[0],
            item[2],
            candidates[item[1]].source_id,
        ),
    )
    rows = []
    per_active_rank: dict[int, int] = {}
    for output_index, (active_index, candidate_index, distance) in enumerate(
        ordered,
        1,
    ):
        per_active_rank[active_index] = per_active_rank.get(active_index, 0) + 1
        active = actives.iloc[active_index]
        candidate = candidates[candidate_index]
        row = {
            "decoy_id": f"DYRK1A_D{output_index:06d}",
            "parent_smiles": candidate.parent_smiles,
            "source_id": candidate.source_id,
            "source_path": candidate.source_path,
            "source_line": candidate.source_line,
            "matched_active_id": str(active["molecule_id"]),
            "match_rank_for_active": per_active_rank[active_index],
            "scaled_property_distance": float(distance),
            "max_active_tanimoto": float(maxima[candidate_index]),
            **{
                column: candidate.properties[index]
                for index, column in enumerate(MATCH_COLUMNS)
            },
        }
        rows.append(row)

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows


def select_decoys(
    active_intake_path: Path,
    pool_paths: list[Path],
    outdir: Path,
    *,
    exclusion_paths: list[Path] | None = None,
    per_active: int = DEFAULT_PER_ACTIVE,
    neighbors_per_active: int = DEFAULT_NEIGHBORS_PER_ACTIVE,
    dissimilar_fraction: float = DEFAULT_DISSIMILAR_FRACTION,
    max_tanimoto: float = DEFAULT_TOPOLOGY_THRESHOLD,
) -> dict:
    """Select, intake, and audit a balanced set of presumed decoys."""

    started = time.perf_counter()
    if per_active < 1:
        raise DecoySelectionError("per_active must be at least 1")
    if neighbors_per_active < per_active:
        raise DecoySelectionError(
            "neighbors_per_active must be at least per_active"
        )
    if not 0.0 < dissimilar_fraction <= 1.0:
        raise DecoySelectionError("dissimilar_fraction must be in (0, 1]")
    if not 0.0 <= max_tanimoto <= 1.0:
        raise DecoySelectionError("max_tanimoto must be between 0 and 1")
    if not pool_paths:
        raise DecoySelectionError("At least one pool SMILES file is required")

    active_intake_path = Path(active_intake_path)
    pool_paths = [Path(path) for path in pool_paths]
    exclusion_paths = [Path(path) for path in (exclusion_paths or [])]
    outdir = Path(outdir)
    if outdir.exists():
        raise DecoySelectionError(f"Selection output path already exists: {outdir}")

    actives, active_audit = _load_intake(active_intake_path, "actives")
    active_features = _active_feature_table(actives)
    active_parents = set(actives["parent_smiles"].astype(str))
    excluded_parents, exclusion_counts = _read_exclusions(exclusion_paths)
    excluded_parents.update(active_parents)
    exclusion_counts["accepted_active_parents_added"] = len(active_parents)
    exclusion_counts["total_unique_parent_exclusions"] = len(excluded_parents)

    print("Reading and canonicalizing local source pools...", flush=True)
    candidates, pool_counts = _read_pool(
        pool_paths,
        excluded_parents=excluded_parents,
        active_charges={
            int(value) for value in active_features[:, -1]
        },
    )
    if len(candidates) < len(actives) * per_active:
        raise DecoySelectionError(
            f"Only {len(candidates):,} eligible unique pool parents for "
            f"{len(actives) * per_active:,} required assignments"
        )

    scales = _matching_scales(active_features)
    print(
        f"Building property neighborhoods for {len(actives):,} actives "
        f"from {len(candidates):,} unique eligible pool parents...",
        flush=True,
    )
    neighborhoods = _property_neighborhoods(
        active_features,
        candidates,
        scales=scales,
        neighbors_per_active=neighbors_per_active,
    )
    empty = [
        str(actives.iloc[index]["molecule_id"])
        for index, neighborhood in enumerate(neighborhoods)
        if not neighborhood
    ]
    if empty:
        raise DecoySelectionError(
            "No same-charge candidates for active(s): " + ", ".join(empty[:10])
        )

    shortlisted_indices = {
        candidate_index
        for neighborhood in neighborhoods
        for candidate_index, _ in neighborhood
    }
    print(
        f"Screening {len(shortlisted_indices):,} property-shortlisted "
        "candidates against every active topology...",
        flush=True,
    )
    maxima = _maximum_active_similarities(
        actives,
        candidates,
        shortlisted_indices,
    )
    options = _topology_filtered_options(
        neighborhoods,
        candidates,
        maxima,
        per_active=per_active,
        dissimilar_fraction=dissimilar_fraction,
        max_tanimoto=max_tanimoto,
    )
    assignments = _balanced_unique_assignment(
        actives,
        options,
        candidates,
        per_active=per_active,
    )

    outdir.mkdir(parents=True)
    assignments_path = outdir / "assignments.csv"
    rows = _write_assignments(
        assignments_path,
        assignments,
        actives,
        candidates,
        maxima,
    )
    decoys_path = outdir / "decoys.smi"
    temporary_decoys = decoys_path.with_name(f".{decoys_path.name}.tmp")
    temporary_decoys.write_text(
        "".join(
            f"{row['parent_smiles']} {row['decoy_id']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary_decoys.replace(decoys_path)

    intake_dir = outdir / "intake"
    run_intake(decoys_path, intake_dir)             # once decoys have been selected they are put through
    audit_dir = outdir / "audit"                    # the same intake that the actives went through
    write_decoy_audit(
        active_intake_path,
        intake_dir / "molecules.csv",
        audit_dir,
        topology_threshold=max_tanimoto,
    )
    audit = json.loads(
        (audit_dir / "quality.json").read_text(encoding="utf-8")            # for the audit, the tanimoto is a smoke test
    )                                                                       # the real part of the audit is the property side
                                                                            # SMD (less or even to 0.10) — Standardized Mean Difference
    selection = {                                                           # and KS — Kolmogorov-Smirnov Statistic for the shape and not jsut the mean difference
        "schema_version": SELECTION_SCHEMA_VERSION,
        "stage": "local_property_matched_decoy_selection",
        "interpretation": {
            "label": "project-derived presumed DYRK1A decoys",
            "official_dude_or_dudez_output": False,
            "experimentally_proven_inactive": False,
            "no_composite_molecular_score": True,
        },
        "parameters": {
            "per_active": per_active,
            "neighbors_per_active": neighbors_per_active,
            "dissimilar_fraction_within_property_neighborhood": (
                dissimilar_fraction
            ),
            "max_active_tanimoto": max_tanimoto,
            "matching_columns": list(MATCH_COLUMNS),
            "matching_scales": {
                column: float(scale)
                for column, scale in zip(
                    CONTINUOUS_MATCH_COLUMNS,
                    scales,
                )
            },
            "formal_charge_matching": "exact",
            "fingerprint": (
                "RDKit Morgan radius 2, 2048 bits, chirality excluded"
            ),
            "assignment": (
                "unique candidate; one property-first allocation round per "
                "active with augmenting-path collision resolution"
            ),
        },
        "counts": {
            "accepted_active_parents": int(len(actives)),
            "required_decoys": int(len(actives) * per_active),
            "selected_unique_decoys": int(len(rows)),
            "property_shortlisted_unique_candidates": int(
                len(shortlisted_indices)
            ),
            "pool": pool_counts,
            "exclusions": exclusion_counts,
        },
        "active_intake_audit": active_audit,                        
        "pre_docking_audit_status": audit["status"],
        "inputs": {
            "active_intake": runtime.file_record(active_intake_path),
            "pool_files": [
                runtime.file_record(path) for path in pool_paths
            ],
            "exclusion_files": [
                runtime.file_record(path) for path in exclusion_paths
            ],
        },
        "outputs": {
            "decoys_smi": runtime.file_record(decoys_path),
            "assignments_csv": runtime.file_record(assignments_path),
            "decoy_intake_molecules": runtime.file_record(
                intake_dir / "molecules.csv"
            ),
            "pre_docking_quality": runtime.file_record(
                audit_dir / "quality.json"
            ),
            "pre_docking_report": runtime.file_record(
                audit_dir / "report.html"
            ),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
        },
    }
    runtime.write_json_atomic(outdir / "selection.json", selection)
    return selection






























def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and pre-audit local property-matched DYRK1A decoys"
    )
    parser.add_argument("--active-intake", type=Path, required=True)
    parser.add_argument(
        "--pool",
        type=Path,
        action="append",
        required=True,
        help="normalized SMILES<space>ID pool; repeat to combine pools",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=None,
        help="SMILES file whose parents must not become decoys; repeatable",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--per-active", type=int, default=DEFAULT_PER_ACTIVE)
    parser.add_argument(
        "--neighbors-per-active",
        type=int,
        default=DEFAULT_NEIGHBORS_PER_ACTIVE,
    )
    parser.add_argument(
        "--dissimilar-fraction",
        type=float,
        default=DEFAULT_DISSIMILAR_FRACTION,
    )
    parser.add_argument(
        "--max-tanimoto",
        type=float,
        default=DEFAULT_TOPOLOGY_THRESHOLD,
    )
    args = parser.parse_args()

    selection = select_decoys(
        args.active_intake,
        args.pool,
        args.outdir,
        exclusion_paths=args.exclude,
        per_active=args.per_active,
        neighbors_per_active=args.neighbors_per_active,
        dissimilar_fraction=args.dissimilar_fraction,
        max_tanimoto=args.max_tanimoto,
    )
    print(
        f"Selected {selection['counts']['selected_unique_decoys']:,} unique "
        f"presumed decoys for "
        f"{selection['counts']['accepted_active_parents']:,} actives."
    )
    print(
        "Pre-docking audit: "
        f"{selection['pre_docking_audit_status'].upper()}"
    )
    print(f"Outputs: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
