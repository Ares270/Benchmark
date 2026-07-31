"""
Build a labelled active-versus-decoy cohort for one local benchmark run.

Selects a reproducible subset of the accepted DYRK1A actives, then carries over
the decoys that select_decoys.py already assigned to exactly those actives.

The decoys are never re-sampled here. Property matching was decided once, at
selection time, against a specific parent active. Picking decoys independently
of the actives would break that pairing and quietly turn a matched benchmark
back into an unmatched one.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from . import config, runtime

# Frozen v1 cohort produced by select_decoys.py on 2026-07-19.
EXTERNAL_DIR = config.DATA_DIR / "external"
DEFAULT_ACTIVES_SMI = EXTERNAL_DIR / "dyrk1a_actives_intake_20260719" / "accepted.smi"
DEFAULT_ASSIGNMENTS = EXTERNAL_DIR / "dyrk1a_decoys_v1_20260719" / "assignments.csv"

DEFAULT_SEED = 42


class CohortError(RuntimeError):
    """Raised when the requested cohort cannot be built honestly."""


def read_actives(path: Path) -> dict[str, str]:
    """Return {molecule_id: smiles} from a two-column .smi file."""

    actives: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise CohortError(f"{path} line {lineno} has no molecule ID: {line!r}")
        smiles, mol_id = parts[0], parts[1]
        if mol_id in actives:
            raise CohortError(f"{path} line {lineno} repeats molecule ID {mol_id}")
        actives[mol_id] = smiles
    if not actives:
        raise CohortError(f"{path} contains no molecules")
    return actives


def read_assignments(path: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Return {active_id: [(rank, decoy_id, decoy_smiles), ...]} sorted by rank."""

    required = {"decoy_id", "parent_smiles", "matched_active_id", "match_rank_for_active"}
    by_active: dict[str, list[tuple[int, str, str]]] = {}
    seen_decoys: set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise CohortError(f"{path} is missing column(s): {missing}")
        for row in reader:
            decoy_id = row["decoy_id"].strip()
            if decoy_id in seen_decoys:
                raise CohortError(f"{path} repeats decoy ID {decoy_id}")
            seen_decoys.add(decoy_id)
            try:
                rank = int(row["match_rank_for_active"])
            except (TypeError, ValueError) as error:
                raise CohortError(
                    f"{path} has a non-integer match rank for {decoy_id}"
                ) from error
            by_active.setdefault(row["matched_active_id"].strip(), []).append(
                (rank, decoy_id, row["parent_smiles"].strip())
            )

    for entries in by_active.values():
        entries.sort()
    return by_active


def choose_actives(
    active_ids: list[str], n_actives: int, seed: int, *, first: bool
) -> list[str]:
    """Pick n_actives reproducibly.

    Default is a seeded random sample. Taking the first N instead would follow
    ChEMBL intake order, which tracks deposition date and therefore clusters
    chemical series -- a subset that is reproducible but not representative.
    """

    ordered = sorted(active_ids)
    if n_actives > len(ordered):
        raise CohortError(
            f"Requested {n_actives} actives but only {len(ordered)} are available"
        )
    if first:
        return ordered[:n_actives]
    return sorted(random.Random(seed).sample(ordered, n_actives))


def build_cohort(
    outdir: Path,
    *,
    n_actives: int,
    seed: int = DEFAULT_SEED,
    first: bool = False,
    decoys_per_active: int | None = None,
    actives_smi: Path = DEFAULT_ACTIVES_SMI,
    assignments_csv: Path = DEFAULT_ASSIGNMENTS,
) -> dict:
    """Write actives.smi, decoys.smi, and cohort.json into outdir."""

    outdir = Path(outdir)
    actives_smi = Path(actives_smi)
    assignments_csv = Path(assignments_csv)
    if n_actives < 1:
        raise CohortError("n_actives must be at least 1")
    if decoys_per_active is not None and decoys_per_active < 1:
        raise CohortError("decoys_per_active must be at least 1")
    for path in (actives_smi, assignments_csv):
        if not path.is_file():
            raise CohortError(f"Input does not exist: {path}")

    actives = read_actives(actives_smi)
    assignments = read_assignments(assignments_csv)

    chosen = choose_actives(list(actives), n_actives, seed, first=first)

    # An active with no assigned decoys would silently shrink the negative set.
    orphans = [mol_id for mol_id in chosen if not assignments.get(mol_id)]
    if orphans:
        raise CohortError(
            f"Selected active(s) have no assigned decoys: {', '.join(orphans[:8])}"
        )

    decoy_rows: list[tuple[str, str]] = []
    per_active: dict[str, int] = {}
    for mol_id in chosen:
        entries = assignments[mol_id]
        if decoys_per_active is not None:
            if len(entries) < decoys_per_active:
                raise CohortError(
                    f"{mol_id} has only {len(entries)} assigned decoys, "
                    f"{decoys_per_active} requested"
                )
            entries = entries[:decoys_per_active]  # best-ranked matches first
        per_active[mol_id] = len(entries)
        decoy_rows.extend((smiles, decoy_id) for _, decoy_id, smiles in entries)

    outdir.mkdir(parents=True, exist_ok=True)
    actives_out = outdir / "actives.smi"
    decoys_out = outdir / "decoys.smi"
    actives_out.write_text(
        "".join(f"{actives[mol_id]} {mol_id}\n" for mol_id in chosen), encoding="utf-8"
    )
    decoys_out.write_text(
        "".join(f"{smiles} {decoy_id}\n" for smiles, decoy_id in decoy_rows),
        encoding="utf-8",
    )

    record = {
        "stage": "build_cohort",
        "schema_version": 1,
        "selection": {
            "n_actives": len(chosen),
            "n_decoys": len(decoy_rows),
            "method": "first_n_by_sorted_id" if first else "seeded_random_sample",
            "seed": None if first else seed,
            "decoys_per_active": decoys_per_active,
            "active_ids": chosen,
            "decoys_per_active_actual": per_active,
        },
        "inputs": {
            "actives_smi": runtime.file_record(actives_smi),
            "assignments_csv": runtime.file_record(assignments_csv),
        },
        "outputs": {
            "actives_smi": runtime.file_record(actives_out),
            "decoys_smi": runtime.file_record(decoys_out),
        },
        "labelling": (
            "Decoys are project-derived, property-matched presumed decoys. "
            "They are not experimentally confirmed DYRK1A inactives."
        ),
    }
    runtime.write_json_atomic(outdir / "cohort.json", record)

    print(f"Actives: {len(chosen)} -> {actives_out}")
    print(f"Decoys:  {len(decoy_rows)} -> {decoys_out}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labelled active+assigned-decoy cohort for a local run"
    )
    parser.add_argument("outdir", type=Path, help="cohort directory to create")
    parser.add_argument("--n-actives", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--first", action="store_true",
        help="take the first N actives by sorted ID instead of sampling",
    )
    parser.add_argument(
        "--decoys-per-active", type=int, default=None,
        help="truncate to the N best-ranked assigned decoys (default: all 50)",
    )
    parser.add_argument("--actives-smi", type=Path, default=DEFAULT_ACTIVES_SMI)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    args = parser.parse_args()

    try:
        build_cohort(
            args.outdir,
            n_actives=args.n_actives,
            seed=args.seed,
            first=args.first,
            decoys_per_active=args.decoys_per_active,
            actives_smi=args.actives_smi,
            assignments_csv=args.assignments,
        )
    except CohortError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
