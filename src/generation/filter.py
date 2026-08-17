"""Apply the arm-agnostic pre-dock instrument-range gate.

Known weaknesses:

1. Rule 1 is bent, not obeyed, for size. Heavy-atom count correlates strongly
   (rho approximately 0.95) with molecular weight, which is a reported
   property. Filtering on size therefore truncates a reported distribution.
   The mitigation is that the truncation points (10 and 62) lie entirely
   outside the observed active range (12--46), so the reported distribution
   within the region of scientific interest is unaffected. Exclusions are
   reported per gate.
2. The bounds are mildly target-informed. They are derived from DYRK1A actives,
   so a legitimately dockable 70-heavy-atom molecule is excluded for reasons
   unrelated to its quality. The mitigation is generous headroom plus per-arm
   exclusion reporting, so a large loss to the size gate is visible.
3. ``CalcNumRotatableBonds`` is a proxy, not the true torsion count. Meeko's
   active torsion count in the final PDBQT can differ because of amide handling,
   ring systems, and ``RIGID_MACROCYCLES = True``. The RDKit definition is used
   because it is available before preparation; residual mismatch is recorded
   as a Tier-2 preparation failure.
4. The torsion-degradation claim needs a literature anchor before publication.
   The qualitative statement is uncontroversial, but a citable quantitative
   source has not yet been recorded and remains an open item.
5. Two DYRK1A actives (CHEMBL4288096, CHEMBL5176894) contain silicon and
   are excluded by the element gate. They remain in the reference actives file
   and in the harness-validation docking cohort, where they will receive
   unparameterized Smina scores. At 2/1219 (0.16%) this is recorded as a
   limitation rather than corrected, because editing the actives file would
   invalidate the decoy set's hash chain.

Scope: why the validation cohort is not gated.

The gate applies to the four generative arms only. The actives/decoys
harness-validation cohort does not pass through it: the size and torsion bounds
were derived from the actives and the decoys are property-matched to those same
actives, so gating that cohort would be circular and uninformative.

This is an instrument-range gate, not a drug-likeness filter. It deliberately
does not compute or gate on QED, SA score, Lipinski compliance, structural
alerts, molecular weight, TPSA, logP, hydrogen bonding, rings, charge,
similarity, or novelty.

Usage:
    python -m src.generation.filter INPUT.smi OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from src.harness import runtime

from . import filter_config


GATE_DECISIONS_NAME = "gate_decisions.csv"
GATE_PASS_NAME = "gate_pass.smi"
GATE_SUMMARY_NAME = "gate_summary.json"

DECISION_COLUMNS = (
    "molecule_id",
    "parent_smiles",
    "heavy_atoms",
    "rotatable_bonds",
    "disallowed_elements",
    "pass_elements",
    "pass_size",
    "pass_torsions",
    "passed",
)


###### Appended to the end of the file by the user for context ######

INTERPRETATION = (
    "This gate declares the range over which Smina scores are treated as "
    "measurements rather than artifacts. It is not a drug-likeness filter. "
    "Exclusion counts are a reported per-arm metric."
)

COUNTING_NOTE = (
    "Gate exclusion counts are independent and may sum to more than "
    "n_submitted - n_passed because one molecule can fail multiple gates."
)


class GateInputError(ValueError):
    """Raised when post-intake gate input is missing, malformed, or ambiguous."""


@dataclass(frozen=True)
class GateDecision:
    molecule_id: str                # the molecule's verdict
    parent_smiles: str
    heavy_atoms: int
    rotatable_bonds: int
    disallowed_elements: tuple[str, ...]
    pass_elements: bool
    pass_size: bool
    pass_torsions: bool

    @property                       # passing has to agree with the boolean fields
    def passed(self) -> bool:
        return self.pass_elements and self.pass_size and self.pass_torsions

    def csv_row(self) -> dict:
        return {
            "molecule_id": self.molecule_id,
            "parent_smiles": self.parent_smiles,
            "heavy_atoms": self.heavy_atoms,
            "rotatable_bonds": self.rotatable_bonds,
            "disallowed_elements": ";".join(self.disallowed_elements),
            "pass_elements": self.pass_elements,
            "pass_size": self.pass_size,
            "pass_torsions": self.pass_torsions,
            "passed": self.passed,
        }




##### The whole run, frozen and reproducible #####

@dataclass(frozen=True)
class GateResult:
    decisions: tuple[GateDecision, ...]
    filter_schema_version: int
    allowed_elements: frozenset[str]
    min_heavy_atoms: int
    max_heavy_atoms: int
    max_rotatable_bonds: int

    @property
    def passed_records(self) -> tuple[tuple[str, str], ...]:    # filter survivors  and return id, smiles
        return tuple(
            (decision.molecule_id, decision.parent_smiles)
            for decision in self.decisions
            if decision.passed
        )

    @property
    def exclusions_by_gate(self) -> dict[str, int]:
        return {
            "elements": sum(not decision.pass_elements for decision in self.decisions),
            "size_low": sum(
                decision.heavy_atoms < self.min_heavy_atoms
                for decision in self.decisions
            ),
            "size_high": sum(
                decision.heavy_atoms > self.max_heavy_atoms
                for decision in self.decisions
            ),
            "torsions": sum(
                decision.rotatable_bonds > self.max_rotatable_bonds
                for decision in self.decisions
            ),
        }

    @property
    def disallowed_element_histogram(self) -> dict[str, int]:   # count how manytimes each disallowed element appears in the decisions
        histogram: Counter[str] = Counter()
        for decision in self.decisions:
            histogram.update(decision.disallowed_elements)
        return dict(sorted(histogram.items()))


def _config_values(config_module: ModuleType) -> tuple[int, frozenset[str], int, int, int]:
    return (
        int(config_module.FILTER_SCHEMA_VERSION),
        frozenset(config_module.ALLOWED_ELEMENTS),
        int(config_module.MIN_HEAVY_ATOMS),
        int(config_module.MAX_HEAVY_ATOMS),
        int(config_module.MAX_ROTATABLE_BONDS),
    )






####### The only function that makes a scientific descision ########

def apply_gate(
    records: Iterable[tuple[str, str]],
    config_module: ModuleType,
) -> GateResult:
    """Evaluate every gate independently for every post-intake parent record."""

    (
        schema_version,                         # For each (id, smiles) pair:
        allowed_elements,                               # Reject blank IDs or blank SMILES outright.
        min_heavy_atoms,                                # Parse the SMILES with RDKit
        max_heavy_atoms,                                # Count heavy atoms, count rotatable bonds, collect the set of element symbols
        max_rotatable_bonds,                            # Build the GateDecision with all three gates evaluated
    ) = _config_values(config_module)
    decisions = []

    for molecule_id, parent_smiles in records:
        molecule_id = str(molecule_id)
        parent_smiles = str(parent_smiles)
        if not molecule_id or not parent_smiles:
            raise GateInputError("Every gate record requires a molecule_id and parent_smiles")
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(parent_smiles)
        if molecule is None:
            raise GateInputError(
                f"Post-intake parent no longer parses for molecule {molecule_id}"
            )

        heavy_atoms = int(molecule.GetNumHeavyAtoms())
        rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(molecule))
        elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        disallowed_elements = tuple(sorted(elements - allowed_elements))

        decisions.append(
            GateDecision(
                molecule_id=molecule_id,
                parent_smiles=parent_smiles,
                heavy_atoms=heavy_atoms,
                rotatable_bonds=rotatable_bonds,
                disallowed_elements=disallowed_elements,
                pass_elements=not disallowed_elements,
                pass_size=(
                    min_heavy_atoms <= heavy_atoms <= max_heavy_atoms
                ),
                pass_torsions=rotatable_bonds <= max_rotatable_bonds,
            )
        )

    return GateResult(
        decisions=tuple(decisions),
        filter_schema_version=schema_version,
        allowed_elements=allowed_elements,
        min_heavy_atoms=min_heavy_atoms,
        max_heavy_atoms=max_heavy_atoms,
        max_rotatable_bonds=max_rotatable_bonds,
    )







##########    Reads the input .smi and splits each line into SMILES + ID.   ##########

def _read_records(input_path: Path) -> list[tuple[str, str]]:
    records = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise GateInputError(
                f"{input_path}:{line_number} is not SMILES<space>molecule_id"
            )
        parent_smiles, molecule_id = parts
        records.append((molecule_id, parent_smiles))
    return records


def _write_decisions(path: Path, result: GateResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(DECISION_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(decision.csv_row() for decision in result.decisions)
    temporary.replace(path)


def _write_pass(path: Path, result: GateResult) -> None:
    text = "".join(
        f"{parent_smiles} {molecule_id}\n"
        for molecule_id, parent_smiles in result.passed_records
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)




##### the orchestrator #####


def run_gate(input_path: Path, output_dir: Path) -> dict:
    """Run the frozen gate and write its immutable, provenance-bearing outputs."""

    started = time.perf_counter()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise GateInputError(f"Gate input file does not exist: {input_path}")
    if output_dir.exists():
        raise GateInputError(f"Gate output directory already exists: {output_dir}")

    records = _read_records(input_path)
    result = apply_gate(records, filter_config)

    output_dir.mkdir(parents=True)
    decisions_path = output_dir / GATE_DECISIONS_NAME
    pass_path = output_dir / GATE_PASS_NAME
    _write_decisions(decisions_path, result)
    _write_pass(pass_path, result)

    n_submitted = len(result.decisions)
    n_passed = len(result.passed_records)
    config_path = Path(filter_config.__file__).resolve()
    summary = {
        "stage": "predock_gate",
        "schema_version": runtime.RUNTIME_SCHEMA_VERSION,
        "filter_schema_version": result.filter_schema_version,
        "filter_config_sha256": runtime.sha256_file(config_path),
        "constants": {
            "allowed_elements": sorted(result.allowed_elements),
            "min_heavy_atoms": result.min_heavy_atoms,
            "max_heavy_atoms": result.max_heavy_atoms,
            "max_rotatable_bonds": result.max_rotatable_bonds,
        },
        "n_submitted": n_submitted,
        "n_passed": n_passed,
        "pass_fraction": n_passed / n_submitted if n_submitted else 0.0,
        "exclusions_by_gate": result.exclusions_by_gate,
        "counting_note": COUNTING_NOTE,
        "disallowed_element_histogram": result.disallowed_element_histogram,
        "input": runtime.file_record(input_path),
        "outputs": {
            "gate_decisions_csv": runtime.file_record(decisions_path),
            "gate_pass_smi": runtime.file_record(pass_path),
        },
        "hardware": runtime.hardware_record(),
        "timing": runtime.timing_record(
            started,
            attempted_tasks=n_submitted,
            workers=1,
        ),
        "interpretation": {
            "statement": INTERPRETATION,
        },
    }
    runtime.write_json_atomic(output_dir / GATE_SUMMARY_NAME, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen pre-dock Smina instrument-range gate"
    )
    parser.add_argument("input_smi", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    summary = run_gate(args.input_smi, args.output_dir)
    print(
        f"Pre-dock gate complete: {summary['n_passed']:,}/"
        f"{summary['n_submitted']:,} passed"
    )
    print(f"Outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
