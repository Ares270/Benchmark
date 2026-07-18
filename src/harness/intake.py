"""Validate and characterize a two-column molecular submission.

The intake stage sits before 3D preparation and docking.  It answers three
different questions without blending them into one score:

1. Can RDKit parse and sanitize the submitted structure?
2. Is the canonical isomeric structure unique within this submission?
3. What transparent 2D properties does the submitted structure have?

Usage:
    python -m src.harness.intake INPUT.smi OUTPUT_DIR

Input format:
    one ``SMILES<whitespace>molecule_id`` pair per non-blank line, no header.

Outputs:
    molecules.csv  - every non-blank submitted row, including rejected rows
    accepted.smi   - valid, unique canonical structures ready for preparation
    summary.json   - aggregate validity/uniqueness, provenance, and stage timing

QED is the default weighted QED from Bickerton et al. (2012),
doi:10.1038/nchem.1243.  SA score is the RDKit Contrib implementation of the
Ertl-Schuffenhauer heuristic (2009), doi:10.1186/1758-2946-1-8.  SA score is a
heuristic difficulty estimate, not evidence that a synthesis route exists.
"""     
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from types import ModuleType

import rdkit
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


class IntakeInputError(ValueError):                                             # intake specific readable error creation
    """Raised when the intake file or requested output is ambiguous."""


SCHEMA_VERSION = 1
MOLECULES_NAME = "molecules.csv"                 # 3 fixed filenames
ACCEPTED_NAME = "accepted.smi"
SUMMARY_NAME = "summary.json"

# IDs become filenames during ligand preparation.  Reject unsafe IDs here
# instead of silently rewriting them and risking filename collisions.
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")          ###### filename sanitization

CSV_COLUMNS = [
    "line_number",          ##################3
    "molecule_id",
    "input_smiles",      
    "canonical_smiles",    # Administrative columns
    "parent_smiles",         #
    "parent_was_extracted",  # new
    "status",
    "reason",
    "duplicate_of_line",
    "structure_valid",          #######################
    "fragment_count",
    "heavy_atom_count",
    "molecular_weight",
    "clogp",
    "tpsa_a2",
    "hbond_donors",         # chemical columns
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "formal_charge",
    "qed",
    "sa_score",           ###################################
]



"""

Current canonicalization policy

stereochemistry_preserved: true
salt_or_fragment_stripping: false
tautomer_standardization: false
charge_neutralization: false

"""

############   Finding the Synthetic Accesibility score   ############

def _load_sa_scorer() -> ModuleType:
    """Load RDKit's installed Contrib SA scorer without vendoring its data."""

    module_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    if not module_path.is_file():
        raise RuntimeError(
            "RDKit Contrib SA_Score is missing; expected " + str(module_path)
        )
    specification = importlib.util.spec_from_file_location(
        "_dyrk1a_rdkit_sa_scorer", module_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load RDKit SA scorer from {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_SA_SCORER = _load_sa_scorer()






#######   read a file and calculate its SHA-256 fingerprint   ########

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



####   Resumability   ####

def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

















###########    CORE CHEMISTRY SECTOR, FINDS ALL DESCRIPTORS    ###########


def _descriptors(molecule: Chem.Mol) -> dict:
    return {
        "fragment_count": len(Chem.GetMolFrags(molecule)),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt(molecule),
        "clogp": Crippen.MolLogP(molecule),
        "tpsa_a2": rdMolDescriptors.CalcTPSA(molecule),
        "hbond_donors": Lipinski.NumHDonors(molecule),
        "hbond_acceptors": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "ring_count": rdMolDescriptors.CalcNumRings(molecule),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(molecule),
        "formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        "qed": QED.qed(molecule),
        "sa_score": _SA_SCORER.calculateScore(molecule),
    }






########  new module for fragment handling  ########

def _largest_fragment_parent(molecule: Chem.Mol) -> tuple[Chem.Mol, str, bool]:
    """Choose one deterministic parent without changing tautomer, charge, or stereo."""

    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    ranked = []
    for fragment in fragments:
        smiles = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        ranked.append((fragment.GetNumHeavyAtoms(), Descriptors.MolWt(fragment), smiles, fragment))
    _, _, parent_smiles, parent = max(ranked, key=lambda item: item[:3])
    return parent, parent_smiles, len(fragments) > 1








#####   EMPTY AUDIT ROW   #######

def _blank_row(line_number: int, input_smiles: str, molecule_id: str) -> dict:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "line_number": line_number,
            "molecule_id": molecule_id,
            "input_smiles": input_smiles,
            "structure_valid": False,
        }
    )
    return row















###########      STARTING THE RUN      ###########

def run_intake(input_path: Path, output_dir: Path) -> dict:
    """Run intake once and return the JSON-serializable aggregate summary."""   

    started_wall = time.perf_counter()                          # Starts wall-clock timing
    started_cpu = time.process_time()                           # Starts CPU timing
    input_path = Path(input_path)                               # Converts supplied paths into filesystem objects
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise IntakeInputError(f"Input file does not exist: {input_path}")             
    if output_dir.exists():
        raise IntakeInputError(f"Output directory already exists: {output_dir}")      # Refuses an existing output directory
    output_dir.mkdir(parents=True)                                                    # Creates a new output directory  

    physical_lines = input_path.read_text(encoding="utf-8-sig").splitlines()          # tolerate a hidden UTF-8 byte-order marker 
    rows: list[dict] = []                                                             # (sometimes added by spreadsheet software)
    accepted: list[tuple[str, str]] = []
    seen_ids: dict[str, int] = {}
    first_structure_line: dict[str, int] = {}                       # lists and dictionaries remember
    accepted_structures: dict[str, str] = {}                        # All audit rows
    blank_lines = 0                                                 # Accepted molecules
                                                                    # Previously seen IDs
    for line_number, raw_line in enumerate(physical_lines, 1):      # First occurrence of every structure
        stripped = raw_line.strip()                                 # Structures already accepted downstream
        if not stripped:                                            # Number of blank lines
            blank_lines += 1
            continue

        parts = stripped.split()
        input_smiles = parts[0]
        molecule_id = parts[1] if len(parts) >= 2 else ""               # parse and split columns into 
        row = _blank_row(line_number, input_smiles, molecule_id)        # 1. SMILES
        reasons: list[str] = []                                         # 2. molecule_id
                                                                        # raise error if less or more than 2
        if len(parts) == 1:
            reasons.append("missing_molecule_id")
        elif len(parts) > 2:
            reasons.append("unexpected_columns")

        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(input_smiles)                 # molecular parsing and canonicalization
        if molecule is None:
            reasons.append("rdkit_parse_or_sanitize_failed")
        else:
            canonical = Chem.MolToSmiles(                                               # If parsing succeeds:
                molecule, canonical=True, isomericSmiles=True                                    # RDKit generates one canonical spelling
            )                                                                                    # Stereochemistry is retained
            parent, parent_smiles, parent_was_extracted = _largest_fragment_parent(molecule)     # strip to biggest fragment
            row["canonical_smiles"] = canonical                                                  # full structure, as submitted
            row["parent_smiles"] = parent_smiles                                                 # desalted parent, what we actually evaluate
            row["parent_was_extracted"] = parent_was_extracted                                   # True if we threw away a fragment
            row["structure_valid"] = True                                                        # The molecule is marked structurally valid
            row.update(_descriptors(parent))                                                     # Descriptors computed on the PARENT, not the salt
            row["fragment_count"] = len(Chem.GetMolFrags(molecule))                              # but remember the input's real fragment count
            if parent_smiles in first_structure_line:                                            # dedup keys on parent now
                row["duplicate_of_line"] = first_structure_line[parent_smiles]
            else:
                first_structure_line[parent_smiles] = line_number

        if molecule_id:                                                     # quick id validation
            if _SAFE_ID.fullmatch(molecule_id) is None:
                reasons.append("unsafe_molecule_id")
            elif molecule_id in seen_ids:
                reasons.append(f"duplicate_molecule_id_line_{seen_ids[molecule_id]}")
            else:
                seen_ids[molecule_id] = line_number

        parent_smiles = row["parent_smiles"]
        if not reasons and parent_smiles:
            if parent_smiles in accepted_structures:
                reasons.append(
                    f"duplicate_structure_of_{accepted_structures[parent_smiles]}"
                )
            else:
                accepted_structures[parent_smiles] = molecule_id
                accepted.append((parent_smiles, molecule_id))

        row["status"] = "accepted" if not reasons else "rejected"
        row["reason"] = ";".join(reasons)
        rows.append(row)

    molecules_path = output_dir / MOLECULES_NAME
    accepted_path = output_dir / ACCEPTED_NAME
    _write_csv_atomic(molecules_path, rows)
    accepted_text = "".join(f"{smiles} {molecule_id}\n" for smiles, molecule_id in accepted)
    _write_text_atomic(accepted_path, accepted_text)

    n_submitted = len(rows)
    n_valid = sum(bool(row["structure_valid"]) for row in rows)
    n_unique = len(first_structure_line)
    n_accepted = len(accepted)
    rejected_reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in filter(None, str(row["reason"]).split(";")):
            rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1











###########   Bunch of Aggregate counts   ############

    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "molecule_intake",
        "input": {
            "path": str(input_path.resolve()),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
            "format": "SMILES<whitespace>molecule_id",
        },
        "outputs": {
            "molecules_csv": MOLECULES_NAME,
            "accepted_smi": ACCEPTED_NAME,
            "molecules_csv_sha256": _sha256(molecules_path),
            "accepted_smi_sha256": _sha256(accepted_path),
        },
        "counts": {                                                     # Calculate
            "physical_lines": len(physical_lines),                          # 
            "blank_lines": blank_lines,                                     # Blank lines
            "submitted_rows": n_submitted,                                  # Submitted rows
            "valid_structures": n_valid,                                    # Parseable structures
            "unique_valid_structures": n_unique,                            # Unique valid structures
            "accepted_for_preparation": n_accepted,                         # Accepted structures
            "rejected_rows": n_submitted - n_accepted,                      # Frequency of every rejection reason
            "multifragment_valid_structures": sum(
                bool(row["structure_valid"]) and int(row["fragment_count"]) > 1
                for row in rows
            ),
            "rejection_reasons": rejected_reason_counts,
        },
        "aggregate_metrics": {
            "validity": n_valid / n_submitted if n_submitted else 0.0,
            "uniqueness_among_valid": n_unique / n_valid if n_valid else 0.0,
            "accepted_fraction": n_accepted / n_submitted if n_submitted else 0.0,
        },
        "canonicalization": {
            "rdkit_canonical_smiles": True,
            "stereochemistry_preserved": True,
            "salt_or_fragment_stripping": False,
            "tautomer_standardization": False,
            "charge_neutralization": False,
        },
        "property_methods": {
            "qed": "RDKit default weighted QED; higher is more drug-like (0 to 1)",
            "sa_score": "RDKit Contrib SA_Score; lower is heuristically easier (1 to 10)",
            "clogp": "RDKit Crippen MolLogP",
            "tpsa_a2": "RDKit topological polar surface area",
            "rotatable_bonds": "RDKit Lipinski strict default",
        },
        "timing": {
            "wall_seconds": elapsed_wall,
            "cpu_seconds": elapsed_cpu,
            "submitted_rows_per_wall_second": (
                n_submitted / elapsed_wall if elapsed_wall else None
            ),
        },
        "versions": {
            "rdkit": rdkit.__version__,
            "intake_schema": SCHEMA_VERSION,
        },
    }
    _write_text_atomic(
        output_dir / SUMMARY_NAME,
        json.dumps(summary, indent=2, allow_nan=False) + "\n",                  # plsu a json file
    )
    return summary















######### command ##########

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, canonicalize, deduplicate, and characterize a SMILES submission"
    )
    parser.add_argument("input_smi", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = run_intake(args.input_smi, args.output_dir)
    counts = summary["counts"]
    metrics = summary["aggregate_metrics"]
    print(
        f"Intake complete: {counts['accepted_for_preparation']}/"
        f"{counts['submitted_rows']} accepted | "
        f"validity={metrics['validity']:.3f} | "
        f"uniqueness={metrics['uniqueness_among_valid']:.3f}"
    )
    print(f"Outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
