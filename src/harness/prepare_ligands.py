"""


SMILES -> 3D PDBQT 


Pipeline per molecule (see config.py knobs):
    SMILES --RDKit--> add H --ETKDGv3 embed--> MMFF94 optimize
           --Meeko--> AutoDock PDBQT string --> <out_dir>/<ID>.pdbqt


Usage:
    python -m src.harness.prepare_ligands INPUT.smi OUTPUT_DIR [--workers N]
    .smi format: one "SMILES<space>ID" per line, no header or else


"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy






try:                                         # run bare or module (module better)
    from . import config, runtime
except ImportError:                          # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.harness import config, runtime
RDLogger.DisableLog("rdApp.*")               #disable annoying log

_SAFE = re.compile(r"[^A-Za-z0-9._-]")       # Characters allowed in an output filename; everything else -> underscore.







def safe_name(mol_id: str) -> str:
    return _SAFE.sub("_", mol_id.strip())



###### core Smi to PDBQT funtion #########

def smiles_to_pdbqt(smi: str) -> tuple[str | None, str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "rdkit_parse_failed"

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = config.EMBED_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True                        # Retry once with random coords if smth is stubborn
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None, "embed_failed"

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)                # MMFF94 if parameterizable, 
        else:
            AllChem.UFFOptimizeMolecule(mol)                 # else UFF
    except Exception as e:  
        return None, f"optimize_error:{type(e).__name__}"    # both fail, its cool, docking engine searches space anyways

    try:
        prep = MoleculePreparation(rigid_macrocycles=config.RIGID_MACROCYCLES)    #setup the machine
        setups = prep.prepare(mol)                                                #run the machine, return list of setups
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])                #grab 1st setup, get pdbqt, or an error
    except Exception as e:
        return None, f"meeko_error:{type(e).__name__}"           #meeko crashed for some reason ig

    if not ok:                                                   #meeko did its job, and its job was to tell me why it failed
        return None, f"meeko_write_failed:{err.strip()[:60]}"
    return pdbqt, ""






 ####### worker ########

def _worker(task: tuple[str, str, str]) -> tuple[str, str, str, str]:         # create the sweat shop
    """(mol_id, smiles, out_path) -> (mol_id, smiles, status, reason)."""     # 3-tuple in, 4-tuple out
    mol_id, smi, out_path = task                                              
    pdbqt, reason = smiles_to_pdbqt(smi)                                      # worker runs function
    if pdbqt is None:
        return mol_id, smi, "fail", reason
    Path(out_path).write_text(pdbqt)                                          # worker writes on output file
    return mol_id, smi, "ok", ""




####### parsing #########

def read_smi(path: Path) -> list[tuple[str, str]]:                       # must be .smi , parsed as mol_id, smiles pairs
    pairs = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):     # make line numbers to correlate any issues
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            print(f"  WARN: line {lineno} has no ID, skipping: {line!r}")
            continue
        smi, mol_id = parts[0], parts[1]                                 # split line on white space, take column 0 as smiles, column 1 as ID
        pairs.append((mol_id, smi))                                      # dont mix up id with smiles, or its all over
    return pairs





##### Boss ######

def prepare_batch(smi_path: Path, out_dir: Path, workers: int) -> dict[str, int]:
    started = time.perf_counter()                           
    smi_path = Path(smi_path)                              # New block starts time 
    out_dir = Path(out_dir)                                # normalize paths
    if workers < 1:                                        # reject 0 workers, because that would be a waste of time, and a waste of electricity, and a waste of money, and a waste of life
        raise ValueError("workers must be at least 1")

    out_dir.mkdir(parents=True, exist_ok=True)                                      # make directory and stfu if it exists
    pairs = read_smi(smi_path)
    print(f"Read {len(pairs)} molecules from {smi_path}")

## 2 lists, tasks and lists, for resumability ##

    tasks, skipped = [], 0
    for mol_id, smi in pairs:
        out_path = out_dir / f"{safe_name(mol_id)}.pdbqt"            # does out path exist?
        if out_path.exists() and out_path.stat().st_size > 0:        # is it non empty?
            skipped += 1                                             # if yes then skip
            continue
        tasks.append((mol_id, smi, str(out_path)))

    print(f"  {skipped} already prepared (skipped), {len(tasks)} to do, "
          f"{workers} workers")

    n_ok, failures = 0, []

##### start the tasks #####

    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:      # get the workers 
            futures = [ex.submit(_worker, t) for t in tasks]    
            for i, fut in enumerate(as_completed(futures), 1):    # list not written as mols qeued, but how fast they finished(speed reasons)
                mol_id, smi, status, reason = fut.result()
                if status == "ok":                                # go over lines, if process lands, ok, 
                    n_ok += 1
                else:
                    failures.append((mol_id, smi, reason))        # if not, append to failure list
                if i % 100 == 0 or i == len(tasks):
                    print(f"  progress: {i}/{len(tasks)}  "
                          f"(ok={n_ok}, fail={len(failures)})")




###########    CSV for the failiures    ###########  

        manifest = out_dir / "_prep_failures.csv"
        with manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["molecule_id", "smiles", "reason"])
            w.writerows(failures)
        print(f"  wrote {len(failures)} failures to {manifest}")


    summary = {"total": len(pairs), "skipped": skipped,
               "prepared": n_ok, "failed": len(failures)}
    prepared_files = sorted(out_dir.glob("*.pdbqt"))                ####### EXTRA MODULE #######
    failure_manifest = out_dir / "_prep_failures.csv"                   # for writing _prep_summary.json
    cost_summary = {
        "schema_version": runtime.RUNTIME_SCHEMA_VERSION,
        "stage": "ligand_preparation",
        "input": {
            "smiles": runtime.file_record(smi_path),
        },
        "outputs": {
            "prepared_pdbqt": runtime.file_set_record(prepared_files),
            "failures_csv": (
                runtime.file_record(failure_manifest)
                if failure_manifest.is_file()
                else None
            ),
        },
        "counts": {
            **summary,
            "attempted_this_invocation": len(tasks),
            "successful_pdbqt_available": len(prepared_files),
        },
        "parameters": {
            "workers": workers,
            "embed_method": "RDKit ETKDGv3",
            "embed_seed": config.EMBED_SEED,
            "optimization": "MMFF94 when parameterized, otherwise UFF",
            "rigid_macrocycles": config.RIGID_MACROCYCLES,
        },
        "timing": runtime.timing_record(
            started,
            attempted_tasks=len(tasks),
            workers=workers,
        ),
        "hardware": runtime.hardware_record(),
    }
    runtime.write_json_atomic(out_dir / "_prep_summary.json", cost_summary)
    print(f"Done: {summary}")
    return summary





def main() -> None:
    ap = argparse.ArgumentParser(description="Batch SMILES -> PDBQT for docking")
    ap.add_argument("smi", type=Path, help="input .smi (SMILES<space>ID/line)")
    ap.add_argument("out_dir", type=Path, help="output dir for <ID>.pdbqt files")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()
    prepare_batch(args.smi, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
