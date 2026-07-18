"""
Batch docking: PDBQT ligands -> Smina best-mode affinity.


Docks every <ID>.pdbqt in a directory against the receptor and writes a scores CSV.


Score = the best pose's affinity, read at full precision from the output    (for sorting / ranking purposes)
pose file's `REMARK minimizedAffinity` line                                 (Smina writes models best-first and hopefully continues to do so),


Usage:
    python -m src.harness.dock LIGAND_DIR OUT_DIR [--scores scores.csv] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from . import config, runtime
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.harness import config, runtime




#########    Open Output File, Read Affinity    ###########

def parse_affinity(pose_pdbqt: Path):                           # function called twice
    try:                                                        # once for resumability
        for line in pose_pdbqt.read_text().splitlines():        # once for reading the smina results, cause smina outputs files
            if line.startswith("REMARK minimizedAffinity"):
                return float(line.split()[-1])                    
    except (OSError, ValueError):
        return None
    return None






#########   The Docking a Ligand Function   ########

def dock_one(ligand_pdbqt: Path, pose_out: Path) -> tuple[str, float | None, str, str]:       # function returns a status
    mol_id = ligand_pdbqt.stem


###  Check if output already exists  ###

    if pose_out.exists() and pose_out.stat().st_size > 0:                             # resumability again, only for fixed config btw
        score = parse_affinity(pose_out)             # if file exists,                # if you crash and want to change config, make sure
        if score is not None:                        # and has a score,               # you wipe out the out_dir,
            return mol_id, score, "cached", ""       # dont even dock                 # cause the filename is the cache key


###  The Rules for the Dock  ###

    cmd = [
        "smina",
        "-r", str(config.RECEPTOR_PDBQT),
        "-l", str(ligand_pdbqt),
        *config.smina_box_args(),
        "--exhaustiveness", str(config.EXHAUSTIVENESS),
        "--seed", str(config.SEED),
        "--num_modes", str(config.NUM_MODES),
        "--energy_range", str(config.ENERGY_RANGE),
        "--cpu", str(config.SMINA_CPU),
        "-o", str(pose_out),
        "--quiet",
    ]
   
   
 ###  If no return on resumability function, run the dock  ###
   
    try:
        proc = subprocess.run(                                                        # Run Smina
            cmd, capture_output=True, text=True, timeout=config.DOCK_TIMEOUT_S        # Hand Over the Rules and Instructions
        )                                                     
    except subprocess.TimeoutExpired:
        return mol_id, None, "timeout", f">{config.DOCK_TIMEOUT_S}s"                  # The time out

  
  ###  If Smina returns a failure  ###
  
    if proc.returncode != 0:                                                   
        reason = (proc.stderr or proc.stdout).strip().splitlines()
        return mol_id, None, "dock_failed", (reason[-1] if reason else "rc!=0")[:80]


### Read fresh pose file and check affinity ###

    score = parse_affinity(pose_out)
    if score is None:
        return mol_id, None, "no_score", "affinity not found in output"
    return mol_id, score, "ok", ""







#########    Manager Function     ############

def dock_batch(ligand_dir: Path, out_dir: Path, scores_csv: Path,         # 4 Inputs
               workers: int) -> dict[str, int]:                           # Dictionary Output

    started = time.perf_counter()                   # extra timing module for 
    ligand_dir = Path(ligand_dir)                   # for computational cost records
    out_dir = Path(out_dir)
    scores_csv = Path(scores_csv)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    out_dir.mkdir(parents=True, exist_ok=True)                            # Make Output Dir
    scores_csv.parent.mkdir(parents=True, exist_ok=True)



####  Input  ####

    ligands = sorted(ligand_dir.glob("*.pdbqt"))                          # Grab the .pdbqt in the input dir and sort
    print(f"Found {len(ligands)} ligand PDBQTs in {ligand_dir}")
    if not ligands:
        print("  nothing to dock.")
        return {"total": 0, "ok": 0, "cached": 0, "failed": 0}

    results: list[tuple[str, float | None, str, str]] = []
    counts = {"ok": 0, "cached": 0, "failed": 0}



####  Parallel Machinery  #####

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(dock_one, lig, out_dir / f"{lig.stem}_out.pdbqt"): lig        # schedules "dock_one function" to run on a thread
            for lig in ligands                                                      # submit for parallelism
        }                                                                           
        for i, fut in enumerate(as_completed(futures), 1):                          # results + enumeration
            mol_id, score, status, reason = fut.result()
            results.append((mol_id, score, status, reason))                         # uncouple the tuple to 4 components
          
            if status == "ok":
                counts["ok"] += 1
            elif status == "cached":
                counts["cached"] += 1
            else:
                counts["failed"] += 1
            if i % 50 == 0 or i == len(ligands):
                print(f"  progress: {i}/{len(ligands)}  "
                      f"(ok={counts['ok']}, cached={counts['cached']}, "
                      f"failed={counts['failed']})")


####  Sort and Write  #####

    results.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0.0))   # sort the affinity
    with scores_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id", "score_kcal_mol", "status", "reason"])
        for mol_id, score, status, reason in results:
            w.writerow([mol_id, "" if score is None else f"{score:.4f}",
                        status, reason])
    print(f"Wrote scores to {scores_csv}")

    summary = {"total": len(ligands), **counts}
    stage_timing = runtime.timing_record(
        started,
        attempted_tasks=len(ligands),
        workers=workers,
        cpu_slots_per_task=config.SMINA_CPU,
    )
    stage_timing["fresh_successes_per_wall_second"] = (
        counts["ok"] / stage_timing["wall_seconds"]
        if stage_timing["wall_seconds"]
        else None
    )
    preparation_summary = ligand_dir / "_prep_summary.json"
    receptor_record = (
        runtime.file_record(config.RECEPTOR_PDBQT)
        if config.RECEPTOR_PDBQT.is_file()
        else {
            "path": str(config.RECEPTOR_PDBQT.resolve()),
            "missing": True,
        }
    )
    cost_summary = {
        "schema_version": runtime.RUNTIME_SCHEMA_VERSION,
        "stage": "smina_docking",
        "inputs": {
            "ligand_pdbqt": runtime.file_set_record(ligands),
            "receptor_pdbqt": receptor_record,
            "preparation_summary": (
                runtime.file_record(preparation_summary)
                if preparation_summary.is_file()
                else None
            ),
        },
        "outputs": {
            "scores_csv": runtime.file_record(scores_csv),
            "pose_pdbqt": runtime.file_set_record(
                sorted(out_dir.glob("*_out.pdbqt"))
            ),
        },
        "counts": summary,
        "parameters": {
            "workers": workers,
            "smina_cpu_per_job": config.SMINA_CPU,
            "box_center_a": list(config.BOX_CENTER),
            "box_size_a": list(config.BOX_SIZE),
            "exhaustiveness": config.EXHAUSTIVENESS,
            "seed": config.SEED,
            "num_modes": config.NUM_MODES,
            "energy_range_kcal_mol": config.ENERGY_RANGE,
            "timeout_seconds_per_ligand": config.DOCK_TIMEOUT_S,
        },
        "timing": stage_timing,
        "hardware": runtime.hardware_record(),
    }
    runtime.write_json_atomic(out_dir / "_dock_summary.json", cost_summary)
    print(f"Done: {summary}")
    return summary








def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Smina docking of PDBQT ligands")
    ap.add_argument("ligand_dir", type=Path, help="dir of <ID>.pdbqt inputs")
    ap.add_argument("out_dir", type=Path, help="dir for <ID>_out.pdbqt poses")
    ap.add_argument("--scores", type=Path, default=None,
                    help="scores CSV path (default: OUT_DIR/scores.csv)")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()
    scores = args.scores or (args.out_dir / "scores.csv")
    dock_batch(args.ligand_dir, args.out_dir, scores, args.workers)


if __name__ == "__main__":
    main()
