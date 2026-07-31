"""
Run one complete labelled benchmark on this machine: prepare, dock, analyse.

This is the laptop path. It is a thin orchestrator -- every scientific step is
the same module the cluster path calls, so a local run and a cluster run cannot
drift apart. For large campaigns use src.harness.chunks instead; chunking exists
to make work schedulable and independently verifiable across nodes, not because
it does anything different to a molecule.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config, dock, prepare_ligands, runtime

from ..analysis import config as analysis_config
from ..analysis.dataset import AnalysisInputError
from ..analysis.run_analysis import run as run_analysis


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def _stage(cohort: str, smi: Path, ligand_dir: Path, pose_dir: Path, workers: int) -> Path:
    """Prepare and dock one cohort; return its scores.csv path."""

    _banner(f"[{cohort}] prepare ligands")
    prepare_ligands.prepare_batch(smi, ligand_dir, workers)

    _banner(f"[{cohort}] dock")
    scores_csv = pose_dir / "scores.csv"
    dock.dock_batch(ligand_dir, pose_dir, scores_csv, workers)
    return scores_csv


def run_local(
    cohort_dir: Path,
    *,
    name: str | None = None,
    workers: int = config.WORKERS,
    analysis_outdir: Path = config.REPO_ROOT / "results",
    missing_policy: str = analysis_config.MISSING_SCORE_POLICY,
    chemistry: bool = False,
    active_intake: Path | None = None,
    decoy_intake: Path | None = None,
) -> Path:
    """Prepare, dock, and analyse a cohort directory built by build_cohort."""

    started = time.perf_counter()
    cohort_dir = Path(cohort_dir)
    actives_smi = cohort_dir / "actives.smi"
    decoys_smi = cohort_dir / "decoys.smi"
    for path in (actives_smi, decoys_smi):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found -- build the cohort first with "
                f"python -m src.harness.build_cohort {cohort_dir} --n-actives N"
            )
    run_name = name or cohort_dir.name

    active_scores = _stage(
        "actives", actives_smi,
        cohort_dir / "ligands" / "actives",
        cohort_dir / "docking" / "actives",
        workers,
    )
    decoy_scores = _stage(
        "decoys", decoys_smi,
        cohort_dir / "ligands" / "decoys",
        cohort_dir / "docking" / "decoys",
        workers,
    )

    _banner("labelled benchmark analysis")
    report = run_analysis(
        active_scores,
        decoy_scores,
        None,
        run_name,
        Path(analysis_outdir),
        active_intake_path=active_intake if chemistry else None,
        decoy_intake_path=decoy_intake if chemistry else None,
        active_smi_path=actives_smi,        # the cohort .smi files that were docked,
        decoy_smi_path=decoys_smi,          # so the size diagnostic uses the same molecules
        missing_policy=missing_policy,
    )

    runtime.write_json_atomic(
        cohort_dir / "run_local.json",
        {
            "stage": "run_local",
            "schema_version": 1,
            "name": run_name,
            "workers": workers,
            "missing_policy": missing_policy,
            "chemical_profiling": chemistry,
            "inputs": {
                "actives_smi": runtime.file_record(actives_smi),
                "decoys_smi": runtime.file_record(decoys_smi),
            },
            "outputs": {
                "active_scores": runtime.file_record(active_scores),
                "decoy_scores": runtime.file_record(decoy_scores),
                "report_html": str(Path(report).resolve()),
            },
            # attempted_tasks is molecules docked, so this is end-to-end
            # pipeline cost including preparation and analysis, not dock-only.
            "timing": runtime.timing_record(
                started,
                attempted_tasks=(
                    len(prepare_ligands.read_smi(actives_smi))
                    + len(prepare_ligands.read_smi(decoys_smi))
                ),
                workers=workers,
                cpu_slots_per_task=config.SMINA_CPU,
            ),
            "hardware": runtime.hardware_record(),
        },
    )

    _banner(f"done -> {report}")
    return Path(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, dock, and analyse a labelled cohort on this machine"
    )
    parser.add_argument("cohort_dir", type=Path, help="directory from build_cohort")
    parser.add_argument("--name", default=None, help="run identifier (default: dir name)")
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--analysis-outdir", type=Path, default=config.REPO_ROOT / "results"
    )
    parser.add_argument(
        "--missing-policy",
        choices=analysis_config.VALID_MISSING_SCORE_POLICIES,
        default=analysis_config.MISSING_SCORE_POLICY,
    )
    parser.add_argument(
        "--chemistry", action="store_true",
        help=(
            "add chemical profiling from the intake tables; only meaningful for "
            "a full cohort, because intake statistics cover every accepted "
            "molecule rather than the docked subset"
        ),
    )
    parser.add_argument("--active-intake", type=Path, default=None)
    parser.add_argument("--decoy-intake", type=Path, default=None)
    args = parser.parse_args()

    if args.chemistry and not (args.active_intake and args.decoy_intake):
        parser.error("--chemistry requires --active-intake and --decoy-intake")

    try:
        run_local(
            args.cohort_dir,
            name=args.name,
            workers=args.workers,
            analysis_outdir=args.analysis_outdir,
            missing_policy=args.missing_policy,
            chemistry=args.chemistry,
            active_intake=args.active_intake,
            decoy_intake=args.decoy_intake,
        )
    except (FileNotFoundError, AnalysisInputError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
