"""Run an existing candidate cohort through the complete local pipeline.

This runner deliberately starts from an existing SMILES cohort. It never opens
the multi-million-row ChEMBL source used to construct the naive baseline, so a
50-molecule pipeline test pays only for those 50 submissions plus docking cause we used to have
like a hour wait time just for testing 5 molecules.

Pipeline:
    raw subsample -> intake -> pre-dock gate -> ligand preparation -> Smina
    docking -> candidate-only analysis
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.analysis.run_candidates import VALID_ROLES, run_candidate_analysis
from src.harness import config, dock, intake, prepare_ligands, runtime

from . import filter as predock_filter


PIPELINE_SCHEMA_VERSION = 1


class CandidatePipelineError(ValueError):
    """Raised when a candidate pipeline request is ambiguous or unsafe."""


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)






#####  this defines the sampling of the candidate cohort from a source SMILES file  #####

def _raw_rows(source_smi: Path) -> list[tuple[int, str]]:
    source_smi = Path(source_smi)
    if not source_smi.is_file():
        raise CandidatePipelineError(f"Candidate input does not exist: {source_smi}")
    rows = [
        (line_number, line.strip())
        for line_number, line in enumerate(                                 # enumaration
            source_smi.read_text(encoding="utf-8-sig").splitlines(), 1      # strip BOM if present
        )                                                                   # BLANK LINES ARE EXCLUDED
        if line.strip()
    ]
    if not rows:
        raise CandidatePipelineError(f"Candidate input is empty: {source_smi}")
    return rows





def _recorded_id(line_number: int, raw_line: str) -> str:
    fields = raw_line.split()
    return fields[1] if len(fields) == 2 else f"SOURCE_L{line_number}"











#### The sampling Decisions are here ####

def create_submission(
    source_smi: Path,
    work_dir: Path,
    *,
    n: int | None,
    seed: int,
) -> tuple[Path, dict]:
    """Create and record a raw, uniform-without-replacement subsample."""

    source_smi = Path(source_smi)
    work_dir = Path(work_dir)
    if work_dir.exists():                   # Immutable run dir
        raise CandidatePipelineError(
            f"Candidate pipeline work directory already exists: {work_dir}"
        )
    if seed < 0:
        raise CandidatePipelineError("seed must be non-negative")

    rows = _raw_rows(source_smi)
    requested = len(rows) if n is None else int(n)
    if requested < 1:
        raise CandidatePipelineError("n must be at least 1")    # you cant have a candidate pipeline with no candidates because then you have no candidates
    if requested > len(rows):
        raise CandidatePipelineError(
            f"Requested {requested:,} submissions from only {len(rows):,} raw rows"
        )

    if requested == len(rows):
        selected_indices = np.arange(len(rows), dtype=int)
        method = "all non-blank raw submissions in source order"
    else:
        selected_indices = np.asarray(
            np.random.default_rng(seed).choice(
                len(rows), size=requested, replace=False
            ),
            dtype=int,
        )
        method = (
            "numpy.random.default_rng(seed).choice without replacement over "
            "non-blank raw submissions"
        )

    selected = [rows[int(index)] for index in selected_indices]
    work_dir.mkdir(parents=True)
    submission_path = work_dir / "submission.smi"
    ids_path = work_dir / "subsample_ids.txt"
    _atomic_text(
        submission_path,
        "".join(f"{raw_line}\n" for _, raw_line in selected),
    )
    _atomic_text(
        ids_path,
        "".join(
            f"{_recorded_id(line_number, raw_line)}\n"
            for line_number, raw_line in selected
        ),
    )
    record = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "candidate_raw_subsample",
        "method": method,
        "seed": seed,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "sampling_without_replacement": True,
        "input": runtime.file_record(source_smi),
        "outputs": {
            "submission_smi": runtime.file_record(submission_path),
            "subsample_ids": runtime.file_record(ids_path),
        },
    }
    runtime.write_json_atomic(work_dir / "subsample.json", record)
    return submission_path, record












###########    orchestration and the funnel    ############

def run_candidate_pipeline(
    source_smi: Path,
    work_dir: Path,
    *,
    name: str,
    role: str,
    source_description: str,
    n: int | None,
    seed: int,
    workers: int,
    analysis_outdir: Path,
) -> Path:
    """Run one immutable candidate cohort and return its HTML report path."""

    if role not in VALID_ROLES:
        raise CandidatePipelineError(
            f"Unknown role {role!r}; choose one of {VALID_ROLES}"
        )
    if workers < 1:
        raise CandidatePipelineError("workers must be at least 1")

    started = time.perf_counter()
    work_dir = Path(work_dir)
    submission_path, subsample = create_submission(
        source_smi,
        work_dir,
        n=n,
        seed=seed,
    )

    intake_dir = work_dir / "intake"
    intake_summary = intake.run_intake(submission_path, intake_dir)

    gate_dir = work_dir / "gate"
    gate_summary = predock_filter.run_gate(
        intake_dir / "accepted.smi",
        gate_dir,
    )

    prepared_dir = work_dir / "prepared"
    preparation = prepare_ligands.prepare_batch(
        gate_dir / "gate_pass.smi",
        prepared_dir,
        workers,
    )
    prepared_available = len(list(prepared_dir.glob("*.pdbqt")))
    if prepared_available == 0:
        raise CandidatePipelineError(
            "No gate-passing candidate produced a PDBQT; candidate analysis "
            "cannot continue without a docking score"
        )

    docking_dir = work_dir / "docking"
    scores_path = docking_dir / "scores.csv"
    docking = dock.dock_batch(
        prepared_dir,
        docking_dir,
        scores_path,
        workers,
    )
    successfully_scored = int(docking["ok"] + docking["cached"])
    if successfully_scored == 0:
        raise CandidatePipelineError(
            "Docking produced no observed scores; candidate analysis cannot continue"
        )

    report_path = run_candidate_analysis(
        scores_path,
        intake_dir / "molecules.csv",
        name,
        Path(analysis_outdir),
        role=role,
        source_description=source_description,
    )

    intake_counts = intake_summary["counts"]                    #  7 Keys for the analysis later on
    funnel = {
        "submitted": int(intake_counts["submitted_rows"]),
        "accepted_at_intake": int(intake_counts["accepted_for_preparation"]),
        "passed_predock_gate": int(gate_summary["n_passed"]),
        "prepared_pdbqt_available": prepared_available,
        "preparation_failed": int(preparation["failed"]),
        "successfully_scored": successfully_scored,
        "docking_failed": int(docking["failed"]),
    }
    summary = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "candidate_pipeline",
        "name": name,
        "role": role,
        "source_description": source_description,
        "parameters": {
            "requested_subsample_size": n,
            "subsample_seed": seed,
            "workers": workers,
        },
        "funnel": funnel,
        "rates": {
            "dockable_per_submitted": (
                funnel["passed_predock_gate"] / funnel["submitted"]
            ),
            "scored_per_submitted": (
                funnel["successfully_scored"] / funnel["submitted"]
            ),
        },
        "inputs": {
            "source_smi": subsample["input"],
            "submission_smi": runtime.file_record(submission_path),
        },
        "outputs": {
            "subsample_record": runtime.file_record(work_dir / "subsample.json"),
            "intake_summary": runtime.file_record(intake_dir / "summary.json"),
            "gate_summary": runtime.file_record(gate_dir / "gate_summary.json"),
            "preparation_summary": runtime.file_record(
                prepared_dir / "_prep_summary.json"
            ),
            "docking_summary": runtime.file_record(
                docking_dir / "_dock_summary.json"
            ),
            "scores_csv": runtime.file_record(scores_path),
            "candidate_report": str(Path(report_path).resolve()),
        },
        "timing": runtime.timing_record(
            started,
            attempted_tasks=funnel["submitted"],
            workers=workers,
            cpu_slots_per_task=config.SMINA_CPU,
        ),
        "hardware": runtime.hardware_record(),
    }
    runtime.write_json_atomic(work_dir / "pipeline_summary.json", summary)
    return Path(report_path)













def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subsample an existing candidate cohort and run intake, gate, "
            "preparation, docking, and candidate analysis"
        )
    )
    parser.add_argument("source_smi", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", choices=VALID_ROLES, default="pilot")
    parser.add_argument("--source-description", default="")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="raw submissions to sample (default: use the full source cohort)",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--analysis-outdir",
        type=Path,
        default=config.REPO_ROOT / "results" / "candidates",
    )
    args = parser.parse_args()

    try:
        report = run_candidate_pipeline(
            args.source_smi,
            args.work_dir,
            name=args.name,
            role=args.role,
            source_description=args.source_description,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
            analysis_outdir=args.analysis_outdir,
        )
    except (CandidatePipelineError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Candidate pipeline report: {report}")


if __name__ == "__main__":
    main()
