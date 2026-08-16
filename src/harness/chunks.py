"""Create, run, verify, and merge portable docking chunk bundles.

The bundle is deliberately self-contained: it keeps the normalized source
SMILES, deterministic chunks, content hashes, and a Slurm array script.  A
cluster run therefore does not depend on a path that only exists on the laptop.

Typical flow:
    python -m src.harness.chunks create accepted.smi hpc_bundle
    sbatch hpc_bundle/submit_slurm_array.sh
    python -m src.harness.chunks merge hpc_bundle/manifest.json \
        hpc_bundle/run merged_docking
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from . import config, runtime


CHUNK_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CPUS_PER_TASK = 8
DEFAULT_MEMORY = "8G"
DEFAULT_TIME_LIMIT = "04:00:00"
SCORE_COLUMNS = ("molecule_id", "score_kcal_mol", "status", "reason")
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


class ChunkManifestError(ValueError):
    """Raised when a chunk bundle or completed array is ambiguous."""






def _read_smi_strict(path: Path) -> list[tuple[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise ChunkManifestError(f"SMILES file does not exist: {path}")
    rows: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ChunkManifestError(
                f"{path}:{line_number} must be exactly SMILES<space>ID"
            )
        smiles, molecule_id = parts
        if not SAFE_ID.fullmatch(molecule_id):
            raise ChunkManifestError(
                f"{path}:{line_number} has unsafe molecule ID {molecule_id!r}; "
                "use only letters, digits, '.', '_', and '-'"
            )
        if molecule_id in seen_ids:
            raise ChunkManifestError(
                f"{path} contains duplicate molecule ID {molecule_id!r}"
            )
        seen_ids.add(molecule_id)
        rows.append((smiles, molecule_id))
    if not rows:
        raise ChunkManifestError(f"SMILES file contains no molecules: {path}")
    return rows


def _write_smi(path: Path, rows: list[tuple[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(f"{smiles} {molecule_id}\n" for smiles, molecule_id in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_record(path: Path, root: Path) -> dict:
    record = runtime.file_record(path)
    record["path"] = str(Path(path).resolve().relative_to(Path(root).resolve()))
    return record


def _slurm_script(
    n_chunks: int,
    cpus_per_task: int,
    memory: str,
    time_limit: str,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=dyrk1a-dock
#SBATCH --array=0-{n_chunks - 1}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${{BENCH_REPO_ROOT:-${{SLURM_SUBMIT_DIR:-$PWD}}}}"
RUN_ROOT="${{BENCH_RUN_ROOT:-${{BUNDLE_DIR}}/run}}"
PYTHON_BIN="${{BENCH_PYTHON:-python}}"

mkdir -p "${{RUN_ROOT}}"
cd "${{REPO_ROOT}}"
"${{PYTHON_BIN}}" -m src.harness.chunks run \\
  "${{BUNDLE_DIR}}/manifest.json" \\
  "${{SLURM_ARRAY_TASK_ID}}" \\
  "${{RUN_ROOT}}" \\
  --workers "${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
"""


def create_manifest(
    source_smi: Path,
    outdir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cpus_per_task: int = DEFAULT_CPUS_PER_TASK,
    memory: str = DEFAULT_MEMORY,
    time_limit: str = DEFAULT_TIME_LIMIT,
) -> dict:
    """Create an immutable, portable chunk bundle."""

    source_smi = Path(source_smi)
    outdir = Path(outdir)
    if chunk_size < 1:
        raise ChunkManifestError("chunk_size must be at least 1")
    if cpus_per_task < 1:
        raise ChunkManifestError("cpus_per_task must be at least 1")
    if outdir.exists():
        raise ChunkManifestError(f"Chunk output path already exists: {outdir}")
    rows = _read_smi_strict(source_smi)

    outdir.mkdir(parents=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir()
    bundled_source = outdir / "source.smi"
    _write_smi(bundled_source, rows)

    chunk_records = []
    for index, start in enumerate(range(0, len(rows), chunk_size)):
        chunk_rows = rows[start : start + chunk_size]
        chunk_path = chunk_dir / f"chunk_{index:04d}.smi"
        _write_smi(chunk_path, chunk_rows)
        chunk_records.append(
            {
                "index": index,
                "path": str(chunk_path.relative_to(outdir)),
                "n_molecules": len(chunk_rows),
                "first_molecule_id": chunk_rows[0][1],
                "last_molecule_id": chunk_rows[-1][1],
                "sha256": runtime.sha256_file(chunk_path),
                "bytes": chunk_path.stat().st_size,
            }
        )

    script_path = outdir / "submit_slurm_array.sh"
    script_path.write_text(
        _slurm_script(
            len(chunk_records),
            cpus_per_task,
            memory,
            time_limit,
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manifest = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "stage": "portable_docking_chunk_manifest",
        "interpretation": {
            "purpose": "deterministic laptop-to-cluster docking handoff",
            "chunks_are_independently_resumable": True,
            "cluster_partition_and_account_are_intentionally_unset": True,
        },
        "source_input": runtime.file_record(source_smi),
        "bundle_source": _relative_record(bundled_source, outdir),
        "parameters": {
            "chunk_size": chunk_size,
            "cpus_per_task": cpus_per_task,
            "memory": memory,
            "time_limit": time_limit,
            "smina_cpu_per_job": config.SMINA_CPU,
        },
        "counts": {
            "molecules": len(rows),
            "chunks": len(chunk_records),
        },
        "chunks": chunk_records,
        "slurm_script": _relative_record(script_path, outdir),
    }
    runtime.write_json_atomic(outdir / "manifest.json", manifest)
    return manifest


def _load_manifest(manifest_path: Path) -> tuple[Path, dict]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ChunkManifestError(f"Manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChunkManifestError(f"Cannot read manifest {manifest_path}: {error}")
    if manifest.get("schema_version") != CHUNK_SCHEMA_VERSION:
        raise ChunkManifestError(
            f"Unsupported chunk schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("stage") != "portable_docking_chunk_manifest":
        raise ChunkManifestError(f"Not a docking chunk manifest: {manifest_path}")
    return manifest_path.resolve(), manifest


def _bundle_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ChunkManifestError(
            f"Manifest path escapes its bundle: {relative_path!r}"
        ) from error
    return candidate


def _check_file_record(path: Path, record: dict, label: str) -> None:
    if not path.is_file():
        raise ChunkManifestError(f"Missing {label}: {path}")
    actual_hash = runtime.sha256_file(path)
    if actual_hash != record.get("sha256"):
        raise ChunkManifestError(
            f"{label} hash mismatch for {path}; expected "
            f"{record.get('sha256')}, found {actual_hash}"
        )
    if path.stat().st_size != record.get("bytes"):
        raise ChunkManifestError(f"{label} byte count mismatch for {path}")


def verify_manifest(manifest_path: Path) -> dict:
    """Verify hashes, molecule order, and chunk coverage."""

    manifest_path, manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    source_record = manifest.get("bundle_source", {})
    source_path = _bundle_path(root, str(source_record.get("path", "")))
    _check_file_record(source_path, source_record, "bundled source")
    source_rows = _read_smi_strict(source_path)

    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ChunkManifestError("Manifest contains no chunks")
    combined_rows: list[tuple[str, str]] = []
    for expected_index, record in enumerate(chunks):
        if record.get("index") != expected_index:
            raise ChunkManifestError("Chunk indexes must be consecutive from zero")
        chunk_path = _bundle_path(root, str(record.get("path", "")))
        _check_file_record(chunk_path, record, f"chunk {expected_index}")
        chunk_rows = _read_smi_strict(chunk_path)
        if len(chunk_rows) != record.get("n_molecules"):
            raise ChunkManifestError(
                f"Chunk {expected_index} molecule count does not match manifest"
            )
        if (
            chunk_rows[0][1] != record.get("first_molecule_id")
            or chunk_rows[-1][1] != record.get("last_molecule_id")
        ):
            raise ChunkManifestError(
                f"Chunk {expected_index} endpoint IDs do not match manifest"
            )
        combined_rows.extend(chunk_rows)

    if combined_rows != source_rows:
        raise ChunkManifestError(
            "Concatenated chunks do not exactly reproduce bundled source order"
        )
    if len(source_rows) != manifest.get("counts", {}).get("molecules"):
        raise ChunkManifestError("Manifest source molecule count is incorrect")
    if len(chunks) != manifest.get("counts", {}).get("chunks"):
        raise ChunkManifestError("Manifest chunk count is incorrect")

    script_record = manifest.get("slurm_script", {})
    script_path = _bundle_path(root, str(script_record.get("path", "")))
    _check_file_record(script_path, script_record, "Slurm script")
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "molecules": len(source_rows),
        "chunks": len(chunks),
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in SCORE_COLUMNS
        ):
            raise ChunkManifestError(
                f"{path} must contain columns {list(SCORE_COLUMNS)}"
            )
        return [
            {column: str(row.get(column, "") or "").strip() for column in SCORE_COLUMNS}
            for row in reader
        ]


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCORE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _preparation_failures(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("molecule_id", "")).strip(): str(row.get("reason", "")).strip()
            for row in reader
            if str(row.get("molecule_id", "")).strip()
        }


def _complete_chunk_scores(
    chunk_rows: list[tuple[str, str]],
    raw_scores_path: Path,
    failures_path: Path,
    output_path: Path,
) -> list[dict[str, str]]:
    raw_rows = _read_csv_rows(raw_scores_path) if raw_scores_path.is_file() else []
    expected_ids = [molecule_id for _, molecule_id in chunk_rows]
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        molecule_id = row["molecule_id"]
        if molecule_id not in expected_set:
            raise ChunkManifestError(
                f"Docking produced unexpected molecule ID {molecule_id!r}"
            )
        if molecule_id in by_id:
            raise ChunkManifestError(
                f"Docking produced duplicate molecule ID {molecule_id!r}"
            )
        by_id[molecule_id] = row
    failures = _preparation_failures(failures_path)
    completed = []
    for molecule_id in expected_ids:
        if molecule_id in by_id:
            completed.append(by_id[molecule_id])
        else:
            completed.append(
                {
                    "molecule_id": molecule_id,
                    "score_kcal_mol": "",
                    "status": "not_prepared",
                    "reason": failures.get(
                        molecule_id,
                        "no prepared PDBQT was available",
                    ),
                }
            )
    _write_csv_rows(output_path, completed)
    return completed


def _optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_chunk(
    manifest_path: Path,
    index: int,
    run_root: Path,
    *,
    workers: int,
) -> dict:
    """Prepare and dock one verified chunk; safe to rerun in the same path."""

    started = time.perf_counter()
    verify_manifest(manifest_path)
    manifest_path, manifest = _load_manifest(manifest_path)
    if workers < 1:
        raise ChunkManifestError("workers must be at least 1")
    chunks = manifest["chunks"]
    if index < 0 or index >= len(chunks):
        raise ChunkManifestError(
            f"Chunk index {index} is outside 0-{len(chunks) - 1}"
        )

    from .dock import dock_batch
    from .prepare_ligands import prepare_batch

    chunk_record = chunks[index]
    chunk_path = _bundle_path(manifest_path.parent, chunk_record["path"])
    chunk_rows = _read_smi_strict(chunk_path)
    chunk_root = Path(run_root) / f"chunk_{index:04d}"
    prepared_dir = chunk_root / "prepared"
    poses_dir = chunk_root / "poses"
    raw_scores_path = chunk_root / "raw_scores.csv"
    scores_path = chunk_root / "scores.csv"
    chunk_root.mkdir(parents=True, exist_ok=True)

    prepare_batch(chunk_path, prepared_dir, workers)
    dock_batch(prepared_dir, poses_dir, raw_scores_path, workers)
    score_rows = _complete_chunk_scores(
        chunk_rows,
        raw_scores_path,
        prepared_dir / "_prep_failures.csv",
        scores_path,
    )

    prep_summary_path = prepared_dir / "_prep_summary.json"
    dock_summary_path = poses_dir / "_dock_summary.json"
    prep_summary = _optional_json(prep_summary_path)
    dock_summary = _optional_json(dock_summary_path)
    estimated_slot_hours = sum(
        float(record.get("timing", {}).get("estimated_requested_cpu_slot_hours", 0.0))
        for record in (prep_summary, dock_summary)
        if record is not None
    )
    successful = sum(
        row["status"].lower() in {"ok", "cached"} for row in score_rows
    )
    result = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "stage": "portable_docking_chunk_result",
        "status": "complete",
        "chunk_index": index,
        "manifest": runtime.file_record(manifest_path),
        "input_chunk": runtime.file_record(chunk_path),
        "counts": {
            "expected": len(chunk_rows),
            "score_rows": len(score_rows),
            "successful_scores": successful,
            "failed_or_missing_scores": len(score_rows) - successful,
        },
        "outputs": {
            "scores_csv": runtime.file_record(scores_path),
            "preparation_summary": runtime.file_record(prep_summary_path),
            "docking_summary": (
                runtime.file_record(dock_summary_path)
                if dock_summary_path.is_file()
                else None
            ),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "workers_requested": workers,
            "estimated_requested_cpu_slot_hours": estimated_slot_hours,
            "note": "Chunk wall time includes preparation, docking, and bookkeeping.",
        },
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "google_batch_job_name": os.environ.get("BENCH_JOB_NAME"),
            "google_batch_task_index": os.environ.get("BATCH_TASK_INDEX"),
            "google_batch_task_count": os.environ.get("BATCH_TASK_COUNT"),
            "google_batch_retry_attempt": os.environ.get(
                "BATCH_TASK_RETRY_ATTEMPT"
            ),
            "container_image_uri": os.environ.get("BENCH_IMAGE_URI"),
        },
        "hardware": runtime.hardware_record(),
    }
    runtime.write_json_atomic(chunk_root / "chunk_result.json", result)
    return result


def _score_rows_for_merge(
    path: Path,
    expected_ids: list[str],
) -> list[dict[str, str]]:
    rows = _read_csv_rows(path)
    expected = set(expected_ids)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        molecule_id = row["molecule_id"]
        if molecule_id not in expected:
            raise ChunkManifestError(
                f"{path} contains unexpected molecule ID {molecule_id!r}"
            )
        if molecule_id in by_id:
            raise ChunkManifestError(
                f"{path} contains duplicate molecule ID {molecule_id!r}"
            )
        by_id[molecule_id] = row
    return [
        by_id.get(
            molecule_id,
            {
                "molecule_id": molecule_id,
                "score_kcal_mol": "",
                "status": "missing_result",
                "reason": "molecule absent from completed chunk score table",
            },
        )
        for molecule_id in expected_ids
    ]


def merge_chunks(
    manifest_path: Path,
    run_root: Path,
    outdir: Path,
) -> dict:
    """Merge completed chunk scores in original source order."""

    verify_manifest(manifest_path)
    manifest_path, manifest = _load_manifest(manifest_path)
    run_root = Path(run_root)
    outdir = Path(outdir)
    if outdir.exists():
        raise ChunkManifestError(f"Merge output path already exists: {outdir}")

    merged_rows: list[dict[str, str]] = []
    chunk_results = []
    docking_summaries = []
    manifest_hash = runtime.sha256_file(manifest_path)
    for chunk_record in manifest["chunks"]:
        index = int(chunk_record["index"])
        chunk_rows = _read_smi_strict(
            _bundle_path(manifest_path.parent, chunk_record["path"])
        )
        expected_ids = [molecule_id for _, molecule_id in chunk_rows]
        chunk_root = run_root / f"chunk_{index:04d}"
        result_path = chunk_root / "chunk_result.json"
        if not result_path.is_file():
            raise ChunkManifestError(
                f"Chunk {index} is incomplete; missing {result_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("stage") != "portable_docking_chunk_result"
            or result.get("status") != "complete"
            or result.get("chunk_index") != index
        ):
            raise ChunkManifestError(f"Invalid result record for chunk {index}")
        if result.get("manifest", {}).get("sha256") != manifest_hash:
            raise ChunkManifestError(
                f"Chunk {index} was run from a different manifest"
            )
        if result.get("input_chunk", {}).get("sha256") != chunk_record["sha256"]:
            raise ChunkManifestError(
                f"Chunk {index} result does not match its assigned SMILES chunk"
            )

        scores_path = chunk_root / "scores.csv"
        recorded_hash = result.get("outputs", {}).get("scores_csv", {}).get("sha256")
        if not scores_path.is_file() or runtime.sha256_file(scores_path) != recorded_hash:
            raise ChunkManifestError(f"Chunk {index} scores do not match its result record")
        docking_record = result.get("outputs", {}).get("docking_summary")
        if docking_record is not None:
            docking_path = chunk_root / "poses" / "_dock_summary.json"
            if (
                not docking_path.is_file()
                or runtime.sha256_file(docking_path) != docking_record.get("sha256")
            ):
                raise ChunkManifestError(
                    f"Chunk {index} docking summary does not match its result record"
                )
            docking_summary = json.loads(
                docking_path.read_text(encoding="utf-8")
            )
            if docking_summary.get("stage") != "smina_docking":
                raise ChunkManifestError(
                    f"Chunk {index} has an invalid docking summary stage"
                )
            docking_summaries.append(docking_summary)
        merged_rows.extend(_score_rows_for_merge(scores_path, expected_ids))
        chunk_results.append(result)

    source_rows = _read_smi_strict(
        _bundle_path(manifest_path.parent, manifest["bundle_source"]["path"])
    )
    if [row["molecule_id"] for row in merged_rows] != [
        molecule_id for _, molecule_id in source_rows
    ]:
        raise ChunkManifestError("Merged score order does not match bundled source")

    outdir.mkdir(parents=True)
    scores_path = outdir / "scores.csv"
    _write_csv_rows(scores_path, merged_rows)
    statuses = [row["status"].lower() for row in merged_rows]
    ok = statuses.count("ok")
    cached = statuses.count("cached")
    failed = len(statuses) - ok - cached
    wall_values = [float(result["timing"]["wall_seconds"]) for result in chunk_results]
    slot_hours = sum(
        float(result["timing"].get("estimated_requested_cpu_slot_hours", 0.0))
        for result in chunk_results
    )
    ideal_wall = max(wall_values)
    receptor_record = None
    docking_parameters = {}
    if docking_summaries:
        receptor_record = docking_summaries[0].get("inputs", {}).get(
            "receptor_pdbqt"
        )
        docking_parameters = docking_summaries[0].get("parameters", {})
        protocol_reference = {
            "receptor_sha256": (
                receptor_record.get("sha256")
                if isinstance(receptor_record, dict)
                else None
            ),
            "parameters": {
                key: value
                for key, value in docking_parameters.items()
                if key != "workers"
            },
        }
        for position, docking_summary in enumerate(docking_summaries[1:], 2):
            receptor = docking_summary.get("inputs", {}).get("receptor_pdbqt")
            candidate_protocol = {
                "receptor_sha256": (
                    receptor.get("sha256") if isinstance(receptor, dict) else None
                ),
                "parameters": {
                    key: value
                    for key, value in docking_summary.get("parameters", {}).items()
                    if key != "workers"
                },
            }
            if candidate_protocol != protocol_reference:
                raise ChunkManifestError(
                    f"Scientific docking protocol differs in completed chunk summary {position}"
                )
    summary = {
        "schema_version": runtime.RUNTIME_SCHEMA_VERSION,
        "stage": "smina_docking",
        "execution_mode": "portable_chunk_array",
        "inputs": {
            "manifest": runtime.file_record(manifest_path),
            "chunk_results": {
                "n_files": len(chunk_results),
                "run_root": str(run_root.resolve()),
            },
            "receptor_pdbqt": receptor_record,
        },
        "outputs": {"scores_csv": runtime.file_record(scores_path)},
        "counts": {
            "total": len(merged_rows),
            "ok": ok,
            "cached": cached,
            "failed": failed,
        },
        "parameters": {
            **docking_parameters,
            "n_chunks": len(chunk_results),
            "chunking": manifest["parameters"],
        },
        "timing": {
            "wall_seconds": ideal_wall,
            "workers_requested": manifest["parameters"]["cpus_per_task"],
            "cpu_slots_per_task": config.SMINA_CPU,
            "fresh_successes_per_wall_second": (
                ok / ideal_wall if ideal_wall else None
            ),
            "estimated_requested_cpu_slot_hours": slot_hours,
            "sum_chunk_wall_seconds": sum(wall_values),
            "max_chunk_wall_seconds": ideal_wall,
            "wall_time_interpretation": (
                "max_chunk_wall_seconds is an ideal fully-concurrent lower-bound; "
                "it excludes scheduler queue time, staggered starts, and contention"
            ),
        },
    }
    runtime.write_json_atomic(outdir / "_dock_summary.json", summary)
    merge_record = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "stage": "portable_docking_chunk_merge",
        "status": "complete",
        "counts": summary["counts"],
        "timing": summary["timing"],
        "outputs": {
            "scores_csv": runtime.file_record(scores_path),
            "docking_summary": runtime.file_record(outdir / "_dock_summary.json"),
        },
    }
    runtime.write_json_atomic(outdir / "merge.json", merge_record)
    return merge_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable, resumable docking chunks for laptop or Slurm"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a portable chunk bundle")
    create.add_argument("source_smi", type=Path)
    create.add_argument("outdir", type=Path)
    create.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    create.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK)
    create.add_argument("--memory", default=DEFAULT_MEMORY)
    create.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)

    verify = subparsers.add_parser("verify", help="verify bundle hashes and order")
    verify.add_argument("manifest", type=Path)

    run = subparsers.add_parser("run", help="prepare and dock one chunk")
    run.add_argument("manifest", type=Path)
    run.add_argument("index", type=int)
    run.add_argument("run_root", type=Path)
    run.add_argument("--workers", type=int, default=config.WORKERS)

    merge = subparsers.add_parser("merge", help="merge completed chunk results")
    merge.add_argument("manifest", type=Path)
    merge.add_argument("run_root", type=Path)
    merge.add_argument("outdir", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create_manifest(
                args.source_smi,
                args.outdir,
                chunk_size=args.chunk_size,
                cpus_per_task=args.cpus_per_task,
                memory=args.memory,
                time_limit=args.time_limit,
            )
            print(
                f"Created {result['counts']['chunks']} chunks for "
                f"{result['counts']['molecules']} molecules in {args.outdir}"
            )
        elif args.command == "verify":
            result = verify_manifest(args.manifest)
            print(
                f"PASS: {result['molecules']} molecules across "
                f"{result['chunks']} verified chunks"
            )
        elif args.command == "run":
            result = run_chunk(
                args.manifest,
                args.index,
                args.run_root,
                workers=args.workers,
            )
            print(
                f"Chunk {args.index} complete: "
                f"{result['counts']['successful_scores']}/"
                f"{result['counts']['expected']} scored"
            )
        else:
            result = merge_chunks(args.manifest, args.run_root, args.outdir)
            print(
                f"Merged {result['counts']['total']} rows into "
                f"{args.outdir / 'scores.csv'}"
            )
    except ChunkManifestError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
