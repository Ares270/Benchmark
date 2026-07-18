"""Shared provenance and computational-cost records for harness stages."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Iterable


RUNTIME_SCHEMA_VERSION = 1





####### calc file fingerprint ########

def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# package record #

def file_record(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_set_record(paths: Iterable[Path]) -> dict:
    """Hash both file contents and relative names as one reproducible set."""

    files = sorted((Path(path) for path in paths), key=lambda path: str(path))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        file_hash = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "n_files": len(files),
        "total_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }







#### Timing and Hardware Records ####

def hardware_record() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus_available": os.cpu_count(),
    }







######## Computational cost records ########

def timing_record(
    started: float,
    *,
    attempted_tasks: int,
    workers: int,
    cpu_slots_per_task: int = 1,
) -> dict:
    wall_seconds = time.perf_counter() - started
    occupied_workers = min(max(workers, 0), attempted_tasks)
    requested_slots = occupied_workers * max(cpu_slots_per_task, 0)    ### Records
    return {                                                                # Wall time
        "wall_seconds": wall_seconds,                                       # Tasks per wall-second
        "attempted_tasks_per_wall_second": (                                # Requested workers
            attempted_tasks / wall_seconds if wall_seconds else None        # CPU slots per task
        ),                                                                  # Maximum simultaneous requested CPU slots
        "workers_requested": workers,                                       # Estimated requested CPU-slot hours
        "cpu_slots_per_task": cpu_slots_per_task,
        "maximum_concurrent_cpu_slots_requested": (
            workers * cpu_slots_per_task
        ),
        "estimated_requested_cpu_slot_hours": (
            wall_seconds * requested_slots / 3600
        ),
        "cpu_accounting_note": (
            "Requested slot-hours are a wall-time estimate, not measured child-"
            "process CPU consumption."
        ),
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
