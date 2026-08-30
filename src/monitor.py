"""Live, read-only terminal monitor for immutable benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.harness import config


SPINNER = ("|", "/", "-", "\\")
ARM_DIRS = (
    ("Baseline", "naive_property_matched"),
    ("Gen1", "gen1_guacamol"),
    ("Gen2", "gen2_warmmolgenone"),
    ("Gen3", "gen3_molexar"),
)
DEFAULT_RESULTS_DIR = config.REPO_ROOT / "results"
# One of these marks a run directory that reached a durable summary.
RUN_MARKERS = (
    "pipeline_summary.json",
    "campaign_summary.json",
    "all_arms_summary.json",
    "run_local.json",
    "metrics.json",
)
# A run that has only just started has no summary yet, so it is recognized by
# the first artifacts its runner writes instead.
IN_PROGRESS_MARKERS = (
    "cohort.json",
    "subsample.json",
    "intake",
    "gate",
    "prepared",
    "docking",
    "ligands",
    "samples_*",
    "screening_*",
    *(directory for _, directory in ARM_DIRS),
)
# run_local writes ligands/<cohort> and docking/<cohort>; runs from before that
# rename (results/smoke_5x250) used lig_<cohort> and dock_<cohort> instead.
LEGACY_COHORT_PREFIXES = {"ligands": "lig", "docking": "dock"}
COHORT_STAGES = (("actives", "actives"), ("decoys", "decoys"))


class MonitorError(ValueError):
    """Raised when a run cannot be monitored."""


def _is_run_directory(path: Path) -> bool:
    """True for a finished run, a live run, or a hidden .partial working copy."""

    if not path.is_dir():
        return False
    if path.name.startswith(".") and path.name.endswith(".partial"):
        return True
    if any((path / marker).is_file() for marker in RUN_MARKERS):
        return True
    if (path.parent / f"{path.name}.console.log").is_file():
        return True
    for marker in IN_PROGRESS_MARKERS:
        if "*" in marker:
            if any(path.glob(marker)):
                return True
        elif (path / marker).exists():
            return True
    return False


def _run_mtime(path: Path) -> float:
    """Latest activity: the directory itself, its entries, or its console log."""

    times = [path.stat().st_mtime]
    log = path.parent / f"{path.name}.console.log"
    if log.is_file():
        times.append(log.stat().st_mtime)
    try:
        times.extend(entry.stat().st_mtime for entry in path.iterdir())
    except OSError:
        pass
    return max(times)


def latest_run(results_dir: Path | None = None) -> Path:
    """Return the most recently active run directory, finished or in progress."""

    root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    if not root.is_dir():
        raise MonitorError(f"No results directory to search: {root}")
    candidates: list[tuple[float, Path]] = []
    for entry in sorted(root.iterdir()):
        try:
            if _is_run_directory(entry):
                candidates.append((_run_mtime(entry), entry))
        except OSError:
            continue
    if not candidates:
        raise MonitorError(
            f"No run directories found under {root}; name one explicitly"
        )
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _screening_root(root: Path) -> Path | None:
    if (root / "subsample.json").is_file() or (root / "intake").is_dir():
        return root
    candidates = sorted(
        (path for path in root.glob("screening_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _is_cohort_run(root: Path) -> bool:
    """True for a labelled run directory built by build_cohort + run_local."""

    if (root / "cohort.json").is_file():
        return True
    return (root / "actives.smi").is_file() and (root / "decoys.smi").is_file()


def _cohort_directory(root: Path, stage: str, cohort: str) -> Path | None:
    """Locate ligands/<cohort> or docking/<cohort>, tolerating the legacy names."""

    current = root / stage / cohort
    if current.is_dir():
        return current
    legacy = root / f"{LEGACY_COHORT_PREFIXES[stage]}_{cohort}"
    return legacy if legacy.is_dir() else None


def _sample_root(root: Path) -> Path | None:
    candidates = sorted(
        (path for path in root.glob("samples_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _live_count(directory: Path | None, pattern: str) -> int:
    """Count artifacts already on disk, for a stage with no summary yet."""

    if directory is None or not directory.is_dir():
        return 0
    try:
        return sum(1 for _ in directory.glob(pattern))
    except OSError:
        return 0


def _prepared_count(directory: Path | None, summary: dict | None) -> int:
    """Prepared ligands, from the durable summary or the PDBQT files themselves."""

    return _count(
        summary,
        "counts",
        "successful_pdbqt_available",
        default=_live_count(directory, "*.pdbqt"),
    )


def _scored_count(directory: Path | None, summary: dict | None) -> int:
    """Docked ligands, from the durable summary or the poses themselves."""

    if summary is None:
        return _live_count(directory, "*_out.pdbqt")
    return _count(summary, "counts", "ok") + _count(summary, "counts", "cached")


def _count(record: dict | None, *keys: str, default: int = 0) -> int:
    value: object = record or {}
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stage(label: str, complete: bool, current: int = 0, total: int = 0, detail: str = "") -> dict:
    return {
        "label": label,
        "complete": bool(complete),
        "state": "pending",
        "current": int(current),
        "total": int(total),
        "detail": detail,
    }


def _apply_states(stages: list[dict], run_complete: bool) -> None:
    if run_complete:
        for stage in stages:
            stage["state"] = "complete"
        return
    active_assigned = False
    for stage in stages:
        if stage["complete"]:
            stage["state"] = "complete"
        elif not active_assigned:
            stage["state"] = "current"
            active_assigned = True
        else:
            stage["state"] = "pending"


def _single_snapshot(root: Path, label: str | None = None) -> dict:
    root = Path(root)
    campaign = _json(root / "campaign_summary.json")
    direct_pipeline = _json(root / "pipeline_summary.json")
    screening = _screening_root(root)
    pipeline = direct_pipeline or (
        _json(screening / "pipeline_summary.json") if screening else None
    )
    run_complete = campaign is not None or pipeline is not None

    sample = _sample_root(root)
    sampling = _json(sample / "sampling.json") if sample else None
    generation_total = _count(sampling, "counts", "raw_samples")
    if generation_total == 0 and sample:
        generation_total = _line_count(sample / "molecules.smi")
    source_complete = sampling is not None
    source_label = "Generation"
    if sample is None:
        source_label = "Source sample"
        source_complete = bool(screening and (screening / "subsample.json").is_file())

    subsample = _json(screening / "subsample.json") if screening else None
    submitted = _count(subsample, "selected_rows")
    if submitted == 0 and screening:
        submitted = _line_count(screening / "submission.smi")
    intake = _json(screening / "intake/summary.json") if screening else None
    accepted = _count(intake, "counts", "accepted_for_preparation")
    gate = _json(screening / "gate/gate_summary.json") if screening else None
    passed = _count(gate, "n_passed")
    prepared_dir = screening / "prepared" if screening else None
    preparation = _json(prepared_dir / "_prep_summary.json") if prepared_dir else None
    prepared_live = _live_count(prepared_dir, "*.pdbqt")
    prepared = _prepared_count(prepared_dir, preparation)
    docking_dir = screening / "docking" if screening else None
    docking = _json(docking_dir / "_dock_summary.json") if docking_dir else None
    docked_live = _live_count(docking_dir, "*_out.pdbqt")
    scored = _scored_count(docking_dir, docking)

    stages = [
        _stage(
            source_label,
            source_complete,
            generation_total if source_complete else 0,
            generation_total,
            f"{generation_total:,} raw rows" if generation_total else "authenticated input",
        ),
        _stage("Intake", intake is not None, accepted, submitted, f"{accepted:,} accepted" if intake else ""),
        _stage("Gate", gate is not None, passed, accepted, f"{passed:,} passed" if gate else ""),
        _stage("Prepare", preparation is not None, prepared, passed, f"{prepared:,} PDBQT" if prepared_live or preparation else ""),
        _stage("Dock", docking is not None, scored, prepared, f"{scored:,} scored" if docked_live or docking else ""),
        _stage("Analyze", run_complete, 1 if run_complete else 0, 1, "reports written" if run_complete else ""),
    ]
    _apply_states(stages, run_complete)
    funnel = (campaign or pipeline or {}).get("funnel", {})
    completed_stages = sum(stage["state"] == "complete" for stage in stages)
    active = next((stage["label"] for stage in stages if stage["state"] == "current"), "Complete")
    return {
        "label": label or root.name,
        "root": root.resolve(),
        "complete": run_complete,
        "stages": stages,
        "completed_stages": completed_stages,
        "active_stage": active,
        "funnel": funnel,
    }


def _cohort_snapshot(root: Path, label: str | None = None) -> dict:
    """Snapshot a labelled run in run_local's order: cohort, actives, decoys."""

    root = Path(root)
    cohort = _json(root / "cohort.json")
    completion = _json(root / "run_local.json")
    run_complete = completion is not None

    # cohort.json records the selection; the .smi files are the fallback while
    # it is still being written, and the only source for a legacy layout.
    totals = {
        name: (
            _count(cohort, "selection", f"n_{name}")
            or _line_count(root / f"{name}.smi")
        )
        for name, _ in COHORT_STAGES
    }
    selected = totals["actives"] + totals["decoys"]
    # A legacy run has no cohort.json, but its .smi files are the built cohort.
    cohort_built = cohort is not None or (
        selected > 0
        and (root / "actives.smi").is_file()
        and (root / "decoys.smi").is_file()
    )
    stages = [
        _stage(
            "Cohort",
            cohort_built,
            selected if cohort_built else 0,
            selected,
            (
                f"{totals['actives']:,} actives + {totals['decoys']:,} decoys"
                if selected
                else ""
            ),
        )
    ]
    for name, title in COHORT_STAGES:
        ligand_dir = _cohort_directory(root, "ligands", name)
        pose_dir = _cohort_directory(root, "docking", name)
        preparation = _json(ligand_dir / "_prep_summary.json") if ligand_dir else None
        docking = _json(pose_dir / "_dock_summary.json") if pose_dir else None
        prepared_live = _live_count(ligand_dir, "*.pdbqt")
        prepared = _prepared_count(ligand_dir, preparation)
        docked_live = _live_count(pose_dir, "*_out.pdbqt")
        scored = _scored_count(pose_dir, docking)
        stages.append(
            _stage(
                f"Prep {title}",
                preparation is not None,
                prepared,
                totals[name],
                f"{prepared:,} PDBQT" if prepared_live or preparation else "",
            )
        )
        stages.append(
            _stage(
                f"Dock {title}",
                docking is not None,
                scored,
                prepared,
                f"{scored:,} scored" if docked_live or docking else "",
            )
        )
    stages.append(
        _stage(
            "Analyze",
            run_complete,
            1 if run_complete else 0,
            1,
            "reports written" if run_complete else "",
        )
    )
    _apply_states(stages, run_complete)
    completed_stages = sum(stage["state"] == "complete" for stage in stages)
    active = next(
        (stage["label"] for stage in stages if stage["state"] == "current"), "Complete"
    )
    return {
        "label": label or root.name,
        "root": root.resolve(),
        "complete": run_complete,
        "stages": stages,
        "completed_stages": completed_stages,
        "active_stage": active,
        "funnel": (completion or {}).get("funnel", {}),
    }


def _log_tail(path: Path) -> tuple[str, str | None, int | None]:
    if not path.is_file():
        return "", None, None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 48_000))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return "", None, None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    latest = next(
        (
            line
            for line in reversed(lines)
            if not line.startswith(("Repository:", "Command:", "Started:"))
        ),
        "",
    )
    started_match = re.search(r"^Started:\s*(.+)$", text, flags=re.MULTILINE)
    started = started_match.group(1).strip() if started_match else None
    exit_matches = re.findall(r"^Exit code:\s*(\d+)", text, flags=re.MULTILINE)
    exit_code = int(exit_matches[-1]) if exit_matches else None
    return latest, started, exit_code


def snapshot_run(path: Path) -> dict:
    """Build one read-only monitoring snapshot."""

    path = Path(path)
    if not path.exists():
        raise MonitorError(f"Run path does not exist yet: {path}")
    all_summary = _json(path / "all_arms_summary.json")
    arms = []
    has_all_layout = all_summary is not None or any((path / directory).exists() for _, directory in ARM_DIRS)
    if has_all_layout:
        for label, directory in ARM_DIRS:
            arm_path = path / directory
            if arm_path.exists():
                arms.append(_single_snapshot(arm_path, label))
            else:
                pending = _single_snapshot(path / directory, label)
                arms.append(pending)
        complete = all_summary is not None
        mode = "all"
    elif _is_cohort_run(path):
        arms = [_cohort_snapshot(path)]
        complete = arms[0]["complete"]
        mode = "cohort"
    else:
        arms = [_single_snapshot(path)]
        complete = arms[0]["complete"]
        mode = "single"
    log_path = path.parent / f"{path.name}.console.log"
    latest, started, exit_code = _log_tail(log_path)
    failed = exit_code is not None and exit_code != 0
    return {
        "path": path.resolve(),
        "mode": mode,
        "arms": arms,
        "complete": complete,
        "failed": failed,
        "exit_code": exit_code,
        "latest_log": latest,
        "started": started,
        "log_path": log_path.resolve(),
    }


def _bar(current: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return "[" + "·" * width + "]"
    fraction = max(0.0, min(1.0, current / total))
    filled = round(width * fraction)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _elapsed(started: str | None) -> str:
    if not started:
        return "not recorded"
    try:
        start = datetime.fromisoformat(started)
        seconds = max(0, int((datetime.now(timezone.utc) - start).total_seconds()))
    except (TypeError, ValueError):
        return "not recorded"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_snapshot(snapshot: dict, frame: int = 0) -> str:
    """Render a clean terminal snapshot with honest determinate/indeterminate bars."""

    spinner = SPINNER[frame % len(SPINNER)]
    if snapshot["failed"]:
        state = f"FAILED (exit {snapshot['exit_code']})"
        spinner = "!"
    elif snapshot["complete"]:
        state = "COMPLETE"
        spinner = "✓"
    else:
        state = "RUNNING"
    lines = [
        f"DYRK1A BENCHMARK  {spinner}  {state}",
        "═" * 66,
        f"Run      {snapshot['path']}",
        f"Elapsed  {_elapsed(snapshot.get('started'))}",
        f"Log      {snapshot['log_path']}",
        "",
    ]
    if snapshot["mode"] == "all":
        lines.append("FOUR-ARM CAMPAIGN")
        active_index = next(
            (index for index, arm in enumerate(snapshot["arms"]) if not arm["complete"]),
            len(snapshot["arms"]) - 1,
        )
        for index, arm in enumerate(snapshot["arms"]):
            completed = arm["completed_stages"]
            glyph = "✓" if arm["complete"] else (spinner if index == active_index else "·")
            lines.append(
                f" {glyph} {arm['label']:<10} {_bar(completed, 6, 18)} "
                f"{completed}/6  {arm['active_stage']}"
            )
        active_arm = snapshot["arms"][active_index]
        lines.extend(("", f"ACTIVE DETAIL — {active_arm['label']}"))
    else:
        active_arm = snapshot["arms"][0]
        heading = "LABELLED RUN" if snapshot["mode"] == "cohort" else "PIPELINE"
        lines.append(f"{heading} — {active_arm['label']}")
    for stage in active_arm["stages"]:
        glyph = "✓" if stage["state"] == "complete" else (spinner if stage["state"] == "current" else "·")
        if stage["state"] == "current" and stage["total"] <= 0:
            progress = "[" + SPINNER[frame % len(SPINNER)].center(22, "·") + "]"
        else:
            progress = _bar(stage["current"], stage["total"])
        detail = stage["detail"] or stage["state"]
        lines.append(f" {glyph} {stage['label']:<13} {progress}  {detail}")
    if snapshot.get("latest_log"):
        latest = snapshot["latest_log"]
        if len(latest) > 110:
            latest = latest[:107] + "..."
        lines.extend(("", "LATEST OUTPUT", f"  {latest}"))
    lines.extend(("", "Ctrl+C stops the monitor only; it does not stop the benchmark run."))
    return "\n".join(lines)


def watch_run(path: Path, *, interval: float = 1.0, once: bool = False) -> int:
    """Refresh until completion/failure, or print one snapshot for automation."""

    if interval < 0.2:
        raise MonitorError("Monitor interval must be at least 0.2 seconds")
    frame = 0
    interactive = sys.stdout.isatty() and not once
    try:
        while True:
            snapshot = snapshot_run(path)
            rendered = render_snapshot(snapshot, frame)
            if interactive:
                print("\033[2J\033[H", end="")
            print(rendered, flush=True)
            if once or snapshot["complete"] or snapshot["failed"]:
                return 1 if snapshot["failed"] else 0
            frame += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor closed. The benchmark process was not interrupted.")
        return 130


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an immutable benchmark run")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="run directory; omit to watch the most recent run under results/",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    path = args.path
    if path is None:
        try:
            path = latest_run()
        except MonitorError as error:
            raise SystemExit(f"error: {error}") from error
        print(f"Watching the most recent run: {path}")
    try:
        raise SystemExit(watch_run(path, interval=args.interval, once=args.once))
    except MonitorError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
