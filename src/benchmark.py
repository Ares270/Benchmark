"""Human-facing launcher and status console for the locked DYRK1A benchmark.

This module does not implement scientific stages. It translates short, explicit
commands into the existing authenticated runners, performs read-only preflight
checks, and summarizes immutable run directories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.generation.gen1_guacamol import (
    OFFICIAL_CHECKPOINT_BYTES,
    OFFICIAL_CHECKPOINT_SHA256,
)
from src.generation.gen2_warmmolgenone import (
    load_gen2_config,
    verify_checkpoint as verify_gen2_checkpoint,
)
from src.generation.gen3_molexar import (
    load_gen3_config,
    verify_checkpoint as verify_gen3_checkpoint,
    verify_source_tree as verify_gen3_source_tree,
)
from src.generation.run_all_arms import (
    DEFAULT_BASELINE,
    DEFAULT_GEN1_CHECKPOINT,
    DEFAULT_GEN2_MODEL,
    DEFAULT_GEN2_TOKENIZER,
    DEFAULT_GEN3_MODEL,
)
from src.generation.run_gen3_pipeline import DEFAULT_MOLEXAR_PYTHON
from src.harness import config, runtime


REGISTERED_SEED = 20260801
REGISTERED_RAW_N = 10_000
REGISTERED_DOCK_N = 1_000
PILOT_RAW_DEFAULT = 100
PILOT_DOCK_DEFAULT = 10
DEFAULT_ACTIVES = config.REFERENCE_DIR / "dyrk1a_actives_chembl.csv"
DEFAULT_GEN3_SOURCES = config.REPO_ROOT / "Models & Miscellaneous/arm3_sources"


class BenchmarkConsoleError(ValueError):
    """Raised for a human-facing launcher or status error."""


@dataclass(frozen=True)
class Check:
    label: str
    action: Callable[[], str]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkConsoleError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkConsoleError(f"{path} must contain one JSON object")
    return value


def _file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise BenchmarkConsoleError(f"missing: {path}")
    return f"{path} ({path.stat().st_size / (1024 * 1024):.1f} MiB)"


def _directory(path: Path) -> str:
    path = Path(path)
    if not path.is_dir():
        raise BenchmarkConsoleError(f"missing directory: {path}")
    return str(path)


def _gen1_checkpoint() -> str:
    path = Path(DEFAULT_GEN1_CHECKPOINT)
    _file(path)
    if path.stat().st_size != OFFICIAL_CHECKPOINT_BYTES:
        raise BenchmarkConsoleError(
            f"Gen1 byte mismatch: expected {OFFICIAL_CHECKPOINT_BYTES}, "
            f"observed {path.stat().st_size}"
        )
    observed = runtime.sha256_file(path)
    if observed != OFFICIAL_CHECKPOINT_SHA256:
        raise BenchmarkConsoleError(
            f"Gen1 SHA-256 mismatch: expected {OFFICIAL_CHECKPOINT_SHA256}, "
            f"observed {observed}"
        )
    return f"authenticated · {observed[:12]}…"


def _gen2_checkpoint() -> str:
    specification = load_gen2_config()
    record = verify_gen2_checkpoint(DEFAULT_GEN2_MODEL, specification)
    artifacts = specification["tokenizers"]["smiles"]["artifacts"]
    for filename, expected in artifacts.items():
        path = Path(DEFAULT_GEN2_TOKENIZER) / filename
        _file(path)
        if path.stat().st_size != int(expected["bytes"]):
            raise BenchmarkConsoleError(f"Gen2 tokenizer byte mismatch: {path}")
        if runtime.sha256_file(path) != expected["sha256"]:
            raise BenchmarkConsoleError(f"Gen2 tokenizer SHA-256 mismatch: {path}")
    return f"authenticated · {record['weights']['sha256'][:12]}…"


def _gen3_checkpoint() -> str:
    specification = load_gen3_config()
    record = verify_gen3_checkpoint(DEFAULT_GEN3_MODEL, specification)
    verify_gen3_source_tree(DEFAULT_GEN3_SOURCES, specification)
    return f"authenticated · {record['weights']['sha256'][:12]}…"


def _executable(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise BenchmarkConsoleError(f"missing executable: {path}")
    completed = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConsoleError(f"cannot execute {path}")
    return (completed.stdout or completed.stderr).strip()


def _smina() -> str:
    discovered = shutil.which("smina")
    environment_copy = Path(sys.executable).resolve().parent / "smina"
    path = discovered or (
        str(environment_copy) if environment_copy.is_file() else None
    )
    if path is None:
        raise BenchmarkConsoleError(
            "smina is not on PATH; activate the dyrk1a-bench environment"
        )
    completed = subprocess.run(
        [path, "--version"], check=False, capture_output=True, text=True
    )
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return f"{path} · {text[0] if text else 'version command available'}"


def doctor() -> bool:
    """Authenticate tomorrow's inputs without loading or sampling a model."""

    checks = (
        Check("Repository", lambda: _directory(config.REPO_ROOT)),
        Check("Smina", _smina),
        Check("Receptor", lambda: _file(config.RECEPTOR_PDBQT)),
        Check(
            "7O7K pocket PDB",
            lambda: _file(config.TARGET_DIR / "7O7K_protein_noH.pdb"),
        ),
        Check("DYRK1A actives", lambda: _file(DEFAULT_ACTIVES)),
        Check("Naive baseline", lambda: _file(DEFAULT_BASELINE)),
        Check("Gen1 checkpoint", _gen1_checkpoint),
        Check("Gen2 bundle", _gen2_checkpoint),
        Check("Gen3 bundle", _gen3_checkpoint),
        Check("Gen3 isolated runtime", lambda: _executable(DEFAULT_MOLEXAR_PYTHON)),
    )
    print("DYRK1A benchmark preflight")
    print(f"Repository: {config.REPO_ROOT}")
    failures = 0
    for check in checks:
        try:
            detail = check.action()
        except Exception as error:
            failures += 1
            print(f"  [FAIL] {check.label}: {error}")
        else:
            print(f"  [ OK ] {check.label}: {detail}")
    if failures:
        print(f"\nPreflight failed: {failures} check(s) need attention.")
        return False
    print("\nReady: all four registered arms can be launched.")
    return True


def _safe_name(outdir: Path, arm: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(outdir).name).strip("._-")
    if not value:
        value = f"{arm}_{REGISTERED_SEED}"
    return value[:80]


def build_run_command(
    arm: str,
    outdir: Path,
    *,
    pilot: bool,
    raw_n: int,
    dock_n: int,
    workers: int,
) -> list[str]:
    """Translate a short launcher request into one existing locked runner."""

    if workers < 1:
        raise BenchmarkConsoleError("workers must be at least 1")
    if raw_n < 1 or dock_n < 1 or dock_n > raw_n:
        raise BenchmarkConsoleError("Require 1 <= dock_n <= raw_n")
    if not pilot and (
        raw_n != PILOT_RAW_DEFAULT or dock_n != PILOT_DOCK_DEFAULT
    ):
        raise BenchmarkConsoleError(
            "--raw-n and --dock-n are pilot-only; add --pilot or remove them"
        )
    python = sys.executable
    outdir = Path(outdir)
    if arm == "all":
        if pilot:
            raise BenchmarkConsoleError(
                "The all-arm launcher is registered-only; run each arm with "
                "--pilot for a plumbing check"
            )
        return [
            python,
            "-m",
            "src.generation.run_all_arms",
            str(outdir),
            "--workers",
            str(workers),
        ]
    if arm == "baseline":
        submitted = dock_n if pilot else REGISTERED_DOCK_N
        return [
            python,
            "-m",
            "src.generation.run_candidate_pipeline",
            str(DEFAULT_BASELINE),
            str(outdir),
            "--name",
            _safe_name(outdir, arm),
            "--role",
            "naive_baseline" if not pilot else "pilot",
            "--n",
            str(submitted),
            "--seed",
            str(REGISTERED_SEED),
            "--workers",
            str(workers),
            "--analysis-outdir",
            str(outdir.parent / f"{outdir.name}_analysis"),
            "--source-description",
            (
                "Property-matched naive baseline from ChEMBL 37; exact known-active "
                "parents excluded with RDKit-computed parent InChIKeys; uniform raw "
                f"{submitted}-row branch; seed {REGISTERED_SEED}; no top-up"
            ),
        ]

    commands = {
        "gen1": [
            python,
            "-m",
            "src.generation.run_gen1_pipeline",
            str(DEFAULT_GEN1_CHECKPOINT),
            str(outdir),
        ],
        "gen2": [
            python,
            "-m",
            "src.generation.run_gen2_pipeline",
            str(DEFAULT_GEN2_MODEL),
            str(DEFAULT_GEN2_TOKENIZER),
            str(outdir),
        ],
        "gen3": [
            python,
            "-m",
            "src.generation.run_gen3_pipeline",
            str(DEFAULT_GEN3_MODEL),
            str(outdir),
        ],
    }
    try:
        command = commands[arm]
    except KeyError as error:
        raise BenchmarkConsoleError(f"Unknown arm: {arm}") from error
    command.extend(
        ["--seed", str(REGISTERED_SEED), "--device", "cpu", "--workers", str(workers)]
    )
    if pilot:
        command.extend(
            ["--validation", "--raw-n", str(raw_n), "--dock-n", str(dock_n)]
        )
    return command


def launch(command: list[str], *, dry_run: bool, log_path: Path | None = None) -> int:
    """Run one existing pipeline with live output and an optional durable log."""

    print("Command:")
    print(f"  {shlex.join(command)}")
    if dry_run:
        print("Dry run only; nothing was started.")
        return 0
    print("\nStarting. Output is immutable; use a new directory for another attempt.")
    destination = Path(log_path).resolve() if log_path is not None else None
    log_handle = None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_handle = destination.open("x", encoding="utf-8", buffering=1)
        except OSError as error:
            raise BenchmarkConsoleError(
                f"Cannot create live log {destination}: {error}"
            ) from error
        log_handle.write(
            f"Started: {datetime.now(timezone.utc).isoformat()}\n"
            f"Repository: {config.REPO_ROOT}\n"
            f"Command: {shlex.join(command)}\n\n"
        )
        print(f"Live log: {destination}\n")
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=config.REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_handle is not None:
                log_handle.write(line)
        return_code = int(process.wait())
        footer = (
            f"\nFinished: {datetime.now(timezone.utc).isoformat()}\n"
            f"Exit code: {return_code}\n"
        )
        if log_handle is not None:
            log_handle.write(footer)
        print(footer, end="")
        return return_code
    except OSError as error:
        message = f"\nCould not start pipeline: {error}\n"
        if log_handle is not None:
            log_handle.write(message)
        print(message, end="")
        raise BenchmarkConsoleError(
            f"Could not start pipeline command: {error}"
        ) from error
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        message = (
            "\nInterrupted by user (exit 130). Partial artifacts were preserved.\n"
        )
        if log_handle is not None:
            log_handle.write(message)
        print(message, end="")
        return 130
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if log_handle is not None:
            log_handle.close()


def _print_funnel(funnel: dict) -> None:
    fields = (
        ("submitted", "submitted"),
        ("accepted_at_intake", "intake"),
        ("passed_predock_gate", "gate"),
        ("prepared_pdbqt_available", "prepared"),
        ("successfully_scored", "scored"),
    )
    parts = [
        f"{label}={int(funnel[key])}" for key, label in fields if key in funnel
    ]
    if parts:
        print("  Funnel: " + " → ".join(parts))


def _campaign_status(root: Path, summary_path: Path) -> None:
    summary = _read_json(summary_path)
    print(f"  State: COMPLETE ({summary.get('stage', 'campaign')})")
    print(f"  Name: {summary.get('name', summary.get('model_name', root.name))}")
    _print_funnel(summary.get("funnel", {}))

def _referenced_reports(value: object, root: Path) -> set[Path]:

    """Find report paths recorded in a summary, including sibling analyses."""

    reports: set[Path] = set()
    if isinstance(value, dict):
        for item in value.values():
            reports.update(_referenced_reports(item, root))
    elif isinstance(value, list):
        for item in value:
            reports.update(_referenced_reports(item, root))
    elif isinstance(value, str) and value.lower().endswith(".html"):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            reports.add(candidate.resolve())
    return reports


def status(path: Path) -> None:
    """Summarize a complete or interrupted run directory without mutation."""

    path = Path(path).resolve()
    if not path.exists():
        raise BenchmarkConsoleError(f"Run path does not exist: {path}")
    print(f"DYRK1A run status\nPath: {path}")
    all_summary = path / "all_arms_summary.json"
    campaign_summary = path / "campaign_summary.json"
    pipeline_summary = path / "pipeline_summary.json"
    summaries: list[dict] = []
    if all_summary.is_file():
        summary = _read_json(all_summary)
        summaries.append(summary)
        print("  State: COMPLETE (registered four-arm campaign)")
        print(f"  Submitted per arm: {summary.get('submitted_per_arm', 'n/a')}")
    elif campaign_summary.is_file():
        summaries.append(_read_json(campaign_summary))
        _campaign_status(path, campaign_summary)
    elif pipeline_summary.is_file():
        summary = _read_json(pipeline_summary)
        summaries.append(summary)
        print("  State: COMPLETE (candidate pipeline)")
        print(f"  Name: {summary.get('name', path.name)}")
        _print_funnel(summary.get("funnel", {}))
    else:
        markers = (
            ("samples_*", "generation output"),
            ("intake/summary.json", "intake"),
            ("gate/gate_summary.json", "gate"),
            ("prepared/_prep_summary.json", "preparation"),
            ("docking/_dock_summary.json", "docking"),
            ("full_cohort_intake/summary.json", "full-cohort intake"),
            ("full_cohort_analysis/metrics.json", "full-cohort analysis"),
            ("screening_*/intake/summary.json", "screening intake"),
            ("screening_*/gate/gate_summary.json", "gate"),
            ("screening_*/prepared/_prep_summary.json", "preparation"),
            ("screening_*/docking/_dock_summary.json", "docking"),
        )
        found = [label for pattern, label in markers if list(path.glob(pattern))]
        print("  State: INCOMPLETE")
        print("  Completed artifacts: " + (", ".join(found) if found else "none"))
        print("  Inspect the terminal/error log before deciding how to resume.")

    report_set = {report.resolve() for report in path.rglob("report.html")}
    for summary in summaries:
        report_set.update(_referenced_reports(summary, path))
    reports = sorted(report_set)
    if reports:
        print("  Reports:")
        for report in reports:
            print(f"    - {report}")
    else:
        print("  Reports: none yet")

def _prompt_choice(
    title: str,
    choices: tuple[str, ...],
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn(title)
    for index, choice in enumerate(choices, 1):
        suffix = "  (recommended)" if index == 1 else ""
        output_fn(f"  {index}. {choice}{suffix}")
    answer = input_fn("Select: ").strip().lower()
    if not answer:
        return choices[0]
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1]
    if answer in choices:
        return answer
    raise BenchmarkConsoleError(
        f"Unknown selection {answer!r}; choose 1-{len(choices)}"
    )


def _prompt_integer(
    label: str,
    default: int,
    *,
    input_fn: Callable[[str], str],
) -> int:
    answer = input_fn(f"{label} [{default}]: ").strip()
    if not answer:
        return default
    try:
        value = int(answer)
    except ValueError as error:
        raise BenchmarkConsoleError(f"{label} must be an integer") from error
    if value < 1:
        raise BenchmarkConsoleError(f"{label} must be at least 1")
    return value


def configure_run(
    *,
    no_start: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[list[str], Path, bool]:
    """Interactively construct a safe command without scientific knobs."""

    output_fn("DYRK1A benchmark run builder")
    output_fn("Scientific settings are locked here; this builder changes only run logistics.\n")
    kind = _prompt_choice(
        "Run type",
        ("registered", "pilot"),
        input_fn=input_fn,
        output_fn=output_fn,
    )
    pilot = kind == "pilot"
    arms = (
        ("all", "baseline", "gen1", "gen2", "gen3")
        if not pilot
        else ("gen3", "gen1", "gen2", "baseline")
    )
    arm = _prompt_choice(
        "Arm",
        arms,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    stamp = datetime.now().strftime("%Y%m%d")
    default_outdir = Path("results") / f"{arm}_{kind}_{stamp}"
    answer = input_fn(f"New output directory [{default_outdir}]: ").strip()
    outdir = Path(answer) if answer else default_outdir
    if outdir.exists():
        raise BenchmarkConsoleError(
            f"Output already exists: {outdir}. Choose a new name."
        )
    workers = _prompt_integer(
        "Workers", config.WORKERS, input_fn=input_fn
    )
    raw_n = PILOT_RAW_DEFAULT
    dock_n = PILOT_DOCK_DEFAULT
    if pilot:
        if arm != "baseline":
            raw_n = _prompt_integer(
                "Raw molecules to generate", PILOT_RAW_DEFAULT, input_fn=input_fn
            )
        dock_n = _prompt_integer(
            "Raw rows to submit downstream", PILOT_DOCK_DEFAULT, input_fn=input_fn
        )
        raw_n = max(raw_n, dock_n)
    command = build_run_command(
        arm,
        outdir,
        pilot=pilot,
        raw_n=raw_n,
        dock_n=dock_n,
        workers=workers,
    )
    output_fn("\nReview")
    output_fn(f"  Type:    {kind}")
    output_fn(f"  Arm:     {arm}")
    output_fn(f"  Output:  {outdir}")
    output_fn(f"  Workers: {workers}")
    if pilot:
        output_fn(f"  Pilot:   raw={raw_n}, downstream={dock_n}")
    output_fn("\nExact command")
    output_fn(f"  {shlex.join(command)}")
    output_fn(
        "\nScientific references: src/harness/config.py, configs/gen2_warmmolgenone.json, "
        "configs/gen3_molexar.json, and README.md."
    )
    if no_start:
        return command, outdir, False
    start = input_fn("\nStart this run now? [y/N]: ").strip().lower() in {"y", "yes"}
    return command, outdir, start


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Friendly console for the locked four-arm DYRK1A benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "doctor",
        help="authenticate models, targets, baseline, Smina, and runtimes",
    )
    run = subparsers.add_parser(
        "run",
        help="launch one locked arm or the registered four-arm campaign",
    )
    configure = subparsers.add_parser(
        "configure",
        help="interactively build a safe registered or pilot command",
    )
    configure.add_argument("--no-start", action="store_true")
    dashboard = subparsers.add_parser(
        "dashboard",
        help="refresh the offline benchmark home page",
    )
    dashboard.add_argument(
        "--results-dir", type=Path, default=config.REPO_ROOT / "results"
    )
    dashboard.add_argument(
        "--output", type=Path, default=config.REPO_ROOT / "results/benchmark_home.html"
    )
    watch = subparsers.add_parser(
        "watch",
        help="show a live read-only terminal monitor for one run",
    )
    watch.add_argument("path", type=Path)
    watch.add_argument("--interval", type=float, default=1.0)
    watch.add_argument("--once", action="store_true")
    run.add_argument("arm", choices=("baseline", "gen1", "gen2", "gen3", "all"))
    run.add_argument("outdir", type=Path)
    run.add_argument(
        "--pilot",
        action="store_true",
        help="run an explicitly labelled small plumbing check",
    )
    run.add_argument("--raw-n", type=int, default=PILOT_RAW_DEFAULT)
    run.add_argument("--dock-n", type=int, default=PILOT_DOCK_DEFAULT)
    run.add_argument("--workers", type=int, default=config.WORKERS)
    run.add_argument("--dry-run", action="store_true")
    inspect = subparsers.add_parser(
        "status",
        help="summarize a complete or interrupted result directory",
    )
    inspect.add_argument("path", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "dashboard":
            from src.analysis.dashboard import DashboardError, write_dashboard

            try:
                output = write_dashboard(args.results_dir, args.output)
            except DashboardError as error:
                raise BenchmarkConsoleError(str(error)) from error
            print(f"Benchmark home: {output}")
            return
        if args.command == "watch":
            from src.monitor import MonitorError, watch_run

            try:
                return_code = watch_run(
                    args.path, interval=args.interval, once=args.once
                )
            except MonitorError as error:
                raise BenchmarkConsoleError(str(error)) from error
            raise SystemExit(return_code)
        if args.command == "configure":
            command, outdir, start = configure_run(no_start=args.no_start)
            if not start:
                print("\nNothing was started. Copy the command above when ready.")
                return
            log_path = outdir.parent / f"{outdir.name}.console.log"
            raise SystemExit(launch(command, dry_run=False, log_path=log_path))
        if args.command == "doctor":
            raise SystemExit(0 if doctor() else 2)
        if args.command == "status":
            status(args.path)
            return
        command = build_run_command(
            args.arm,
            args.outdir,
            pilot=args.pilot,
            raw_n=args.raw_n,
            dock_n=args.dock_n,
            workers=args.workers,
        )
        log_path = args.outdir.parent / f"{args.outdir.name}.console.log"
        return_code = launch(command, dry_run=args.dry_run, log_path=log_path)
        if return_code != 0:
            print(f"Inspect status: python -m src.benchmark status {shlex.quote(str(args.outdir))}")
        raise SystemExit(return_code)
    except BenchmarkConsoleError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
