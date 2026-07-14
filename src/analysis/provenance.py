"""
Reproducibility metadata captured at analysis time.

Pretty much nothing flows through this script

just a witness

"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd





##########   Hashes a file, absolute path   ##############

def file_record(path: Path) -> dict:
    """Return an absolute path, size, and SHA-256 content digest."""        # Hashing file for guaranteed reproducibility 
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}




##########   whats installed   ##############

def software_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name in ("scipy", "matplotlib", "plotly", "rdkit", "jinja2"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(module.__version__)
        except (ImportError, AttributeError):
            versions[module_name] = "unavailable"

    try:
        process = subprocess.run(
            ["smina", "--version"], capture_output=True, text=True, timeout=15, check=False
        )
        text = (process.stdout or process.stderr).strip().splitlines()
        versions["smina"] = text[0] if text else "unknown"
    except (OSError, subprocess.SubprocessError):
        versions["smina"] = "unavailable"
    return versions



##########   commit hash + whether the working tree is dirty  ##############

def git_state(repo_path: Path) -> dict:
    """Capture commit and dirty paths; a commit alone cannot identify local edits."""

    repo_path = Path(repo_path).resolve()

    def run(*args: str) -> subprocess.CompletedProcess[str]:            
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True, text=True, timeout=15, check=False,
        )

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None, "changed_paths": []}

    commit_text = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    status_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": commit_text,
        "dirty": bool(status_lines) if status.returncode == 0 else None,
        "changed_paths": status_lines,
    }




#############    The docking box/seed/exhaustiveness at analysis time      #############

def harness_snapshot() -> dict:
    """Record current harness settings without claiming they generated the inputs."""

    try:
        from src.harness import config as harness_config
    except ImportError:
        return {"available": False, "analysis_time_snapshot_only": True}

    names = (
        "RECEPTOR_PDBQT", "BOX_CENTER", "BOX_SIZE", "EXHAUSTIVENESS",
        "SEED", "NUM_MODES", "ENERGY_RANGE", "RIGID_MACROCYCLES",
    )
    values = {name.lower(): str(getattr(harness_config, name, "n/a")) for name in names}
    return {"available": True, "analysis_time_snapshot_only": True, **values}
