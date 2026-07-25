"""Fetch a reproducible SMILES-only source pool from the public DUD-E site.

The full DUD-E archive is about 2.8 GB because it includes receptors and 3D
files.  This utility downloads only each target's final active and decoy SMILES
files.  DUD-E actives are retained as an exclusion list; they are never added
to the presumed-decoy pool.

The download directory is resumable.  Existing non-empty raw files are reused,
then every input and normalized output is hashed in summary.json.



ALSO, keep in mind DUD-E actually has 102 targets and hasnt changed anything


Usage:
    python -m src.utils.fetch_dude_decoys --outdir data/external/dude_smiles
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.harness import runtime


TARGETS_URL = "https://dude.docking.org/targets"
TARGET_FILE_URL = "https://dude.docking.org//targets/{target}/{filename}"
ACTIVE_FILENAME = "actives_final.ism"
DECOY_FILENAME = "decoys_final.ism"
EXPECTED_TARGETS = 102
FETCH_SCHEMA_VERSION = 1


class DudeFetchError(RuntimeError):
    """Raised when the public source response cannot be audited safely."""







def discover_targets(page_text: str) -> list[str]:
    """Extract the stable target slugs from the DUD-E target-index HTML."""    # Takes the downloaded list-page text,
                                                                               # pattern-matches out the 102
    targets = sorted(                                                          # protein short names (or slugs)
        {                                                                      # dedupes and sorts them
            match.lower()
            for match in re.findall(
                r'href=["\'](?:https?://dude\.docking\.org)?/targets/([A-Za-z0-9]+)["\']',
                page_text,
            )
        }
    )
    return targets







################    Downloads one file    ###############

def _download_bytes(
    url: str,
    *,
    timeout_seconds: float,                                         # Trie, and if it fails,
    retries: int,                                                   # waits and retries up to x times
) -> bytes:                                                         # waiting a bit longer each time
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DYRK1A-Benchmark/1.0 research reproducibility"},
    )
    last_error = None
    for attempt in range(retries + 1):                               # Treats empty responses as a failure and raises an error
        try:                                                         # (Gives up with an error on the last try)
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                data = response.read()
            if not data:
                raise DudeFetchError(f"Empty response from {url}")
            return data
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise DudeFetchError(
        f"Could not download {url} after {retries + 1} attempts: {last_error}"
    )






##########    Atomic Writting    ###########

def _write_bytes_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)



#######    Parsing and splitting the lines of the file into Smiles + ID    #######

def _parse_source_line(
    line: str,
    *,
    source_path: Path,
    line_number: int,
) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) < 2:
        raise DudeFetchError(
            f"{source_path}:{line_number} has no identifier after SMILES"
        )
    return parts[0], "_".join(parts[1:])






#########  Merges a batch of raw files into one output file  ########

def _normalize_files(
    files: list[tuple[str, Path]],
    destination: Path,
    *,
    id_prefix: str,
) -> int:
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for target, source_path in files:
            for line_number, line in enumerate(
                source_path.read_text(encoding="ascii").splitlines(),
                1,
            ):
                parsed = _parse_source_line(
                    line,
                    source_path=source_path,
                    line_number=line_number,
                )
                if parsed is None:
                    continue
                smiles, source_id = parsed
                normalized_id = f"{id_prefix}_{target.upper()}_{source_id}"     # avoiding collisions 
                output.write(f"{smiles} {normalized_id}\n")
                count += 1
    temporary.replace(destination)                                              # and atomic
    return count



















###############     BODY OF THE SCRIPT     ############

def fetch_dude_smiles(
    outdir: Path,
    *,
    targets: list[str] | None = None,                               
    delay_seconds: float = 0.20,
    timeout_seconds: float = 90.0,
    retries: int = 3,
) -> dict:
    """Fetch or resume the DUD-E SMILES source corpus."""                        # IN ORDER

    if delay_seconds < 0:                                                               # Validate the settings are sane
        raise DudeFetchError("delay_seconds cannot be negative")                        # Get the list page
    if timeout_seconds <= 0:                                                            # Check we got exactly 102
        raise DudeFetchError("timeout_seconds must be positive")                        # Pick Targets
    if retries < 0:                                                                     # Loop Downloading all 204 files
        raise DudeFetchError("retries cannot be negative")                              # Merge decoys into pool.smi
                                                                                        # Merge actives into known_dude_actives.smi
    outdir = Path(outdir)                                                               # Build summary.json
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_page_path = outdir / "targets.html"
    if target_page_path.is_file() and target_page_path.stat().st_size > 0:
        target_page = target_page_path.read_text(encoding="utf-8")
    else:
        target_page_bytes = _download_bytes(
            TARGETS_URL,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        _write_bytes_atomic(target_page_path, target_page_bytes)
        target_page = target_page_bytes.decode("utf-8")

    discovered = discover_targets(target_page)
    if len(discovered) != EXPECTED_TARGETS:
        raise DudeFetchError(
            f"Expected {EXPECTED_TARGETS} targets in {TARGETS_URL}, "
            f"found {len(discovered)}; refusing an incomplete source index"
        )

    if targets is None:
        selected_targets = discovered
    else:
        selected_targets = sorted({target.lower() for target in targets})
        unknown = sorted(set(selected_targets) - set(discovered))
        if unknown:
            raise DudeFetchError(f"Unknown DUD-E target slug(s): {unknown}")
        if not selected_targets:
            raise DudeFetchError("At least one target is required")

    raw_active_files = []
    raw_decoy_files = []
    downloaded_files = 0
    reused_files = 0
    for target in selected_targets:
        for filename, collection in (
            (ACTIVE_FILENAME, raw_active_files),
            (DECOY_FILENAME, raw_decoy_files),
        ):
            local_path = raw_dir / f"{target}_{filename}"
            collection.append((target, local_path))
            if local_path.is_file() and local_path.stat().st_size > 0:
                reused_files += 1
                continue
            url = TARGET_FILE_URL.format(
                target=target,
                filename=filename,
            )
            data = _download_bytes(
                url,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            _write_bytes_atomic(local_path, data)
            downloaded_files += 1
            if delay_seconds:
                time.sleep(delay_seconds)

    pool_path = outdir / "pool.smi"
    exclusions_path = outdir / "known_dude_actives.smi"
    n_decoy_rows = _normalize_files(
        raw_decoy_files,
        pool_path,
        id_prefix="DUDE",
    )
    n_active_rows = _normalize_files(
        raw_active_files,
        exclusions_path,
        id_prefix="DUDE_ACTIVE",
    )

    summary = {
        "schema_version": FETCH_SCHEMA_VERSION,
        "stage": "dude_smiles_source_fetch",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "name": "DUD-E",
            "target_index_url": TARGETS_URL,
            "target_file_url_template": TARGET_FILE_URL,
            "citation_doi": "10.1021/jm300687e",
            "license_note": (
                "DUD-E is publicly provided for research; preserve attribution "
                "and do not represent this derived pool as official target output"
            ),
        },
        "selection_scope": {
            "discovered_targets": len(discovered),
            "selected_targets": selected_targets,
            "selected_target_count": len(selected_targets),
            "files_expected": 2 * len(selected_targets),
        },
        "download": {
            "downloaded_files_this_invocation": downloaded_files,
            "reused_nonempty_files": reused_files,
            "delay_seconds_between_new_files": delay_seconds,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        },
        "counts": {
            "raw_decoy_rows": n_decoy_rows,
            "raw_known_active_rows": n_active_rows,
        },
        "outputs": {
            "target_index": runtime.file_record(target_page_path),
            "pool_smi": runtime.file_record(pool_path),
            "known_dude_actives_smi": runtime.file_record(exclusions_path),
            "raw_active_files": runtime.file_set_record(
                [path for _, path in raw_active_files]
            ),
            "raw_decoy_files": runtime.file_set_record(
                [path for _, path in raw_decoy_files]
            ),
        },
        "interpretation": {
            "pool_members_are_presumed_decoys": True,
            "pool_members_are_not_proven_dyrk1a_inactives": True,
            "known_dude_actives_are_exclusions_only": True,
            "local_property_matching_and_dyrk1a_similarity_filter_required": True,
        },
    }
    runtime.write_json_atomic(outdir / "summary.json", summary)
    return summary



























def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch resumable SMILES-only DUD-E source and exclusion pools"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="optional target slug; repeat for a subset (default: all 102)",
    )
    parser.add_argument("--delay-seconds", type=float, default=0.20)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    summary = fetch_dude_smiles(
        args.outdir,
        targets=args.target,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    print(
        f"DUD-E source ready: "
        f"{summary['counts']['raw_decoy_rows']:,} decoy rows | "
        f"{summary['counts']['raw_known_active_rows']:,} known-active exclusions | "
        f"{summary['selection_scope']['selected_target_count']} targets"
    )
    print(f"Outputs: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
