"""Record provenance for an externally downloaded source library.

Files that arrive by wget have no summary.json, because no module in this
repo produced them. This script writes the equivalent record by hand, once,
and fails loudly if the download does not match the publisher's checksum.
"""
import argparse
import gzip
from datetime import datetime, timezone
from pathlib import Path

from src.harness.runtime import file_record, write_json_atomic


def gz_header_and_rows(path):
    """Return the header line and the number of data rows below it."""
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n")
        n_rows = sum(1 for _ in handle)
    return header, n_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--doi", required=True)
    p.add_argument("--publisher-checksum", required=True)
    p.add_argument("--publisher-checksum-algo", required=True,
                   choices=["sha256", "md5"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    record = file_record(args.path)

    if args.publisher_checksum_algo == "sha256":
        if record["sha256"].lower() != args.publisher_checksum.lower():
            raise SystemExit(
                f"CHECKSUM MISMATCH\n"
                f"  publisher: {args.publisher_checksum}\n"
                f"  computed:  {record['sha256']}\n"
                f"The download is corrupt or truncated. Do not use this file."
            )
        checksum_verified = True
    else:
        checksum_verified = False

    header, n_rows = gz_header_and_rows(args.path)

    provenance = {
        "schema_version": 1,
        "stage": "external_source_download",
        "name": args.name,
        "release": args.release,
        "url": args.url,
        "doi": args.doi,
        "file": record,
        "publisher_checksum": args.publisher_checksum,
        "publisher_checksum_algorithm": args.publisher_checksum_algo,
        "publisher_checksum_verified": checksum_verified,
        "downloaded_utc": datetime.fromtimestamp(
            args.path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "header_line": header,
        "columns": header.split("\t"),
        "n_data_rows": n_rows,
        "interpretation": (
            "Unfiltered ChEMBL structure dump. No drug-likeness, size, or "
            "assay filtering applied by the publisher. Not a training-set "
            "holdout for any benchmark arm."
        ),
    }

    write_json_atomic(args.out, provenance)
    print(f"wrote {args.out}  ({n_rows:,} data rows)")


if __name__ == "__main__":
    main()