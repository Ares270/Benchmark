"""
Pull DYRK1A active compounds from ChEMBL.

Filters: IC50 ≤ 1000 nM, enzymatic assays, human target, exact measurements.
Deduplicates by compound (median IC50).

Criteria
  - Activity cutoff: 1000 nM (1 µM)
  - Assay type: biochemical only (assay_type 'B')
  - Deduplication: median IC50 across measurements
  - Standard relation: '=' or '<' only (no ambiguous '>' overblown IC50 entries)

Output: data/reference/dyrk1a_actives_chembl.csv
Sidecar: data/reference/dyrk1a_actives_chembl.csv.provenance.json

NOTE ON REPRODUCIBILITY
  This script writes to data/reference/. Earlier revisions wrote to
  data/reference_sets/, a directory no consumer in this repo ever read, so
  the committed script could not have produced the committed CSV as-is.
  Fixing the path makes this script correct going forward. It does NOT make
  it the script that produced the frozen file.

  The frozen CSV is an input to the decoy cohort and the matching scales.
  This script refuses to overwrite an existing output unless --force is
  passed. Do not pass --force against data/reference/.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings

# ── CONFIG ──────────────────────────────────────────────────────────────
TARGET_CHEMBL_ID = "CHEMBL2292"          # DYRK1A, Homo sapiens
IC50_CUTOFF_NM   = 1000                  # ≤ 1 µM
ASSAY_TYPE       = "B"                   # Biochemical (enzymatic)
RELATION_WHITELIST = ["=", "'='", "<"]   # literal set, matched verbatim
OUTPUT_DIR       = "data/reference"
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "dyrk1a_actives_chembl.csv")
MANUAL_ADDITIONS = os.path.join(OUTPUT_DIR, "manual_additions.csv")
PROVENANCE_FILE  = OUTPUT_FILE + ".provenance.json"
# ────────────────────────────────────────────────────────────────────────


def sha256_of(path):
    """Return the hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def client_version():
    """Return the installed chembl_webresource_client version."""
    import importlib.metadata as md
    try:
        return md.version("chembl_webresource_client")
    except Exception:
        return None


def fetch_chembl_status():
    """Read the ChEMBL status endpoint and return its payload verbatim.

    Field names here were verified against the live endpoint rather than
    assumed: status.json reports chembl_db_version and chembl_release_date
    and nothing resembling an API version. The payload is stored whole so a
    later reader can see exactly what the service said, typos included
    (ChEMBL itself spells one key 'disinct_compounds').
    """
    url = Settings.Instance().NEW_CLIENT_URL + "/status.json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return url, payload


def load_manual_additions(path):
    """Return manual-addition rows as a DataFrame, or None if absent."""
    if not os.path.exists(path):
        return None
    manual = pd.read_csv(path, dtype={"molecule_chembl_id": str})
    required = {"molecule_chembl_id", "canonical_smiles",
                "median_ic50_nM", "n_measurements"}
    missing = required - set(manual.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    return manual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it already exists. The reference "
             "actives set is a frozen input; do not use this against "
             "data/reference/.",
    )
    args = parser.parse_args()

    if os.path.exists(OUTPUT_FILE) and not args.force:
        raise SystemExit(
            f"REFUSING TO WRITE: {OUTPUT_FILE} already exists.\n"
            f"This file is a frozen input; the decoy cohort and the matching "
            f"scales are derived from it.\n"
            f"Pass --force only if you intend to replace it."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PROVENANCE - capture the service state before querying it
    fetch_started_utc = datetime.now(timezone.utc).isoformat()
    status_url, status = fetch_chembl_status()
    print("ChEMBL status endpoint:", status_url)
    print(f"  chembl_db_version:   {status.get('chembl_db_version')}")
    print(f"  chembl_release_date: {status.get('chembl_release_date')}")
    print(f"  service status:      {status.get('status')}")

    query_constraints = {
        "target_chembl_id": TARGET_CHEMBL_ID,
        "standard_type": "IC50",
        "standard_units": "nM",
        "assay_type": ASSAY_TYPE,
        "standard_relation_in": RELATION_WHITELIST,
        "standard_value_max_nM": IC50_CUTOFF_NM,
        "require_non_empty_smiles": True,
        "dedup": "median IC50 per molecule_chembl_id",
    }

    # FETCH ALL ACTIVES FOR TARGET
    print(f"Querying ChEMBL for target {TARGET_CHEMBL_ID}...")
    activity = new_client.activity
    results = activity.filter(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type="IC50",
        standard_units="nM",
        assay_type=ASSAY_TYPE,
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_relation",
        "standard_type",
        "standard_units",
        "assay_type",
        "pchembl_value",
    ])

    # Convert to DataFrame
    df = pd.DataFrame(list(results))
    counts = {"raw_records": len(df)}
    print(f"  Raw records from ChEMBL: {len(df)}")

    if df.empty:
        print("ERROR: No records returned. Check your internet connection or target ID.")
        return

    # FILTER
    # FILTER 1 - Keep only exact measurements (= or <), not ambiguous '>' or '>>'
    df = df[df["standard_relation"].isin(RELATION_WHITELIST)].copy()
    counts["after_relation_filter"] = len(df)
    print(f"  After relation filter (= or <): {len(df)}")

    # Convert to numeric and apply cutoff
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    counts["after_numeric_value"] = len(df)
    df = df[df["standard_value"] <= IC50_CUTOFF_NM]
    counts["after_ic50_cutoff"] = len(df)
    print(f"  After IC50 ≤ {IC50_CUTOFF_NM} nM: {len(df)}")

    # Drop entries with no SMILES
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    counts["after_smiles_filter"] = len(df)
    print(f"  After dropping missing SMILES: {len(df)}")

    # Handling duplicatess (median IC50)
    grouped = df.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        median_ic50_nM=("standard_value", "median"),
        n_measurements=("standard_value", "count"),
    ).reset_index()

    counts["unique_compounds_from_chembl"] = len(grouped)
    print(f"  Unique compounds after dedup: {len(grouped)}")

    # MANUAL ADDITIONS - compounds not reachable by the query above.
    # Appended after the ChEMBL filters and before the final sort, so they
    # are never silently mistaken for query output.
    manual = load_manual_additions(MANUAL_ADDITIONS)
    manual_ids = []
    if manual is None:
        print(f"  No manual additions file at {MANUAL_ADDITIONS}; skipping.")
        counts["manual_additions_appended"] = 0
    else:
        for _, row in manual.iterrows():
            manual_ids.append(str(row["molecule_chembl_id"]))
            print(f"  Appending manual addition: {row['molecule_chembl_id']}")
        grouped = pd.concat(
            [grouped, manual[["molecule_chembl_id", "canonical_smiles",
                              "median_ic50_nM", "n_measurements"]]],
            ignore_index=True,
        )
        counts["manual_additions_appended"] = len(manual)

    counts["final_rows"] = len(grouped)

    # SAVE
    grouped = grouped.sort_values("median_ic50_nM")
    grouped.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # PROVENANCE SIDECAR
    provenance = {
        "schema_version": 1,
        "stage": "reference_actives_fetch",
        "chembl_db_version": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "api_version": None,
        "api_version_note": (
            "The ChEMBL data API exposes no version identifier. "
            "status.json reports database version and release date only, and "
            "the response carries no API-version header. Recorded as null "
            "rather than guessed. Endpoint and raw payload are stored below."
        ),
        "status_endpoint": status_url,
        "status_payload": status,
        "fetch_started_utc": fetch_started_utc,
        "fetch_completed_utc": datetime.now(timezone.utc).isoformat(),
        "client_package": "chembl_webresource_client",
        "client_package_version": client_version(),
        "query_constraints": query_constraints,
        "row_counts": counts,
        "manual_additions_file": MANUAL_ADDITIONS if manual is not None else None,
        "manual_addition_ids": manual_ids,
        "output_file": OUTPUT_FILE,
        "output_sha256": sha256_of(OUTPUT_FILE),
    }
    with open(PROVENANCE_FILE, "w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(f"Wrote provenance to {PROVENANCE_FILE}")

    # STATS
    print(f"\n--- Summary ---")
    print(f"  Compounds:     {len(grouped)}")
    print(f"  IC50 range:    {grouped['median_ic50_nM'].min():.1f} – {grouped['median_ic50_nM'].max():.1f} nM")
    print(f"  Median IC50:   {grouped['median_ic50_nM'].median():.1f} nM")
    print(f"  Compounds with ≥3 measurements: {(grouped['n_measurements'] >= 3).sum()}")


if __name__ == "__main__":
    main()
