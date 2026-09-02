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
import os

import pandas as pd
from chembl_webresource_client.new_client import new_client

# ── CONFIG ──────────────────────────────────────────────────────────────
TARGET_CHEMBL_ID = "CHEMBL2292"          # DYRK1A, Homo sapiens
IC50_CUTOFF_NM   = 1000                  # ≤ 1 µM
ASSAY_TYPE       = "B"                   # Biochemical (enzymatic)
RELATION_WHITELIST = ["=", "'='", "<"]   # literal set, matched verbatim
OUTPUT_DIR       = "data/reference"
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "dyrk1a_actives_chembl.csv")
# ────────────────────────────────────────────────────────────────────────


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
    print(f"  Raw records from ChEMBL: {len(df)}")

    if df.empty:
        print("ERROR: No records returned. Check your internet connection or target ID.")
        return

    # FILTER
    # FILTER 1 - Keep only exact measurements (= or <), not ambiguous '>' or '>>'
    df = df[df["standard_relation"].isin(RELATION_WHITELIST)].copy()
    print(f"  After relation filter (= or <): {len(df)}")

    # Convert to numeric and apply cutoff
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] <= IC50_CUTOFF_NM]
    print(f"  After IC50 ≤ {IC50_CUTOFF_NM} nM: {len(df)}")

    # Drop entries with no SMILES
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    print(f"  After dropping missing SMILES: {len(df)}")

    # Handling duplicatess (median IC50)
    grouped = df.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        median_ic50_nM=("standard_value", "median"),
        n_measurements=("standard_value", "count"),
    ).reset_index()

    print(f"  Unique compounds after dedup: {len(grouped)}")

    # SAVE
    grouped = grouped.sort_values("median_ic50_nM")
    grouped.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # STATS
    print(f"\n--- Summary ---")
    print(f"  Compounds:     {len(grouped)}")
    print(f"  IC50 range:    {grouped['median_ic50_nM'].min():.1f} – {grouped['median_ic50_nM'].max():.1f} nM")
    print(f"  Median IC50:   {grouped['median_ic50_nM'].median():.1f} nM")
    print(f"  Compounds with ≥3 measurements: {(grouped['n_measurements'] >= 3).sum()}")


if __name__ == "__main__":
    main()
