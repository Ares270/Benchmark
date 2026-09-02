#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provenance/verify_actives.sh
#
# THIS SCRIPT IS THE PROJECT'S ONLY REPRODUCTION CLAIM FOR THE ACTIVES SET.
#
# src/utils/fetch_dyrk1a_actives.py as committed CANNOT byte-reproduce
# data/reference/dyrk1a_actives_chembl.csv, for two independent reasons:
#
#   1. Output path defect. The original script wrote to data/reference_sets/,
#      a directory no consumer in this repo ever read. The committed script
#      therefore cannot have produced the committed file. The path is now
#      fixed, which makes the script correct going forward; it does not make
#      it the script that produced the frozen file.
#
#   2. Row order. The manual addition (leucettine L41, MANUAL_L41) is appended
#      after the ChEMBL filters and before the final sort, so a fresh run
#      places it by IC50. In the frozen file it is the trailing row, an
#      artefact of the original hand-edit. Content is unaffected; byte
#      identity is not recoverable.
#
# What IS verified here is content-level consistency between the frozen set
# and a live ChEMBL query under identical filters. That is the claim the
# paper makes, and it is the only one it makes.
#
# Deliberately a SUBSET check, not an equality check. Today's ChEMBL contains
# compounds curated after the frozen fetch (frozen fetch bounded above at
# 2026-07-05; CHEMBL_37, released 2026-05-01, was and remains current). New
# compounds are REPORTED, not asserted. The script fails loudly only if a
# committed compound is missing, mis-filtered, or structurally different.
#
# Writes only to a temp directory. NEVER writes to data/reference/; the
# frozen files are hashed before and after and asserted unchanged.
#
# Usage:  bash provenance/verify_actives.sh
# Run from the repository root. Requires network access to ChEMBL.
# ---------------------------------------------------------------------------

set -euo pipefail

REFERENCE_DIR="data/reference"
COMMITTED_CSV="${REFERENCE_DIR}/dyrk1a_actives_chembl.csv"
MANUAL_CSV="${REFERENCE_DIR}/manual_additions.csv"
FROZEN_SHA="provenance/actives_frozen.sha256"

for f in "${COMMITTED_CSV}" "${MANUAL_CSV}" "${FROZEN_SHA}"; do
  if [ ! -f "${f}" ]; then
    echo "MISSING REQUIRED INPUT: ${f}" >&2
    exit 2
  fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

case "${WORK}" in
  *"${REFERENCE_DIR}"*)
    echo "REFUSING TO RUN: temp dir ${WORK} is inside ${REFERENCE_DIR}" >&2
    exit 2
    ;;
esac

echo "=========================================="
echo "verify_actives.sh"
echo "  committed CSV : ${COMMITTED_CSV}"
echo "  manual adds   : ${MANUAL_CSV}"
echo "  frozen hash   : ${FROZEN_SHA}"
echo "  work dir      : ${WORK}"
echo "=========================================="
echo

python3 - "${COMMITTED_CSV}" "${MANUAL_CSV}" "${FROZEN_SHA}" "${WORK}" <<'PY'
import csv
import hashlib
import json
import sys
import warnings

warnings.filterwarnings("ignore")

committed_path, manual_path, frozen_sha_path, work = sys.argv[1:5]

import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings

# Filter constants, copied verbatim from src/utils/fetch_dyrk1a_actives.py
TARGET_CHEMBL_ID = "CHEMBL2292"
IC50_CUTOFF_NM = 1000
ASSAY_TYPE = "B"
RELATION_WHITELIST = ["=", "'='", "<"]

fail = 0


def check(label, ok):
    global fail
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        fail = 1


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(smiles):
    """RDKit canonical SMILES, or None if unparseable."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# --- guard: hash the frozen inputs before we do anything ------------------
frozen_before = {}
for p in (committed_path, f"{committed_path.rsplit('/', 1)[0]}/dyrk1a_actives.smi"):
    try:
        frozen_before[p] = sha256_of(p)
    except FileNotFoundError:
        pass

committed = pd.read_csv(committed_path, dtype={"molecule_chembl_id": str})
committed_text = pd.read_csv(committed_path, dtype=str).set_index(
    "molecule_chembl_id")
manual = pd.read_csv(manual_path, dtype={"molecule_chembl_id": str})
manual_ids = set(manual["molecule_chembl_id"])
expected_ids = set(committed["molecule_chembl_id"]) - manual_ids

print(f"Committed compounds:      {len(committed)}")
print(f"Manual additions:         {len(manual)}  ({', '.join(sorted(manual_ids))})")
print(f"ChEMBL-derived expected:  {len(expected_ids)}")
print()

# --- live query -----------------------------------------------------------
status_url = Settings.Instance().NEW_CLIENT_URL + "/status.json"
status = requests.get(status_url, timeout=60).json()
chembl_version_today = status.get("chembl_db_version")
chembl_release_today = status.get("chembl_release_date")

print(f"Querying ChEMBL live ({chembl_version_today}, released "
      f"{chembl_release_today})...")

results = new_client.activity.filter(
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

df = pd.DataFrame(list(results))
print(f"  Raw records from ChEMBL: {len(df)}")
ids_raw = set(df["molecule_chembl_id"])

df = df[df["standard_relation"].isin(RELATION_WHITELIST)].copy()
print(f"  After relation filter:   {len(df)}")
ids_relation = set(df["molecule_chembl_id"])

df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
df = df.dropna(subset=["standard_value"])
df = df[df["standard_value"] <= IC50_CUTOFF_NM]
print(f"  After IC50 <= {IC50_CUTOFF_NM} nM:   {len(df)}")
ids_cutoff = set(df["molecule_chembl_id"])

df = df.dropna(subset=["canonical_smiles"])
df = df[df["canonical_smiles"].str.strip() != ""]
print(f"  After SMILES filter:     {len(df)}")

today = df.groupby("molecule_chembl_id").agg(
    canonical_smiles=("canonical_smiles", "first"),
    median_ic50_nM=("standard_value", "median"),
    n_measurements=("standard_value", "count"),
).reset_index()
print(f"  Unique compounds today:  {len(today)}")
print()

today_ids = set(today["molecule_chembl_id"])
today_by_id = today.set_index("molecule_chembl_id")
committed_by_id = committed.set_index("molecule_chembl_id")

with open(f"{work}/today_filtered.csv", "w", newline="") as fh:
    today.to_csv(fh, index=False)

# --- 1. every committed ChEMBL compound survives today's filters ----------
print("[1] Committed ChEMBL compounds present and passing all filters")
missing = sorted(expected_ids - today_ids)
if missing:
    print(f"        {len(missing)} committed compound(s) absent from today's "
          f"filtered result:")
    for mid in missing[:40]:
        if mid not in ids_raw:
            why = "not returned by the query at all"
        elif mid not in ids_relation:
            why = "dropped at relation filter"
        elif mid not in ids_cutoff:
            why = "dropped at IC50 cutoff"
        else:
            why = "dropped at SMILES filter"
        print(f"          {mid}: {why}")
    if len(missing) > 40:
        print(f"          ... and {len(missing) - 40} more")
check(f"all {len(expected_ids)} committed ChEMBL compounds present and "
      f"passing every filter", not missing)
print()

# --- 2. RDKit-canonical SMILES match --------------------------------------
print("[2] Canonical SMILES match after RDKit canonicalization")
shared = sorted(expected_ids & today_ids)
smiles_mismatch = []
unparseable = []
for mid in shared:
    c_raw = committed_by_id.loc[mid, "canonical_smiles"]
    t_raw = today_by_id.loc[mid, "canonical_smiles"]
    c, t = canon(c_raw), canon(t_raw)
    if c is None or t is None:
        unparseable.append((mid, c_raw, t_raw))
    elif c != t:
        smiles_mismatch.append((mid, c, t))
if unparseable:
    print(f"        {len(unparseable)} compound(s) with unparseable SMILES:")
    for mid, c_raw, t_raw in unparseable[:20]:
        print(f"          {mid}: committed={c_raw!r} today={t_raw!r}")
if smiles_mismatch:
    print(f"        {len(smiles_mismatch)} structural mismatch(es):")
    for mid, c, t in smiles_mismatch[:20]:
        print(f"          {mid}:")
        print(f"            committed: {c}")
        print(f"            today:     {t}")
    if len(smiles_mismatch) > 20:
        print(f"          ... and {len(smiles_mismatch) - 20} more")
check(f"all {len(shared)} shared compounds structurally identical",
      not smiles_mismatch and not unparseable)
print()

# --- 3. manual additions present in the committed file --------------------
print("[3] Manual additions present in the committed file")
manual_missing = []
manual_differs = []
for _, row in manual.iterrows():
    mid = row["molecule_chembl_id"]
    if mid not in committed_by_id.index:
        manual_missing.append(mid)
        continue
    c_row = committed_by_id.loc[mid]
    if canon(c_row["canonical_smiles"]) != canon(row["canonical_smiles"]):
        manual_differs.append((mid, "canonical_smiles"))
    elif float(c_row["median_ic50_nM"]) != float(row["median_ic50_nM"]):
        manual_differs.append((mid, "median_ic50_nM"))
for mid in manual_missing:
    print(f"        {mid}: absent from committed file")
for mid, field in manual_differs:
    print(f"        {mid}: {field} differs from committed file")
check(f"all {len(manual)} manual addition(s) present and matching",
      not manual_missing and not manual_differs)
print()

# --- 4. frozen hash -------------------------------------------------------
print("[4] Committed file hash matches the frozen record")
recorded = open(frozen_sha_path).read().split()[0].strip()
actual = sha256_of(committed_path)
print(f"        recorded: {recorded}")
print(f"        actual:   {actual}")
check("committed CSV sha256 matches provenance/actives_frozen.sha256",
      recorded == actual)
print()

# --- REPORTED, NOT ASSERTED ----------------------------------------------
new_today = sorted(today_ids - set(committed["molecule_chembl_id"]))

def stored_decimals(text):
    """Decimal places the CSV actually stores for a value."""
    text = str(text).strip()
    return len(text.split(".", 1)[1]) if "." in text else 0


# Compare at the precision the committed file stores, not at full float
# precision. The committed median is text round-tripped through CSV at one
# decimal place; today's is recomputed in full precision, so exact float
# equality reports differences of ~1e-14 that are representation noise, not
# curation changes. Rounding today's value to the stored precision reports
# only differences the committed file could actually have recorded.
ic50_deltas = []
for mid in shared:
    c_text = committed_text.loc[mid, "median_ic50_nM"]
    places = stored_decimals(c_text)
    c_val = float(c_text)
    t_val = round(float(today_by_id.loc[mid, "median_ic50_nM"]), places)
    if c_val != t_val:
        ic50_deltas.append((mid, c_val, t_val, t_val - c_val))

print("==========================================")
print("REPORTED, NOT ASSERTED")
print("==========================================")
print(f"  {'ChEMBL version today':<44} {chembl_version_today}")
print(f"  {'ChEMBL release date today':<44} {chembl_release_today}")
print(f"  {'Compounds in today result absent from committed':<44} "
      f"{len(new_today)}")
print(f"  {'Committed compounds with changed median IC50*':<44} "
      f"{len(ic50_deltas)}")
print("  * compared at the precision the committed CSV stores; see"
      " stored_decimals()")
print()

if ic50_deltas:
    print("  Median IC50 deltas (committed -> today, nM):")
    print(f"    {'compound':<16} {'committed':>12} {'today':>12} {'delta':>12}")
    for mid, c_val, t_val, d in sorted(
            ic50_deltas, key=lambda r: -abs(r[3]))[:25]:
        print(f"    {mid:<16} {c_val:>12.1f} {t_val:>12.1f} {d:>+12.1f}")
    if len(ic50_deltas) > 25:
        print(f"    ... and {len(ic50_deltas) - 25} more "
              f"(full list in {work}/ic50_deltas.csv)")
    with open(f"{work}/ic50_deltas.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_chembl_id", "committed_nM", "today_nM", "delta_nM"])
        w.writerows(ic50_deltas)
    print()

# --- guard: frozen files untouched ---------------------------------------
print("[guard] Frozen inputs unchanged by this run")
for p, before in frozen_before.items():
    after = sha256_of(p)
    check(f"{p} unchanged", before == after)
print()

print("==========================================")
if fail == 0:
    print("ALL CHECKS PASSED")
    print("The committed actives set is content-consistent with a live")
    print("ChEMBL query under identical filters.")
else:
    print("ONE OR MORE CHECKS FAILED")
    print("Do not claim reproducibility in Methods until resolved.")
print("==========================================")
sys.exit(fail)
PY
