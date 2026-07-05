import pandas as pd
import sys

# --- Configuration ---
INPUT_CSV  = "data/reference/dyrk1a_actives_chembl.csv"
OUTPUT_SMI = "data/reference/dyrk1a_actives.smi"

# --- Load ---
df = pd.read_csv(INPUT_CSV)

# --- Sanity check ---
# Confirm expected columns exist
for col in ("canonical_smiles", "molecule_chembl_id"):
    if col not in df.columns:
        sys.exit(f"ERROR: column '{col}' not found. Columns are: {list(df.columns)}")

# Drop any rows where SMILES is missing
n_before = len(df)
df = df.dropna(subset=["canonical_smiles"])
n_dropped = n_before - len(df)
if n_dropped:
    print(f"WARNING: dropped {n_dropped} rows with missing SMILES")

# --- Write .smi ---
# Format: SMILES<space>ID, one per line, no header
with open(OUTPUT_SMI, "w") as f:
    for _, row in df.iterrows():
        f.write(f"{row['canonical_smiles']} {row['molecule_chembl_id']}\n")

print(f"Wrote {len(df)} compounds to {OUTPUT_SMI}")