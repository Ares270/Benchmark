import pandas as pd
from rdkit import Chem
from pathlib import Path


# open the cv , check columns, go through library smiles, pas them through rdkit's 
# equation, check that they match and that we have the canonical version, plus check duplicates


repo_root = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(repo_root / "data" / "reference" / "dyrk1a_actives_chembl.csv")
print(df.columns.tolist())

failed = []
canonical = []

for i, smi in enumerate(df["canonical_smiles"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed.append(i)
        canonical.append(smi)  # keep the original so you don't lose the row
    else:
        canonical.append(Chem.MolToSmiles(mol))



changed = sum(1 for old, new in zip(df["canonical_smiles"], canonical) if old != new)
print(f"SMILES that changed: {changed}")

# check for duplicates that emerged after canonicalization
dupes = pd.Series(canonical).duplicated(keep=False)
n_dupes = dupes.sum()
print(f"Duplicate molecules after canonicalization: {n_dupes}")


print(f"Total: {len(df)}")
print(f"Failed to parse: {len(failed)}")
if failed:
    print(f"Failed row indices: {failed}")

