"""
Central configuration for the DYRK1A docking harness.

"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
# repo_root/src/harness/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR      = REPO_ROOT / "data"
TARGET_DIR    = DATA_DIR / "target"
REFERENCE_DIR = DATA_DIR / "reference"
RECEPTOR_PDBQT = TARGET_DIR / "7O7K_protein.pdbqt"

# ── Docking box ──────────────────────────────────────────────────────────
# From the native-ligand (6ZV / abemaciclib) binding site in 7O7K.
BOX_CENTER = (8.631, 17.703, 24.730)   # (x, y, z) Angstrom
BOX_SIZE   = (40.0, 38.0, 45.0)        # (x, y, z) Angstrom

# ── Smina search parameters ──────────────────────────────────────────────
EXHAUSTIVENESS = 8      
SEED           = 42     
NUM_MODES      = 9      
ENERGY_RANGE   = 3.0    # kcal/mol; max gap between best and worst kept pose

# ── Parallelism ──────────────────────────────────────────────────────────
WORKERS   = 4      
SMINA_CPU = 1      # cores per Smina job
DOCK_TIMEOUT_S = 600   # 10 min per ligand

# ── 3D ligand preparation ────────────────────────────────────────────────
EMBED_SEED = 0xf00d

# Keep macrocyclic rings rigid during PDBQT preparation.
# Meeko's default breaks one macrocycle bond to make the ring flexible and
# inserts glue pseudo-atoms typed CG0/CG1/G0/G1. Smina uses AutoDock/Vina atom
# typing and does not recognize those types, so any macrocyclic ligand fails to
# dock. Acceptable and applied uniformly across every generation.
RIGID_MACROCYCLES = True


def smina_box_args() -> list[str]:
    """Box geometry as Smina CLI flags."""
    cx, cy, cz = BOX_CENTER
    sx, sy, sz = BOX_SIZE
    return [
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
    ]
