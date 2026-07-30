"""Pre-dock instrument-range gate. Constants derived 2026-07-30 from the 1,219
DYRK1A actives in data/reference/dyrk1a_actives_chembl.csv. See
docs/PREDOCK_GATE_AND_NAIVE_BASELINE_SPEC.md for the derivation.

PRE-REGISTERED: this file is committed and tagged before any arm generates a
molecule. Do not edit after the tag without a documented, dated amendment.
"""

FILTER_SCHEMA_VERSION = 1

# Elements with AutoDock/Vina atom-type parameters. Anything else either fails
# Meeko preparation or receives an unparameterized (meaningless) score.
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 62
MAX_ROTATABLE_BONDS = 15
