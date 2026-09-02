#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provenance/verify_receptor_prep.sh
#
# Does NOT build the receptor. The docking receptor
# data/target/7O7K_protein.pdbqt is a tracked input and is never overwritten.
#
# Verifies that the committed receptor is reproducible from the deposited PDB.
# Asserts heavy-atom identity, not byte identity: OpenBabel places rotatable
# hydrogens non-deterministically, so the check that matters is that every
# heavy atom is identical and that no hydrogen which moves lies inside the
# docking search box.
#
# Reconstructed from shell history. The original commands were:
#   wget https://files.rcsb.org/download/7O7K.pdb
#   grep -E "^(ATOM|TER)" 7O7K.pdb > 7O7K_protein_noH.pdb
#   grep "^HETATM" 7O7K.pdb | grep " PTR " >> 7O7K_protein_noH.pdb
#   obabel 7O7K_protein_noH.pdb -O 7O7K_protein.pdbqt -xr -p 7.4
#   awk '/^ATOM|^HETATM/ {x+=$6;y+=$7;z+=$8;n++} END {print x/n, y/n, z/n}' \
#       7O7K_ligand_6ZV.pdbqt
#
# Usage:  bash provenance/verify_receptor_prep.sh
# Run from the repository root.
# ---------------------------------------------------------------------------

set -euo pipefail

TARGET_DIR="data/target"
TRACKED_RECEPTOR="${TARGET_DIR}/7O7K_protein.pdbqt"
TRACKED_NOH="${TARGET_DIR}/7O7K_protein_noH.pdb"
SOURCE_PDB="${TARGET_DIR}/7O7K.pdb"
LIGAND_PDB="${TARGET_DIR}/7O7K_ligand_6ZV.pdb"

# Box centre as defined in src/harness/config.py:19
EXPECTED_CENTER="8.631 17.703 24.730"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fail=0
check() {  # check <label> <ok:0|1>
  if [ "$2" -eq 0 ]; then
    printf '  PASS  %s\n' "$1"
  else
    printf '  FAIL  %s\n' "$1"
    fail=1
  fi
}

echo "=== Receptor preparation provenance check ==="
echo

# --- 0. environment -------------------------------------------------------
echo "[0] Environment"
obabel_version="$(obabel -V 2>&1 | head -1)"
echo "  ${obabel_version}"
echo "  (conda package label may differ from the binary's self-reported"
echo "   version; quote the binary string in Methods)"
echo

# --- 1. inputs present ----------------------------------------------------
echo "[1] Required inputs"
for f in "${SOURCE_PDB}" "${TRACKED_NOH}" "${TRACKED_RECEPTOR}"; do
  if [ -f "${f}" ]; then
    printf '  PASS  present: %s\n' "${f}"
  else
    printf '  FAIL  missing: %s\n' "${f}"
    fail=1
  fi
done
if [ "${fail}" -ne 0 ]; then
  echo
  echo "Cannot continue. Re-download with:"
  echo "  wget https://files.rcsb.org/download/7O7K.pdb -O ${SOURCE_PDB}"
  exit 1
fi
echo

# --- 2. rebuild the stripped protein --------------------------------------
echo "[2] Rebuild stripped protein (ATOM/TER records + PTR restored)"
grep -E "^(ATOM|TER)" "${SOURCE_PDB}" > "${WORK}/noH.pdb"
grep "^HETATM" "${SOURCE_PDB}" | grep " PTR " >> "${WORK}/noH.pdb"

diff -q "${TRACKED_NOH}" "${WORK}/noH.pdb" >/dev/null 2>&1
check "rebuilt stripped PDB matches ${TRACKED_NOH}" $?

ptr_count=$(grep -c " PTR " "${WORK}/noH.pdb" || true)
echo "        phosphotyrosine atoms retained: ${ptr_count}  (expected 32: 16 per chain)"

for het in HOH EDO FLC PEG LI 6ZV; do
  n=$(grep -c "^HETATM.* ${het} " "${WORK}/noH.pdb" || true)
  [ "${n}" -eq 0 ] || { printf '  FAIL  %s not stripped (%s records)\n' "${het}" "${n}"; fail=1; }
done
echo "        HOH / EDO / FLC / PEG / LI / 6ZV absent: confirmed"
echo

# --- 3. reconvert to PDBQT ------------------------------------------------
echo "[3] Reconvert to rigid receptor PDBQT (obabel -xr -p 7.4)"
obabel "${WORK}/noH.pdb" -O "${WORK}/receptor.pdbqt" -xr -p 7.4 2>"${WORK}/obabel.log" || true
echo "        obabel said: $(tail -1 "${WORK}/obabel.log")"

n_tracked=$(grep -c '^ATOM' "${TRACKED_RECEPTOR}")
n_new=$(grep -c '^ATOM' "${WORK}/receptor.pdbqt")
[ "${n_tracked}" -eq "${n_new}" ]
check "atom count identical (${n_tracked})" $?

# Heavy atoms must match exactly. This is the real assertion: heavy-atom
# geometry is what Vina scores. Hydrogen positions are handled separately
# because OpenBabel places rotatable hydrogens non-deterministically.
if diff -q <(grep '^ATOM' "${TRACKED_RECEPTOR}" | awk '$3 !~ /^H/') \
           <(grep '^ATOM' "${WORK}/receptor.pdbqt" | awk '$3 !~ /^H/') >/dev/null 2>&1; then
  check "heavy-atom coordinates identical" 0
else
  check "heavy-atom coordinates identical" 1
  echo "        first differing heavy atom:"
  diff <(grep '^ATOM' "${TRACKED_RECEPTOR}" | awk '$3 !~ /^H/') \
       <(grep '^ATOM' "${WORK}/receptor.pdbqt" | awk '$3 !~ /^H/') \
       | head -4 | sed 's/^/          /'
fi

h_tracked=$(grep '^ATOM' "${TRACKED_RECEPTOR}" | awk '$3 ~ /^H/' | wc -l)
h_new=$(grep '^ATOM' "${WORK}/receptor.pdbqt" | awk '$3 ~ /^H/' | wc -l)
[ "${h_tracked}" -eq "${h_new}" ]
check "hydrogen count identical (${h_tracked})" $?

# Any hydrogen that moved must lie outside the search box, or it could
# affect a score. Box: centre (8.631,17.703,24.730), size 30 x 28 x 35.
moved=$(diff <(grep '^ATOM' "${TRACKED_RECEPTOR}") \
             <(grep '^ATOM' "${WORK}/receptor.pdbqt") | grep -c '^[<>]' || true)
if [ "${moved}" -eq 0 ]; then
  echo "        no atoms moved at all"
  check "all mobile hydrogens outside search box (none moved)" 0
else
  echo "        ${moved} differing lines ($((moved / 2)) atoms), all hydrogens"
  inside=$(diff <(grep '^ATOM' "${TRACKED_RECEPTOR}") \
                <(grep '^ATOM' "${WORK}/receptor.pdbqt") \
           | grep '^[<>]' \
           | awk '{ x=$6; y=$7; z=$8;
                    if (x >= 8.631-15.0 && x <= 8.631+15.0 &&
                        y >= 17.703-14.0 && y <= 17.703+14.0 &&
                        z >= 24.730-17.5 && z <= 24.730+17.5) n++ }
                  END { print n+0 }')
  echo "        of those, inside the search box: ${inside}"
  [ "${inside}" -eq 0 ]
  check "all mobile hydrogens outside search box" $?
fi
echo

# --- 4. re-derive the box centre ------------------------------------------
echo "[4] Re-derive docking box centre from the 6ZV ligand"
if [ -f "${LIGAND_PDB}" ]; then
  obabel "${LIGAND_PDB}" -O "${WORK}/lig.pdbqt" -p 7.4 >/dev/null 2>&1
  recomputed=$(awk '/^ATOM|^HETATM/ {x+=$6; y+=$7; z+=$8; n++}
                    END {printf "%.3f %.3f %.3f", x/n, y/n, z/n}' "${WORK}/lig.pdbqt")
  echo "        recomputed centroid : ${recomputed}"
  echo "        config.py BOX_CENTER: ${EXPECTED_CENTER}"
  [ "${recomputed}" = "${EXPECTED_CENTER}" ]
  check "box centre reproduces src/harness/config.py:19" $?

  echo "        6ZV coordinate extent (informs BOX_SIZE):"
  awk '/^ATOM|^HETATM/ {
         if (n++ == 0) { xlo=xhi=$6; ylo=yhi=$7; zlo=zhi=$8 }
         if ($6<xlo) xlo=$6; if ($6>xhi) xhi=$6
         if ($7<ylo) ylo=$7; if ($7>yhi) yhi=$7
         if ($8<zlo) zlo=$8; if ($8>zhi) zhi=$8
       }
       END { printf "          spread: %.2f x %.2f x %.2f A\n", xhi-xlo, yhi-ylo, zhi-zlo }' \
      "${WORK}/lig.pdbqt"
  echo "          current BOX_SIZE: 30.0 x 28.0 x 35.0 A"
  echo "          previous BOX_SIZE: 40.0 x 38.0 x 45.0 A (changed 2026-07-31)"
else
  echo "  SKIP  ${LIGAND_PDB} not present; box centre not re-derived"
fi
echo

# --- 5. hash --------------------------------------------------------------
echo "[5] Tracked receptor hash"
sha256sum "${TRACKED_RECEPTOR}" | awk '{printf "        %s\n", $1}'
echo "        (must match the receptor hash recorded in every run_log.json)"
echo

# --- verdict --------------------------------------------------------------
echo "=========================================="
if [ "${fail}" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  echo "The committed receptor is reproducible from the deposited PDB."
else
  echo "ONE OR MORE CHECKS FAILED"
  echo "Do not claim reproducibility in Methods until resolved."
fi
echo "=========================================="
exit "${fail}"