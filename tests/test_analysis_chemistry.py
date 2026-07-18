from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.chemistry import PROPERTY_COLUMNS, build_chemical_profile
from src.analysis.dataset import AnalysisInputError
from src.harness.intake import run_intake


class ChemistryProfileTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, pd.DataFrame]:
        active_smi = root / "actives.smi"
        decoy_smi = root / "decoys.smi"
        active_smi.write_text(
            "CCO a1\nc1ccccc1 a2\nCCN a3\n",
            encoding="utf-8",
        )
        decoy_smi.write_text(
            "CCCC d1\nO=C=O d2\nCC(=O)O d3\n",
            encoding="utf-8",
        )
        active_dir = root / "active_intake"
        decoy_dir = root / "decoy_intake"
        run_intake(active_smi, active_dir)
        run_intake(decoy_smi, decoy_dir)
        docking = pd.DataFrame(
            {
                "molecule_id": ["a1", "a2", "a3", "d1", "d2", "d3"],
                "label": [1, 1, 1, 0, 0, 0],
                "score": [-8.0, -9.0, -7.0, -6.0, -5.5, -6.5],
                "score_imputed": [False] * 6,
            }
        )
        return (
            active_dir / "molecules.csv",
            decoy_dir / "molecules.csv",
            docking,
        )

    def test_profile_summarizes_every_property_without_composite_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            active_path, decoy_path, docking = self._inputs(Path(temporary))
            frame, profile = build_chemical_profile(
                active_path, decoy_path, docking
            )

            self.assertEqual(len(frame), 6)
            self.assertEqual(
                set(PROPERTY_COLUMNS),
                set(profile["cohorts"]["actives"]["properties"]),
            )
            active_clogp = frame.loc[
                frame["cohort"].eq("actives"), "clogp"
            ].mean()
            decoy_clogp = frame.loc[
                frame["cohort"].eq("decoys"), "clogp"
            ].mean()
            self.assertAlmostEqual(
                profile["mean_difference_actives_minus_decoys"]["clogp"],
                active_clogp - decoy_clogp,
            )
            self.assertNotIn("overall_score", profile)
            self.assertEqual(
                profile["score_property_spearman"]["actives"][
                    "n_observed_scores"
                ],
                3,
            )

    def test_docking_ids_must_match_accepted_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            active_path, decoy_path, docking = self._inputs(Path(temporary))
            docking.loc[0, "molecule_id"] = "not_in_intake"
            with self.assertRaisesRegex(
                AnalysisInputError, "absent from the matching"
            ):
                build_chemical_profile(active_path, decoy_path, docking)


    def test_accepted_parent_without_docking_row_stays_in_property_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            active_path, decoy_path, docking = self._inputs(Path(temporary))
            docking = docking.loc[docking["molecule_id"].ne("a1")].copy()
            frame, profile = build_chemical_profile(
                active_path, decoy_path, docking
            )

            self.assertEqual(
                int(frame["cohort"].eq("actives").sum()),
                3,
            )
            self.assertEqual(
                profile["cohorts"]["actives"][
                    "n_with_observed_docking_score"
                ],
                2,
            )

    def test_modified_intake_table_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            active_path, decoy_path, docking = self._inputs(Path(temporary))
            active_path.write_text(
                active_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AnalysisInputError, "SHA-256"):
                build_chemical_profile(active_path, decoy_path, docking)


if __name__ == "__main__":
    unittest.main()
