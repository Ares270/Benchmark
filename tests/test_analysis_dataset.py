from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis.dataset import AnalysisInputError, build_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def standard_files(self):
        active = self.write(
            "active.csv",
            "molecule_id,score_kcal_mol,status,reason\nA1,-9.0,ok,\nA2,-8.0,cached,\n",
        )
        decoy = self.write(
            "decoy.csv",
            "molecule_id,score_kcal_mol,status,reason\nD1,-6.0,ok,\nD2,-5.0,ok,\n",
        )
        reference = self.write(
            "reference.csv",
            "molecule_chembl_id,canonical_smiles\nA1,C\nA2,CC\nA3,CCC\n",
        )
        return active, decoy, reference

    def test_valid_data_are_audited(self):
        active, decoy, reference = self.standard_files()
        dataset = build_dataset(active, decoy, reference)
        self.assertEqual(len(dataset.frame), 4)
        self.assertEqual(dataset.audit["actives"]["coverage"], 1.0)
        self.assertEqual(dataset.audit["reference"]["n_reference_ids_not_in_scored_actives"], 1)

    def test_duplicate_id_is_fatal(self):
        active, decoy, _ = self.standard_files()
        active.write_text(
            "molecule_id,score_kcal_mol\nA1,-9\nA1,-8\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(AnalysisInputError, "duplicate"):
            build_dataset(active, decoy)

    def test_cross_class_overlap_is_fatal(self):
        active, decoy, _ = self.standard_files()
        decoy.write_text(
            "molecule_id,score_kcal_mol\nA1,-6\nD2,-5\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(AnalysisInputError, "both active and decoy"):
            build_dataset(active, decoy)

    def test_reference_mismatch_is_fatal(self):
        active, decoy, reference = self.standard_files()
        reference.write_text(
            "molecule_chembl_id,canonical_smiles\nA1,C\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(AnalysisInputError, "absent from the active reference"):
            build_dataset(active, decoy, reference)

    def test_missing_reference_id_column_is_fatal(self):
        active, decoy, reference = self.standard_files()
        reference.write_text("wrong_column\nA1\nA2\n", encoding="utf-8")
        with self.assertRaisesRegex(AnalysisInputError, "lacks reference ID column"):
            build_dataset(active, decoy, reference)

    def test_missing_policy_must_be_explicit(self):
        active, decoy, _ = self.standard_files()
        active.write_text(
            "molecule_id,score_kcal_mol,status,reason\nA1,-9,ok,\nA2,,dock_failed,error\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(AnalysisInputError, "Choose --missing-policy"):
            build_dataset(active, decoy)

        excluded = build_dataset(active, decoy, missing_policy="exclude")
        self.assertEqual(len(excluded.frame), 3)
        self.assertEqual(excluded.audit["actives"]["n_missing_score"], 1)

        ranked_last = build_dataset(active, decoy, missing_policy="rank_last")
        missing_row = ranked_last.frame.loc[ranked_last.frame["molecule_id"] == "A2"].iloc[0]
        self.assertTrue(missing_row["score_imputed"])
        self.assertGreater(missing_row["score"], ranked_last.frame.loc[~ranked_last.frame["score_imputed"], "score"].max())

        higher_ranked_last = build_dataset(
            active, decoy, missing_policy="rank_last", score_direction="higher_is_better"
        )
        higher_missing = higher_ranked_last.frame.loc[
            higher_ranked_last.frame["molecule_id"] == "A2"
        ].iloc[0]
        self.assertLess(
            higher_missing["score"],
            higher_ranked_last.frame.loc[~higher_ranked_last.frame["score_imputed"], "score"].min(),
        )

    def test_non_numeric_and_infinite_scores_are_not_missing(self):
        active, decoy, _ = self.standard_files()
        active.write_text("molecule_id,score_kcal_mol\nA1,banana\n", encoding="utf-8")
        with self.assertRaisesRegex(AnalysisInputError, "non-numeric"):
            build_dataset(active, decoy, missing_policy="exclude")
        active.write_text("molecule_id,score_kcal_mol\nA1,inf\n", encoding="utf-8")
        with self.assertRaisesRegex(AnalysisInputError, "infinite"):
            build_dataset(active, decoy, missing_policy="exclude")

    def test_status_score_contradiction_is_fatal(self):
        active, decoy, _ = self.standard_files()
        active.write_text(
            "molecule_id,score_kcal_mol,status\nA1,,ok\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(AnalysisInputError, "successful"):
            build_dataset(active, decoy, missing_policy="exclude")


if __name__ == "__main__":
    unittest.main()
