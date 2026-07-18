from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.harness.intake import IntakeInputError, run_intake


class MoleculeIntakeTests(unittest.TestCase):
    def test_intake_preserves_rejections_and_writes_accepted_structures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "submission.smi"
            input_path.write_text(
                "CCO mol1\n"
                "OCC mol2\n"
                "not_a_smiles bad\n"
                "CCN mol1\n"
                "CC.O salt\n"
                "CCC bad/id\n"
                "\n",
                encoding="utf-8",
            )
            output_dir = root / "intake"

            summary = run_intake(input_path, output_dir)
            table = pd.read_csv(output_dir / "molecules.csv")

            self.assertEqual(summary["counts"]["submitted_rows"], 6)
            self.assertEqual(summary["counts"]["valid_structures"], 5)
            self.assertEqual(summary["counts"]["unique_valid_parents"], 4)
            self.assertEqual(summary["counts"]["accepted_for_preparation"], 2)
            self.assertEqual(summary["counts"]["multifragment_valid_structures"], 1)
            self.assertEqual(summary["counts"]["parent_extractions"], 1)
            self.assertAlmostEqual(summary["aggregate_metrics"]["validity"], 5 / 6)
            self.assertAlmostEqual(
                summary["aggregate_metrics"]["uniqueness_among_valid_parents"], 4 / 5
            )

            duplicate = table.loc[table["molecule_id"] == "mol2"].iloc[0]
            self.assertEqual(duplicate["status"], "rejected")
            self.assertIn("duplicate_parent_of_mol1", duplicate["reason"])
            invalid = table.loc[table["molecule_id"] == "bad"].iloc[0]
            self.assertFalse(invalid["structure_valid"])
            accepted = table.loc[table["molecule_id"] == "mol1"].iloc[0]
            self.assertGreaterEqual(accepted["qed"], 0.0)
            self.assertLessEqual(accepted["qed"], 1.0)
            self.assertGreaterEqual(accepted["sa_score"], 1.0)
            self.assertAlmostEqual(accepted["qed"], 0.40680796565539457)
            self.assertAlmostEqual(accepted["sa_score"], 1.9802570386349831)
            salt = table.loc[table["molecule_id"] == "salt"].iloc[0]
            self.assertEqual(salt["canonical_smiles"], "CC.O")
            self.assertEqual(salt["parent_smiles"], "CC")
            self.assertTrue(salt["parent_was_extracted"])
            self.assertLessEqual(accepted["sa_score"], 10.0)

            accepted_lines = (output_dir / "accepted.smi").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(accepted_lines, ["CCO mol1", "CC salt"])
            on_disk_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                on_disk_summary["evaluation_policy"]["disconnected_smaller_fragments_removed"]
            )
            self.assertEqual(len(on_disk_summary["outputs"]["accepted_smi_sha256"]), 64)

    def test_isomeric_smiles_keep_stereoisomers_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "stereo.smi"
            input_path.write_text(
                "F[C@H](Cl)Br stereo_a\nF[C@@H](Cl)Br stereo_b\n",
                encoding="utf-8",
            )
            summary = run_intake(input_path, root / "intake")
            self.assertEqual(summary["counts"]["accepted_for_preparation"], 2)
            self.assertEqual(summary["aggregate_metrics"]["uniqueness_among_valid_parents"], 1.0)

    def test_different_salts_of_the_same_parent_collapse_to_one_molecule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "salts.smi"
            input_path.write_text(
                "CC.Cl salt_a\nCC.[Na+] salt_b\n",
                encoding="utf-8",
            )
            output_dir = root / "intake"
            summary = run_intake(input_path, output_dir)
            table = pd.read_csv(output_dir / "molecules.csv")

            self.assertEqual(summary["counts"]["valid_structures"], 2)
            self.assertEqual(summary["counts"]["unique_valid_parents"], 1)
            self.assertEqual(summary["counts"]["accepted_for_preparation"], 1)
            self.assertEqual(table.loc[0, "parent_smiles"], "CC")
            self.assertEqual(table.loc[1, "reason"], "duplicate_parent_of_salt_a")

    def test_bad_column_counts_are_preserved_as_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "format.smi"
            input_path.write_text("CCO\nCCN mol2 extra\n", encoding="utf-8")
            summary = run_intake(input_path, root / "intake")

            self.assertEqual(summary["counts"]["submitted_rows"], 2)
            self.assertEqual(summary["counts"]["valid_structures"], 2)
            self.assertEqual(summary["counts"]["accepted_for_preparation"], 0)
            self.assertEqual(
                summary["counts"]["rejection_reasons"],
                {"missing_molecule_id": 1, "unexpected_columns": 1},
            )

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "submission.smi"
            input_path.write_text("CCO mol1\n", encoding="utf-8")
            output_dir = root / "already_here"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(IntakeInputError, "already exists"):
                run_intake(input_path, output_dir)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
