from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.harness.dock import dock_batch
from src.harness.prepare_ligands import prepare_batch


class HarnessRuntimeTests(unittest.TestCase):
    def test_empty_preparation_still_records_cost_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_smi = root / "empty.smi"
            input_smi.write_text("", encoding="utf-8")
            out_dir = root / "prepared"

            counts = prepare_batch(input_smi, out_dir, workers=2)
            record = json.loads(
                (out_dir / "_prep_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(counts["total"], 0)
            self.assertEqual(record["stage"], "ligand_preparation")
            self.assertEqual(record["counts"]["attempted_this_invocation"], 0)
            self.assertEqual(
                len(record["input"]["smiles"]["sha256"]),
                64,
            )
            self.assertIn(
                "not measured",
                record["timing"]["cpu_accounting_note"],
            )

    def test_cached_docking_records_scores_hash_and_requested_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ligand_dir = root / "ligands"
            out_dir = root / "docking"
            ligand_dir.mkdir()
            out_dir.mkdir()
            (ligand_dir / "mol1.pdbqt").write_text(
                "REMARK ligand fixture\n",
                encoding="utf-8",
            )
            (out_dir / "mol1_out.pdbqt").write_text(
                "REMARK minimizedAffinity -7.2500\n",
                encoding="utf-8",
            )

            scores_path = out_dir / "scores.csv"
            counts = dock_batch(
                ligand_dir,
                out_dir,
                scores_path,
                workers=3,
            )
            record = json.loads(
                (out_dir / "_dock_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(counts["cached"], 1)
            self.assertEqual(record["stage"], "smina_docking")
            self.assertEqual(record["counts"]["ok"], 0)
            self.assertEqual(record["counts"]["cached"], 1)
            self.assertEqual(
                record["timing"]["maximum_concurrent_cpu_slots_requested"],
                3,
            )
            self.assertEqual(
                len(record["outputs"]["scores_csv"]["sha256"]),
                64,
            )
            self.assertIn("mol1,-7.2500,cached", scores_path.read_text())


if __name__ == "__main__":
    unittest.main()
