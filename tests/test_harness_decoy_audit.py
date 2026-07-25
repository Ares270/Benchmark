from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.harness.decoy_audit import (
    DecoyAuditError,
    build_decoy_audit,
    write_decoy_audit,
)
from src.harness.intake import run_intake


class DecoyAuditTests(unittest.TestCase):
    def _intake(self, root: Path, name: str, text: str) -> Path:
        source = root / f"{name}.smi"
        output = root / f"{name}_intake"
        source.write_text(text, encoding="utf-8")
        run_intake(source, output)
        return output / "molecules.csv"

    def test_identical_property_distributions_pass_each_property_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_path = self._intake(
                root,
                "actives",
                "CCO a1\nCCC a2\nCCN a3\n",
            )
            decoy_path = self._intake(
                root,
                "decoys",
                "CCN d1\nCCO d2\nCCC d3\n",
            )

            _, audit = build_decoy_audit(
                active_path,
                decoy_path,
                compute_topology=False,
            )

            self.assertTrue(
                all(
                    record["status"] == "pass"
                    for record in audit["property_balance"].values()
                )
            )
            self.assertEqual(audit["counts"]["exact_parent_overlaps"], 3)
            self.assertEqual(audit["status"], "fail")
            self.assertTrue(
                audit["interpretation"]["no_composite_molecular_score"]
            )

    def test_large_property_mismatch_fails_before_docking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_path = self._intake(
                root,
                "actives",
                "CCCCCCCC a1\nCCCCCCCCC a2\nCCCCCCCCCC a3\n",
            )
            decoy_path = self._intake(
                root,
                "decoys",
                "O d1\nN d2\nC d3\n",
            )

            _, audit = build_decoy_audit(
                active_path,
                decoy_path,
                compute_topology=False,
            )

            self.assertEqual(
                audit["property_balance"]["molecular_weight"]["status"],
                "fail",
            )
            self.assertEqual(audit["status"], "fail")
            self.assertEqual(audit["counts"]["exact_parent_overlaps"], 0)

    def test_topology_and_visual_report_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_path = self._intake(
                root,
                "actives",
                "c1ccccc1 a1\nCCO a2\n",
            )
            decoy_path = self._intake(
                root,
                "decoys",
                "Cc1ccccc1 d1\nCCCO d2\n",
            )
            outdir = root / "audit"

            report_path = write_decoy_audit(
                active_path,
                decoy_path,
                outdir,
                topology_threshold=0.10,
            )

            self.assertTrue(report_path.is_file())
            quality_path = outdir / "quality.json"
            self.assertTrue(quality_path.is_file())
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertGreater(
                quality["topology"]["n_above_threshold"],
                0,
            )
            self.assertIn(
                "<h1>Pre-docking decoy audit</h1>",
                report_path.read_text(encoding="utf-8"),
            )
            with self.assertRaisesRegex(DecoyAuditError, "already exists"):
                write_decoy_audit(
                    active_path,
                    decoy_path,
                    outdir,
                    compute_topology=False,
                )


if __name__ == "__main__":
    unittest.main()
