from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.harness.intake import run_intake
from src.harness.select_decoys import (
    DEFAULT_NEIGHBORS_PER_ACTIVE,
    DEFAULT_DISSIMILAR_FRACTION,
    DecoySelectionError,
    _balanced_unique_assignment,
    select_decoys,
)


class DecoySelectionTests(unittest.TestCase):
    def _active_intake(self, root: Path) -> Path:
        active_smi = root / "actives.smi"
        active_smi.write_text(
            "CCO a1\nc1ccccc1 a2\n",
            encoding="utf-8",
        )
        intake_dir = root / "active_intake"
        run_intake(active_smi, intake_dir)
        return intake_dir / "molecules.csv"

    def _pool(self, root: Path) -> Path:
        pool = root / "pool.smi"
        pool.write_text(
            "CCCO p1\n"
            "CC(C)O p2\n"
            "COC p3\n"
            "CCOC p4\n"
            "Cc1ccccc1 p5\n"
            "Fc1ccccc1 p6\n"
            "c1ccncc1 p7\n"
            "Clc1ccccc1 p8\n",
            encoding="utf-8",
        )
        return pool

    def test_balanced_selection_writes_unique_decoys_intake_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_intake = self._active_intake(root)
            pool = self._pool(root)
            outdir = root / "selection"

            selection = select_decoys(
                active_intake,
                [pool],
                outdir,
                per_active=2,
                neighbors_per_active=8,
                dissimilar_fraction=1.0,
                max_tanimoto=1.0,
            )

            self.assertEqual(selection["counts"]["selected_unique_decoys"], 4)
            self.assertFalse(
                selection["interpretation"]["official_dude_or_dudez_output"]
            )
            assignments = pd.read_csv(outdir / "assignments.csv")
            self.assertEqual(assignments["decoy_id"].nunique(), 4)
            self.assertEqual(
                assignments.groupby("matched_active_id").size().to_dict(),
                {"a1": 2, "a2": 2},
            )
            self.assertTrue((outdir / "intake" / "molecules.csv").is_file())
            self.assertTrue((outdir / "audit" / "report.html").is_file())
            on_disk = json.loads(
                (outdir / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk["parameters"]["formal_charge_matching"], "exact")

    def test_insufficient_unique_pool_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_intake = self._active_intake(root)
            pool = root / "tiny.smi"
            pool.write_text("CCCO p1\n", encoding="utf-8")
            outdir = root / "selection"

            with self.assertRaisesRegex(
                DecoySelectionError,
                "eligible unique pool parents",
            ):
                select_decoys(
                    active_intake,
                    [pool],
                    outdir,
                    per_active=2,
                    neighbors_per_active=2,
                    dissimilar_fraction=1.0,
                    max_tanimoto=1.0,
                )
            self.assertFalse(outdir.exists())


    def test_default_and_assignment_prioritize_property_balance(self):
        self.assertEqual(DEFAULT_NEIGHBORS_PER_ACTIVE, 5000)
        self.assertEqual(DEFAULT_DISSIMILAR_FRACTION, 1.0)
        actives = pd.DataFrame({"molecule_id": ["a1", "a2"]})
        options = [
            [(0, 0.10), (1, 0.20), (2, 0.30)],
            [(0, 0.10), (3, 0.20), (4, 0.30)],
        ]

        assignments = _balanced_unique_assignment(
            actives,
            options,
            [],
            per_active=2,
        )

        self.assertEqual(len(assignments), 4)
        self.assertEqual(len({candidate for _, candidate, _ in assignments}), 4)
        self.assertEqual(
            pd.Series(active for active, _, _ in assignments).value_counts().to_dict(),
            {0: 2, 1: 2},
        )
        self.assertLessEqual(max(distance for _, _, distance in assignments), 0.30)


if __name__ == "__main__":
    unittest.main()
