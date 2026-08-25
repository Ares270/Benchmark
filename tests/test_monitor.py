from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.monitor import render_snapshot, snapshot_run


class MonitorTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_completed_pipeline_snapshot_uses_durable_stage_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            screening = root / "screening_5"
            self._write(screening / "subsample.json", {"selected_rows": 5})
            self._write(
                screening / "intake/summary.json",
                {"counts": {"accepted_for_preparation": 4}},
            )
            self._write(screening / "gate/gate_summary.json", {"n_passed": 3})
            self._write(
                screening / "prepared/_prep_summary.json",
                {"counts": {"successful_pdbqt_available": 3}},
            )
            self._write(
                screening / "docking/_dock_summary.json",
                {"counts": {"ok": 2, "cached": 1}},
            )
            self._write(
                screening / "pipeline_summary.json",
                {"funnel": {"submitted": 5, "successfully_scored": 3}},
            )

            snapshot = snapshot_run(root)
            self.assertTrue(snapshot["complete"])
            self.assertEqual(snapshot["arms"][0]["stages"][1]["current"], 4)
            self.assertEqual(snapshot["arms"][0]["stages"][4]["current"], 3)
            rendered = render_snapshot(snapshot)
            self.assertIn("COMPLETE", rendered)
            self.assertIn("3 scored", rendered)
            self.assertIn("does not stop the benchmark run", rendered)

    def test_four_arm_monitor_spins_only_on_first_incomplete_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "four"
            (root / "naive_property_matched").mkdir(parents=True)
            (root / "gen1_guacamol").mkdir()
            snapshot = snapshot_run(root)
            rendered = render_snapshot(snapshot, frame=1)
            arm_lines = [line for line in rendered.splitlines() if "Baseline" in line or "Gen1" in line]
            self.assertTrue(arm_lines[0].lstrip().startswith("/"))
            self.assertTrue(arm_lines[1].lstrip().startswith("·"))


if __name__ == "__main__":
    unittest.main()
