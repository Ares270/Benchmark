from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.run_analysis import run


class AnalysisIntegrationTests(unittest.TestCase):
    def test_fixture_generates_complete_self_contained_run(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            report_path = run(
                root / "data" / "test10_scores.csv",
                root / "data" / "fake_decoys_scores.csv",
                root / "data" / "reference" / "dyrk1a_actives_chembl.csv",
                "integration_test",
                outdir,
                bootstrap_replicates=20,
            )
            run_dir = report_path.parent
            self.assertTrue(report_path.is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue((run_dir / "run_log.json").is_file())
            self.assertEqual(len(list((run_dir / "figures").glob("*.png"))), 5)
            self.assertEqual(len(list((run_dir / "figures").glob("*.svg"))), 5)
            self.assertEqual(len(list((run_dir / "interactive").glob("*.html"))), 5)

            metrics_data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            log_data = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_data["schema_version"], 2)
            self.assertEqual(metrics_data["metrics"]["auc"], 0.9495)
            self.assertEqual(metrics_data["bootstrap"]["replicates"], 20)
            self.assertEqual(log_data["dataset_audit"]["actives"]["n_input"], 10)
            self.assertEqual(len(log_data["provenance"]["inputs"]["active_scores"]["sha256"]), 64)
            self.assertIn("Decoys are presumed negatives", report_path.read_text(encoding="utf-8"))
            self.assertTrue((outdir / "run_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
