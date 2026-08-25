from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.dashboard import discover_runs, write_dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_indexes_campaign_without_recomputing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            campaign = results / "gen1_registered"
            analysis = campaign / "candidate_analysis"
            analysis.mkdir(parents=True)
            report = analysis / "report.html"
            report.write_text("report", encoding="utf-8")
            metrics = analysis / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "stage": "candidate_analysis",
                        "name": "gen1_registered",
                        "role": "model",
                        "intake": {
                            "submitted_rows": 1000,
                            "accepted_for_preparation": 800,
                            "validity": 0.9,
                            "parent_uniqueness": 0.85,
                        },
                        "docking": {
                            "n_with_observed_score": 700,
                            "score_distribution_kcal_mol": {"median": -7.5},
                        },
                        "computational_cost": {
                            "timing": {"estimated_requested_cpu_slot_hours": 2.0}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (campaign / "campaign_summary.json").write_text(
                json.dumps(
                    {
                        "name": "gen1_registered",
                        "design": {"registered_campaign": True},
                        "funnel": {
                            "submitted": 1000,
                            "accepted_at_intake": 800,
                            "passed_predock_gate": 750,
                            "prepared_pdbqt_available": 720,
                            "successfully_scored": 700,
                        },
                        "outputs": {
                            "candidate_metrics": {"path": str(metrics.resolve())},
                            "combined_report": {"path": str(report.resolve())},
                        },
                    }
                ),
                encoding="utf-8",
            )

            records, comparisons = discover_runs(results)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["arm"], "gen1")
            self.assertEqual(comparisons, [])
            output = write_dashboard(results, results / "benchmark_home.html")
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("One study, four locked arms", rendered)
            self.assertIn("gen1_registered", rendered)
            self.assertIn("Neutral readout", rendered)
            self.assertIn("Open report", rendered)


if __name__ == "__main__":
    unittest.main()
