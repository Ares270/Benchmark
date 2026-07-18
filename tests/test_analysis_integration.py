from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.run_analysis import run
from src.harness.intake import run_intake
from src.harness import runtime


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
            self.assertEqual(metrics_data["schema_version"], 3)
            self.assertEqual(metrics_data["metrics"]["auc"], 0.9495)
            self.assertEqual(metrics_data["bootstrap"]["replicates"], 20)
            self.assertEqual(log_data["dataset_audit"]["actives"]["n_input"], 10)
            self.assertEqual(len(log_data["provenance"]["inputs"]["active_scores"]["sha256"]), 64)
            self.assertIn("Decoys are presumed negatives", report_path.read_text(encoding="utf-8"))
            self.assertTrue((outdir / "run_history.csv").is_file())


    def test_intake_profiles_add_property_tables_and_three_figures(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            active_intake_dir = outdir / "active_intake"
            decoy_intake_dir = outdir / "decoy_intake"
            run_intake(
                root / "data" / "reference" / "test10.smi",
                active_intake_dir,
            )

            decoy_scores = pd.read_csv(
                root / "data" / "fake_decoys_scores.csv",
                dtype={"molecule_id": "string"},
            )
            decoy_smi = outdir / "decoys.smi"
            decoy_smi.write_text(
                "".join(
                    f"[{index + 1}CH4] {molecule_id}\n"
                    for index, molecule_id in enumerate(
                        decoy_scores["molecule_id"].astype(str)
                    )
                ),
                encoding="utf-8",
            )
            run_intake(decoy_smi, decoy_intake_dir)

            active_scores_path = outdir / "active_docking" / "scores.csv"
            decoy_scores_path = outdir / "decoy_docking" / "scores.csv"
            active_scores_path.parent.mkdir()
            decoy_scores_path.parent.mkdir()
            active_scores_path.write_bytes(
                (root / "data" / "test10_scores.csv").read_bytes()
            )
            decoy_scores_path.write_bytes(
                (root / "data" / "fake_decoys_scores.csv").read_bytes()
            )
            for scores_path, total, wall_seconds in (
                (active_scores_path, 10, 90.0),
                (decoy_scores_path, 200, 1800.0),
            ):
                runtime.write_json_atomic(
                    scores_path.parent / "_dock_summary.json",
                    {
                        "schema_version": 1,
                        "stage": "smina_docking",
                        "outputs": {
                            "scores_csv": runtime.file_record(scores_path)
                        },
                        "counts": {
                            "total": total,
                            "ok": total,
                            "cached": 0,
                            "failed": 0,
                        },
                        "timing": {
                            "wall_seconds": wall_seconds,
                            "workers_requested": 4,
                            "cpu_slots_per_task": 1,
                            "fresh_successes_per_wall_second": (
                                total / wall_seconds
                            ),
                            "estimated_requested_cpu_slot_hours": (
                                wall_seconds * 4 / 3600
                            ),
                        },
                    },
                )


            report_path = run(
                active_scores_path,
                decoy_scores_path,
                root / "data" / "reference" / "dyrk1a_actives_chembl.csv",
                "chemistry_integration",
                outdir,
                active_intake_path=active_intake_dir / "molecules.csv",
                decoy_intake_path=decoy_intake_dir / "molecules.csv",
                bootstrap_replicates=10,
            )
            run_dir = report_path.parent
            self.assertEqual(
                len(list((run_dir / "figures").glob("*.png"))), 8
            )
            self.assertEqual(
                len(list((run_dir / "interactive").glob("*.html"))), 8
            )

            metrics_data = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("chemistry", metrics_data)
            self.assertEqual(
                metrics_data["chemistry"]["cohorts"]["decoys"][
                    "n_accepted_parents"
                ],
                200,
            )
            self.assertEqual(
                metrics_data["computational_cost"]["decoys"]["counts"]["ok"],
                200,
            )
            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("Chemical-property profile", report_html)
            self.assertIn("active − decoy", report_html)
            self.assertIn("No tautomers", report_html)
            self.assertIn("Recorded docking cost", report_html)


if __name__ == "__main__":
    unittest.main()
