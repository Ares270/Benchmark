from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.dataset import AnalysisInputError
from src.analysis.run_candidates import (
    build_candidate_profile,
    run_candidate_analysis,
)
from src.harness.intake import run_intake


class CandidateAnalysisTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        source = root / "candidates.smi"
        source.write_text(
            "CCO c1\nc1ccccc1 c2\nCCN c3\n",
            encoding="utf-8",
        )
        intake_dir = root / "intake"
        run_intake(source, intake_dir)
        scores = root / "scores.csv"
        scores.write_text(
            "molecule_id,score_kcal_mol,status,reason\n"
            "c1,-7.1000,ok,\n"
            "c2,-6.5000,cached,\n",
            encoding="utf-8",
        )
        return scores, intake_dir / "molecules.csv"

    def test_candidate_profile_is_unlabeled_and_keeps_missing_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            scores, intake = self._inputs(Path(temporary))
            joined, profile = build_candidate_profile(
                scores,
                intake,
                name="generation_1",
                role="model",
            )

            self.assertEqual(len(joined), 3)
            self.assertEqual(profile["stage"], "candidate_analysis")
            self.assertFalse(
                profile["interpretation"]["auc_bedroc_enrichment_computed"]
            )
            self.assertNotIn("auc", profile)
            self.assertEqual(profile["docking"]["n_with_observed_score"], 2)
            self.assertAlmostEqual(
                profile["docking"]["coverage_over_accepted_parents"],
                2 / 3,
            )
            self.assertEqual(
                set(profile["chemistry"]["properties"]),
                {
                    "molecular_weight",
                    "clogp",
                    "tpsa_a2",
                    "hbond_donors",
                    "hbond_acceptors",
                    "rotatable_bonds",
                    "ring_count",
                    "aromatic_ring_count",
                    "fraction_csp3",
                    "formal_charge",
                    "qed",
                    "sa_score",
                },
            )

    def test_score_table_without_optional_status_columns_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, intake = self._inputs(root)
            scores.write_text(
                "molecule_id,score_kcal_mol\n"
                "c1,-7.1000\n"
                "c2,-6.5000\n",
                encoding="utf-8",
            )

            joined, _ = build_candidate_profile(
                scores,
                intake,
                name="generation_1",
                role="model",
            )
            statuses = joined.set_index("molecule_id")["status"].to_dict()
            self.assertEqual(statuses, {"c1": "", "c2": "", "c3": "absent_from_scores"})

    def test_candidate_report_writes_metrics_ranking_and_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, intake = self._inputs(root)
            report = run_candidate_analysis(
                scores,
                intake,
                "generation_1",
                root / "results",
                role="model",
            )

            self.assertTrue(report.is_file())
            rendered = report.read_text()
            self.assertIn("AUC, BEDROC, and enrichment are deliberately not computed", rendered)
            self.assertIn("Candidate structure gallery", rendered)
            self.assertIn("Top 2", rendered)
            self.assertIn("Bottom 2", rendered)
            self.assertIn("<svg", rendered)
            metrics = json.loads(
                (report.parent / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["role"], "model")
            ranking = pd.read_csv(report.parent / "ranked_candidates.csv")
            self.assertEqual(ranking["molecule_id"].tolist(), ["c1", "c2"])
            run_log = json.loads(
                (report.parent / "run_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_log["outputs"]["report"]["path"], str(report.resolve()))


    def test_score_id_outside_intake_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, intake = self._inputs(root)
            scores.write_text(
                "molecule_id,score_kcal_mol,status,reason\n"
                "not_accepted,-9.0,ok,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AnalysisInputError, "absent from accepted"):
                build_candidate_profile(
                    scores,
                    intake,
                    name="bad",
                    role="pilot",
                )


if __name__ == "__main__":
    unittest.main()
