from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.dataset import AnalysisInputError
from src.analysis.run_candidates import (
    LE_COLUMN,
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


    def test_ligand_efficiency_is_score_over_heavy_atoms_and_never_imputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            scores, intake = self._inputs(Path(temporary))
            joined, profile = build_candidate_profile(
                scores,
                intake,
                name="generation_1",
                role="model",
            )

            indexed = joined.set_index("molecule_id")
            # CCO and c1ccccc1 both have exactly 3 and 6 heavy atoms.
            self.assertEqual(indexed.loc["c1", "heavy_atom_count"], 3)
            self.assertEqual(indexed.loc["c2", "heavy_atom_count"], 6)
            self.assertAlmostEqual(
                indexed.loc["c1", LE_COLUMN], -7.1000 / 3
            )
            self.assertAlmostEqual(
                indexed.loc["c2", LE_COLUMN], -6.5000 / 6
            )

            # c3 was accepted at intake but never scored: no score, no LE.
            self.assertTrue(pd.isna(indexed.loc["c3", "score"]))
            self.assertTrue(pd.isna(indexed.loc["c3", LE_COLUMN]))

            le = profile["docking"]["le_distribution"]
            self.assertEqual(le["n"], 2)
            self.assertAlmostEqual(le["minimum_best"], -7.1000 / 3)
            self.assertAlmostEqual(le["maximum_worst"], -6.5000 / 6)

    def test_le_distribution_mirrors_score_distribution_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            scores, intake = self._inputs(Path(temporary))
            _, profile = build_candidate_profile(
                scores,
                intake,
                name="generation_1",
                role="model",
            )
            docking = profile["docking"]

            self.assertEqual(
                set(docking["le_distribution"]),
                set(docking["score_distribution_kcal_mol"]),
            )
            self.assertEqual(
                set(docking["le_distribution"]),
                {
                    "n",
                    "mean",
                    "std",
                    "minimum_best",
                    "p05",
                    "p10",
                    "p25",
                    "median",
                    "p75",
                    "p90",
                    "p95",
                    "maximum_worst",
                },
            )
            self.assertEqual(
                docking["le_normalisation"],
                {
                    "formula": "score_kcal_mol / heavy_atom_count",
                    "descriptor": (
                        "RDKit GetNumHeavyAtoms on the accepted parent at "
                        "molecule intake"
                    ),
                    "score_direction": "lower_is_better",
                    "note": (
                        "Reported alongside raw score; does not modify any "
                        "ranking metric."
                    ),
                },
            )

    def test_ligand_efficiency_does_not_disturb_the_raw_score_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            scores, intake = self._inputs(Path(temporary))
            _, profile = build_candidate_profile(
                scores,
                intake,
                name="generation_1",
                role="model",
            )
            raw = profile["docking"]["score_distribution_kcal_mol"]

            self.assertEqual(raw["n"], 2)
            self.assertAlmostEqual(raw["minimum_best"], -7.1000)
            self.assertAlmostEqual(raw["maximum_worst"], -6.5000)
            self.assertAlmostEqual(raw["median"], -6.8000)

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
