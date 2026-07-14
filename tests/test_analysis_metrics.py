from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import metrics


class RankingMetricTests(unittest.TestCase):
    def test_perfect_and_reversed_auc(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([-10.0, -9.0, -5.0, -4.0])
        self.assertAlmostEqual(metrics.roc_curve(labels, scores)[2], 1.0)
        self.assertAlmostEqual(metrics.roc_curve(labels, -scores)[2], 0.0)
        self.assertAlmostEqual(metrics.enrichment_factor(labels, scores, 0.5), 2.0)

    def test_higher_is_better_is_operational_not_documentary(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([10.0, 9.0, 5.0, 4.0])
        auc = metrics.roc_curve(labels, scores, "higher_is_better")[2]
        self.assertAlmostEqual(auc, 1.0)

    def test_all_ties_are_neutral_and_permutation_invariant(self):
        labels = np.array([1, 0, 1, 0, 0, 1])
        scores = np.full(labels.size, -7.0)
        baseline_bedroc = metrics.bedroc(labels, scores)
        for permutation in (
            np.arange(labels.size),
            np.array([0, 2, 5, 1, 3, 4]),
            np.array([4, 3, 1, 5, 2, 0]),
        ):
            permuted_labels = labels[permutation]
            self.assertAlmostEqual(metrics.roc_curve(permuted_labels, scores)[2], 0.5)
            self.assertAlmostEqual(metrics.enrichment_factor(permuted_labels, scores, 0.01), 1.0)
            self.assertAlmostEqual(metrics.bedroc(permuted_labels, scores), baseline_bedroc)

    def test_operating_point_uses_integer_slice_size(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([-10.0, -9.0, -8.0, -7.0])
        actual_fraction, recovered, ef = metrics.enrichment_operating_point(labels, scores, 0.26)
        self.assertEqual(actual_fraction, 0.5)
        self.assertEqual(recovered, 0.5)
        self.assertEqual(ef, 1.0)

    def test_bedroc_matches_rdkit_for_untied_ranking(self):
        try:
            from rdkit.ML.Scoring.Scoring import CalcBEDROC
        except ImportError:
            self.skipTest("RDKit is not installed")
        labels = np.array([1, 0, 1, 0, 0, 1, 0])
        scores = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
        ordered_labels = labels[np.argsort(scores)]
        rdkit_value = CalcBEDROC([[int(label)] for label in ordered_labels], 0, 20.0)
        self.assertAlmostEqual(metrics.bedroc(labels, scores, alpha=20.0), rdkit_value, places=12)

    def test_fixture_regression(self):
        root = Path(__file__).resolve().parents[1]
        active = pd.read_csv(root / "data" / "test10_scores.csv")
        decoy = pd.read_csv(root / "data" / "fake_decoys_scores.csv")
        labels = np.r_[np.ones(len(active), dtype=int), np.zeros(len(decoy), dtype=int)]
        scores = np.r_[active["score_kcal_mol"], decoy["score_kcal_mol"]]
        values = metrics.summary_metrics(labels, scores)
        self.assertAlmostEqual(values["auc"], 0.9495, places=12)
        self.assertAlmostEqual(values["bedroc"], 0.7562949670579081, places=12)
        self.assertAlmostEqual(values["ef_1pct"], 21.0)
        self.assertAlmostEqual(values["ef_5pct"], 11.454545454545455)
        self.assertAlmostEqual(values["ef_10pct"], 8.0)

    def test_bootstrap_is_reproducible(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 0])
        scores = np.array([-10.0, -8.0, -6.0, -9.0, -7.0, -5.0, -4.0])
        first = metrics.bootstrap_confidence_intervals(labels, scores, n_resamples=25, seed=123)
        second = metrics.bootstrap_confidence_intervals(labels, scores, n_resamples=25, seed=123)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
