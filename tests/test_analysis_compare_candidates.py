from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.chemistry import PROPERTY_COLUMNS
from src.analysis.compare_candidates import (
    CandidateComparisonError,
    build_candidate_comparison,
    write_candidate_comparison,
)


class CandidateComparisonTests(unittest.TestCase):
    def _run(self, name: str, role: str, submitted: int = 100) -> dict:
        parameters = {
            "smina_cpu_per_job": 1,
            "box_center_a": [8.631, 17.703, 24.73],
            "box_size_a": [40.0, 38.0, 45.0],
            "exhaustiveness": 8,
            "seed": 42,
            "num_modes": 9,
            "energy_range_kcal_mol": 3.0,
            "timeout_seconds_per_ligand": 600,
        }
        return {
            "schema_version": 1,
            "stage": "candidate_analysis",
            "name": name,
            "role": role,
            "source_description": (
                "fixed-seed random sample from the declared holdout universe"
                if role == "naive_baseline"
                else f"{name} generated candidates"
            ),
            "intake": {
                "submitted_rows": submitted,
                "validity": 0.95,
                "parent_uniqueness": 0.90,
                "accepted_for_preparation": 85,
            },
            "docking": {
                "coverage_over_accepted_parents": 0.80,
                "successful_per_submitted": 0.68,
                "score_distribution_kcal_mol": {
                    "p10": -8.0,
                    "median": -7.0,
                    "mean": -7.1,
                },
            },
            "chemistry": {
                "properties": {
                    column: {"mean": float(index + 1)}
                    for index, column in enumerate(PROPERTY_COLUMNS)
                }
            },
            "computational_cost": {
                "timing": {"estimated_requested_cpu_slot_hours": 2.5}
            },
            "screening_protocol": {
                "engine_stage": "smina_docking",
                "receptor": {"sha256": "a" * 64, "missing": False},
                "parameters": parameters,
            },
        }

    def _files(self, root: Path, submitted: list[int] | None = None) -> list[Path]:
        submitted = submitted or [100, 100, 100, 100]
        definitions = [
            ("generation_1", "model"),
            ("generation_2", "model"),
            ("generation_3", "model"),
            ("naive", "naive_baseline"),
        ]
        paths = []
        for index, ((name, role), budget) in enumerate(zip(definitions, submitted)):
            path = root / f"run_{index}.json"
            path.write_text(
                json.dumps(self._run(name, role, budget)),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_four_run_contract_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            runs, summary, chemistry, checks = build_candidate_comparison(paths)

            self.assertEqual(len(runs), 4)
            self.assertEqual(summary["role"].value_counts().to_dict(), {"model": 3, "naive_baseline": 1})
            self.assertEqual(list(chemistry.columns), list(PROPERTY_COLUMNS))
            self.assertEqual(checks["equal_submitted_molecule_budget"], 100)
            self.assertTrue(checks["no_composite_winner_score"])

            report = write_candidate_comparison(paths, root / "comparison")
            self.assertTrue(report.is_file())
            self.assertIn("no overall winner score", report.read_text(encoding="utf-8"))
            self.assertTrue((report.parent / "comparison.json").is_file())

    def test_unequal_submitted_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._files(Path(temporary), [100, 100, 99, 100])
            with self.assertRaisesRegex(
                CandidateComparisonError,
                "budgets differ",
            ):
                build_candidate_comparison(paths)


if __name__ == "__main__":
    unittest.main()
