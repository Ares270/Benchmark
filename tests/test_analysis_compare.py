from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.chemistry import PROPERTY_COLUMNS
from src.analysis.compare import (
    build_chemistry_comparison_table,
    build_comparison_table,
    render_comparison,
)


class ComparisonTests(unittest.TestCase):
    def test_build_and_render_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name, auc, ef in (("baseline", 0.70, 4.0), ("model", 0.80, 7.0)):
                path = root / f"{name}.json"
                path.write_text(json.dumps({
                    "schema_version": 2,
                    "name": name,
                    "metrics": {"auc": auc, "bedroc": auc - 0.1, "ef_1pct": ef},
                }), encoding="utf-8")
                paths.append(path)
            table = build_comparison_table(paths)
            self.assertEqual(list(table.index), ["baseline", "model"])
            output = render_comparison(table, root / "comparison.html")
            self.assertTrue(output.is_file())
            self.assertIn("own vertical scale", output.read_text(encoding="utf-8"))


    def test_chemistry_means_are_comparable_across_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name, offset in (("model_a", 0.0), ("model_b", 2.0)):
                properties = {
                    column: {"mean": offset + index / 10}
                    for index, column in enumerate(PROPERTY_COLUMNS)
                }
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 3,
                            "name": name,
                            "metrics": {"auc": 0.5 + offset / 10},
                            "chemistry": {
                                "cohorts": {
                                    "actives": {"properties": properties}
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            chemistry_table = build_chemistry_comparison_table(paths)
            self.assertIsNotNone(chemistry_table)
            self.assertAlmostEqual(
                chemistry_table.loc["model_b", "clogp"]
                - chemistry_table.loc["model_a", "clogp"],
                2.0,
            )
            docking_table = build_comparison_table(paths)
            output = render_comparison(
                docking_table,
                root / "comparison.html",
                chemistry_table,
            )
            self.assertIn(
                "Mean evaluated-parent properties",
                output.read_text(encoding="utf-8"),
            )

    def test_duplicate_method_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(2):
                path = root / f"run_{index}.json"
                path.write_text(json.dumps({"name": "same", "metrics": {"auc": 0.5}}), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "unique"):
                build_comparison_table(paths)


if __name__ == "__main__":
    unittest.main()
