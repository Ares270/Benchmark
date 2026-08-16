"""Dock-time configuration capture and the score-versus-size diagnostic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.analysis import report, size
from src.analysis.dataset import AnalysisInputError
from src.harness import config as harness_config
from src.harness import dock


def docking_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["molecule_id", "label", "score"])
    frame["score_imputed"] = frame["score"].isna()
    return frame


def write_smi(path: Path, pairs) -> Path:
    path.write_text(
        "".join(f"{smiles} {mol_id}\n" for mol_id, smiles in pairs), encoding="utf-8"
    )
    return path


class HarnessConfigCaptureTests(unittest.TestCase):
    def test_smina_resolution_falls_back_to_active_python_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_bin = Path(temporary) / "bin"
            env_bin.mkdir()
            python_path = env_bin / "python"
            smina_path = env_bin / "smina"
            smina_path.write_text("fixture", encoding="utf-8")
            with (
                mock.patch.object(dock.shutil, "which", return_value=None),
                mock.patch.object(dock.sys, "executable", str(python_path)),
            ):
                resolved = dock.resolve_smina_executable()
        self.assertEqual(resolved, str(smina_path))

    def test_record_describes_the_configuration_in_effect(self):
        record = dock.harness_config_record(workers=3)
        self.assertEqual(record["schema_version"], "1")
        self.assertEqual(record["box_center"], list(harness_config.BOX_CENTER))
        self.assertEqual(record["box_size"], list(harness_config.BOX_SIZE))
        self.assertEqual(record["exhaustiveness"], harness_config.EXHAUSTIVENESS)
        self.assertEqual(record["seed"], harness_config.SEED)
        self.assertEqual(record["num_modes"], harness_config.NUM_MODES)
        self.assertEqual(record["cpu_per_job"], harness_config.SMINA_CPU)
        self.assertEqual(record["num_workers"], 3)
        self.assertEqual(record["scoring_function"], "vina")

    def test_absent_values_are_omitted_rather_than_faked(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "no_such_receptor.pdbqt"
            with mock.patch.object(harness_config, "RECEPTOR_PDBQT", missing), \
                    mock.patch.object(dock, "smina_version", return_value=None):
                record = dock.harness_config_record(workers=1)
        self.assertNotIn("receptor_sha256", record)
        self.assertNotIn("smina_version", record)
        self.assertEqual(record["receptor_path"], str(missing.resolve()))


class HarnessConfigReportTests(unittest.TestCase):
    def _cost(self, active_config, decoy_config) -> dict:
        return {
            "actives": {"harness_config": active_config} if active_config else {},
            "decoys": {"harness_config": decoy_config} if decoy_config else {},
        }

    def test_missing_block_is_reported_as_not_recorded(self):
        recorded = {"box_center": [1.0, 2.0, 3.0], "exhaustiveness": 8}
        rows, complete = report._harness_config_rows(
            self._cost(recorded, None)
        )
        self.assertFalse(complete)
        self.assertTrue(rows[0]["recorded"])
        self.assertFalse(rows[1]["recorded"])
        self.assertEqual(rows[0]["fields"]["box_center"], "1, 2, 3")
        self.assertEqual(rows[0]["fields"]["seed"], "n/a")

    def test_no_cost_records_leaves_both_cohorts_unrecorded(self):
        rows, complete = report._harness_config_rows(None)
        self.assertFalse(complete)
        self.assertEqual([row["recorded"] for row in rows], [False, False])

    def test_differing_scoring_configuration_aborts_the_report(self):
        base = {
            "receptor_sha256": "a" * 64,
            "box_center": [8.6, 17.7, 24.7],
            "box_size": [30.0, 28.0, 35.0],
            "exhaustiveness": 8,
            "scoring_function": "vina",
        }
        for field, replacement in (
            ("receptor_sha256", "b" * 64),
            ("box_size", [40.0, 38.0, 45.0]),
            ("exhaustiveness", 16),
            ("scoring_function", "vinardo"),
        ):
            with self.subTest(field=field):
                other = dict(base, **{field: replacement})
                with self.assertRaises(report.HarnessConfigurationMismatch) as caught:
                    report.check_harness_configuration(self._cost(base, other))
                self.assertIn(field, str(caught.exception))

    def test_identical_configurations_are_accepted(self):
        base = {"receptor_sha256": "a" * 64, "exhaustiveness": 8}
        rows, complete = report._harness_config_rows(self._cost(base, dict(base)))
        self.assertTrue(complete)
        self.assertTrue(all(row["recorded"] for row in rows))


class SizeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.actives = write_smi(
            self.directory / "actives.smi",
            [("A1", "CCO"), ("A2", "c1ccccc1"), ("A3", "CCCCCCCC")],
        )
        self.decoys = write_smi(
            self.directory / "decoys.smi",
            [("D1", "CC"), ("D2", "CCCC"), ("D3", "c1ccc2ccccc2c1")],
        )
        self.frame = docking_frame(
            [
                ("A1", 1, -7.0), ("A2", 1, -8.0), ("A3", 1, -9.0),
                ("D1", 0, -5.0), ("D2", 0, -6.0), ("D3", 0, -10.0),
            ]
        )

    def test_heavy_atom_counts_come_from_the_input_smiles(self):
        frame, profile = size.build_size_profile(
            self.frame, self.actives, self.decoys
        )
        counts = dict(zip(frame["molecule_id"], frame["heavy_atoms"]))
        self.assertEqual(counts["A1"], 3)     # CCO
        self.assertEqual(counts["A2"], 6)     # benzene
        self.assertEqual(counts["D3"], 10)    # naphthalene
        self.assertEqual(profile["n_analyzed"], 6)
        self.assertEqual(profile["n_correlated"], 6)
        self.assertEqual(profile["n_excluded_missing_score"], 0)

    def test_perfect_monotone_size_dependence_is_reported(self):
        frame = docking_frame(
            [("A1", 1, -3.0), ("A2", 1, -6.0), ("A3", 1, -8.0),
             ("D1", 0, -2.0), ("D2", 0, -4.0), ("D3", 0, -10.0)]
        )
        _, profile = size.build_size_profile(frame, self.actives, self.decoys)
        self.assertAlmostEqual(
            profile["correlations"]["pooled"]["spearman_rho"], -1.0
        )
        self.assertLess(profile["correlations"]["pooled"]["spearman_p_value"], 0.01)
        self.assertIsNotNone(profile["correlations"]["pooled"]["pearson_r"])

    def test_molecules_without_a_score_are_excluded_and_counted(self):
        frame = self.frame.copy()
        frame.loc[frame["molecule_id"].eq("A3"), "score"] = float("nan")
        frame["score_imputed"] = frame["score"].isna()
        _, profile = size.build_size_profile(frame, self.actives, self.decoys)
        self.assertEqual(profile["n_analyzed"], 6)
        self.assertEqual(profile["n_correlated"], 5)
        self.assertEqual(profile["n_excluded_missing_score"], 1)
        self.assertEqual(profile["heavy_atom_counts"]["actives"]["n"], 3)

    def test_unmapped_scored_id_raises_and_names_the_molecule(self):
        frame = pd.concat(
            [self.frame, docking_frame([("D9", 0, -7.5)])], ignore_index=True
        )
        with self.assertRaises(AnalysisInputError) as caught:
            size.build_size_profile(frame, self.actives, self.decoys)
        self.assertIn("decoys:D9", str(caught.exception))

    def test_unparsable_smiles_raises(self):
        broken = write_smi(
            self.directory / "broken.smi",
            [("D1", "CC"), ("D2", "CCCC"), ("D3", "not_a_smiles")],
        )
        with self.assertRaises(AnalysisInputError) as caught:
            size.build_size_profile(self.frame, self.actives, broken)
        self.assertIn("decoys:D3", str(caught.exception))

    def test_duplicate_ids_in_the_smi_input_raise(self):
        duplicated = write_smi(
            self.directory / "duplicated.smi",
            [("D1", "CC"), ("D2", "CCCC"), ("D3", "CCC"), ("D1", "CCCCC")],
        )
        with self.assertRaises(AnalysisInputError):
            size.build_size_profile(self.frame, self.actives, duplicated)

    def test_standardized_mean_difference_uses_the_pooled_sd(self):
        _, profile = size.build_size_profile(
            self.frame, self.actives, self.decoys
        )
        active_mean = profile["heavy_atom_counts"]["actives"]["mean"]
        decoy_mean = profile["heavy_atom_counts"]["decoys"]["mean"]
        difference = profile["standardized_mean_difference_actives_minus_decoys"]
        self.assertEqual(
            difference > 0, active_mean > decoy_mean
        )


if __name__ == "__main__":
    unittest.main()
