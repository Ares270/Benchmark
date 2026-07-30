from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from src.generation import filter_config
from src.generation.filter import apply_gate, run_gate
from src.harness import runtime
from src.harness.intake import _largest_fragment_parent


REPO_ROOT = Path(__file__).resolve().parents[1]


class GenerationFilterTests(unittest.TestCase):
    def test_size_and_torsion_boundaries_are_inclusive(self):
        ring_9 = "C1" + ("C" * 8) + "1"
        ring_10 = "C1" + ("C" * 9) + "1"
        ring_62 = "C1" + ("C" * 61) + "1"
        ring_63 = "C1" + ("C" * 62) + "1"
        torsions_15 = "C" * 18
        torsions_16 = "C" * 19
        result = apply_gate(
            [
                ("size_below", ring_9),
                ("size_min", ring_10),
                ("size_max", ring_62),
                ("size_above", ring_63),
                ("torsion_max", torsions_15),
                ("torsion_above", torsions_16),
            ],
            filter_config,
        )
        decisions = {
            decision.molecule_id: decision for decision in result.decisions
        }

        self.assertEqual(decisions["size_min"].heavy_atoms, filter_config.MIN_HEAVY_ATOMS)
        self.assertTrue(decisions["size_min"].passed)
        self.assertEqual(decisions["size_max"].heavy_atoms, filter_config.MAX_HEAVY_ATOMS)
        self.assertTrue(decisions["size_max"].passed)
        self.assertFalse(decisions["size_below"].pass_size)
        self.assertFalse(decisions["size_above"].pass_size)
        self.assertEqual(
            decisions["torsion_max"].rotatable_bonds,
            filter_config.MAX_ROTATABLE_BONDS,
        )
        self.assertTrue(decisions["torsion_max"].passed)
        self.assertEqual(
            decisions["torsion_above"].rotatable_bonds,
            filter_config.MAX_ROTATABLE_BONDS + 1,
        )
        self.assertFalse(decisions["torsion_above"].pass_torsions)

    def test_boron_and_silicon_are_rejected_and_histogrammed(self):
        result = apply_gate(
            [
                ("boron", "CCCCBCCCCCC"),
                ("silicon", "CCCC[Si](C)CCCCC"),
            ],
            filter_config,
        )

        self.assertEqual(result.disallowed_element_histogram, {"B": 1, "Si": 1})
        self.assertTrue(all(not decision.pass_elements for decision in result.decisions))
        self.assertTrue(all(not decision.passed for decision in result.decisions))

    def test_multiple_failures_are_counted_independently(self):
        oversized_boron_ring = "B1" + ("C" * 62) + "1"
        result = apply_gate(
            [("two_gates", oversized_boron_ring)],
            filter_config,
        )

        self.assertEqual(result.exclusions_by_gate["elements"], 1)
        self.assertEqual(result.exclusions_by_gate["size_high"], 1)
        self.assertGreater(
            sum(result.exclusions_by_gate.values()),
            len(result.decisions) - len(result.passed_records),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "parents.smi"
            input_path.write_text(
                f"{oversized_boron_ring} two_gates\n",
                encoding="utf-8",
            )
            summary = run_gate(input_path, root / "gate")
            self.assertIn("may sum to more than", summary["counting_note"])

    def _active_gate_result(self):
        active_path = REPO_ROOT / "data/reference/dyrk1a_actives_chembl.csv"
        records = []
        with active_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                molecule = Chem.MolFromSmiles(row["canonical_smiles"])
                self.assertIsNotNone(molecule, row["molecule_chembl_id"])
                _, parent_smiles, _ = _largest_fragment_parent(molecule)
                records.append((row["molecule_chembl_id"], parent_smiles))

        return apply_gate(records, filter_config)

    def test_all_dyrk1a_actives_pass_derived_size_and_torsion_bounds(self):
        """The active-derived numeric bounds must contain every deriving active."""

        result = self._active_gate_result()
        self.assertEqual(len(result.decisions), 1219)
        self.assertTrue(all(decision.pass_size for decision in result.decisions))
        self.assertTrue(all(decision.pass_torsions for decision in result.decisions))

    def test_exactly_two_silicon_actives_fail_only_the_element_gate(self):
        """ALLOWED_ELEMENTS came from AutoDock/Vina atom-type parameterization.

        It was not derived from the actives. Requiring the actives to satisfy a
        bound they had no role in setting was an unearned assumption in the
        original specification.
        """

        result = self._active_gate_result()
        excluded = {
            decision.molecule_id: decision
            for decision in result.decisions
            if not decision.passed
        }

        self.assertEqual(len(result.passed_records), 1217)
        self.assertEqual(
            set(excluded),
            {"CHEMBL4288096", "CHEMBL5176894"},
        )
        for decision in excluded.values():
            self.assertEqual(decision.disallowed_elements, ("Si",))
            self.assertFalse(decision.pass_elements)
            self.assertTrue(decision.pass_size)
            self.assertTrue(decision.pass_torsions)

    def test_summary_records_config_hash_and_literal_constants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "parents.smi"
            input_path.write_text("C1CCCCCCCCC1 mol1\n", encoding="utf-8")
            output_dir = root / "gate"

            summary = run_gate(input_path, output_dir)
            on_disk = json.loads(
                (output_dir / "gate_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary, on_disk)
            self.assertEqual(
                summary["filter_config_sha256"],
                runtime.sha256_file(Path(filter_config.__file__)),
            )
            self.assertEqual(
                summary["constants"],
                {
                    "allowed_elements": sorted(filter_config.ALLOWED_ELEMENTS),
                    "min_heavy_atoms": filter_config.MIN_HEAVY_ATOMS,
                    "max_heavy_atoms": filter_config.MAX_HEAVY_ATOMS,
                    "max_rotatable_bonds": filter_config.MAX_ROTATABLE_BONDS,
                },
            )

    def test_gate_decisions_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "parents.smi"
            input_path.write_text(
                "C1CCCCCCCCC1 first\nCCCCBCCCCCC second\n",
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"

            run_gate(input_path, first)
            run_gate(input_path, second)

            self.assertEqual(
                (first / "gate_decisions.csv").read_bytes(),
                (second / "gate_decisions.csv").read_bytes(),
            )

    def test_cli_and_runner_expose_no_threshold_override(self):
        runner_parameters = set(inspect.signature(run_gate).parameters)
        self.assertEqual(runner_parameters, {"input_path", "output_dir"})
        main_source = inspect.getsource(
            __import__("src.generation.filter", fromlist=["main"]).main
        )
        self.assertNotIn("--min-heavy", main_source)
        self.assertNotIn("--max-heavy", main_source)
        self.assertNotIn("--max-rotatable", main_source)


if __name__ == "__main__":
    unittest.main()
