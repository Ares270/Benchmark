from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.generation.gen3_molexar import (
    DEFAULT_CONFIG_PATH,
    MODEL_NAME,
    MOLECULES_NAME,
    RAW_SAMPLES_NAME,
    SAMPLING_NAME,
    Gen3CheckpointError,
    Gen3ConfigurationError,
    generate_gen3_samples,
    load_gen3_config,
    sample_molecules,
    verify_checkpoint,
)


class _ToyEngine:
    device = "cpu"
    config = SimpleNamespace(gvp_node_in_dim=[11, 3])

    def generate(self, **kwargs):
        count = int(kwargs["num_samples"])
        values = torch.randint(1, 4, (count, 4))
        return ["".join("[C]" for _ in row.tolist()) for row in values]

    def convert_to_smiles(self, rows, **kwargs):
        self.conversion_kwargs = kwargs
        return [(row, "C" * max(1, row.count("[C]"))) for row in rows]


class Gen3ConfigurationAndSamplingTests(unittest.TestCase):
    def test_locked_config_pocket_and_registered_design_validate(self):
        specification = load_gen3_config()
        self.assertEqual(specification["arm"]["name"], MODEL_NAME)
        self.assertEqual(specification["target"]["center_angstrom"], [8.631, 17.703, 24.73])
        self.assertEqual(specification["target"]["atoms_used"], 425)
        self.assertEqual(specification["sampling"]["raw_samples"], 10000)
        self.assertEqual(specification["sampling"]["docking_subsample"], 1000)
        self.assertFalse(specification["sampling"]["replacement_or_top_up"])

    def test_changed_pocket_center_is_rejected(self):
        specification = load_gen3_config()
        specification["target"]["center_angstrom"][0] += 1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(specification), encoding="utf-8")
            with self.assertRaisesRegex(Gen3ConfigurationError, "docking center"):
                load_gen3_config(path)

    def test_nonofficial_checkpoint_is_rejected_before_model_load(self):
        specification = load_gen3_config()
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for filename in specification["model"]["artifacts"]:
                (model_dir / filename).write_bytes(b"not official")
            with self.assertRaisesRegex(Gen3CheckpointError, "byte count mismatch"):
                verify_checkpoint(model_dir, specification)

    def test_downloaded_official_checkpoint_authenticates_when_present(self):
        repository_root = DEFAULT_CONFIG_PATH.parents[1]
        model_dir = repository_root / "Models & Miscellaneous/molexar-10m-omni"
        if not (model_dir / "pytorch_model.bin").is_file():
            self.skipTest("local official Molexar checkpoint is not present")
        provenance = verify_checkpoint(model_dir, load_gen3_config())
        self.assertEqual(
            provenance["weights"]["sha256"],
            load_gen3_config()["model"]["artifacts"]["pytorch_model.bin"]["sha256"],
        )

    def test_sampler_is_deterministic_and_retains_exact_count(self):
        sampling = dict(load_gen3_config()["sampling"])
        first = sample_molecules(
            _ToyEngine(), {}, n=7, seed=20260801, sampling=sampling
        )
        second = sample_molecules(
            _ToyEngine(), {}, n=7, seed=20260801, sampling=sampling
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertTrue(all(row["conversion_success"] for row in first))

    def test_generated_transport_is_reproducible_with_mocked_bundle(self):
        specification = load_gen3_config()
        checkpoint = {
            "declared_model": specification["model"],
            "weights": {
                "sha256": specification["model"]["artifacts"]["pytorch_model.bin"]["sha256"]
            },
            "runtime": specification["runtime"],
        }
        pocket = {
            "all_nonhydrogen_atoms": 5702,
            "atoms_within_radius_before_truncation": 1849,
            "atoms_used": 425,
            "center_angstrom": [8.631, 17.703, 24.73],
            "radius_angstrom": 25.0,
            "max_atoms": 425,
            "knn_k": 8,
            "edge_count": 3400,
            "graph_sha256": "toy",
            "tensor_shapes": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_record = {"path": str(root / "target.pdb"), "bytes": 1, "sha256": "x"}
            with mock.patch(
                "src.generation.gen3_molexar.verify_target",
                return_value=target_record,
            ), mock.patch(
                "src.generation.gen3_molexar.load_model_bundle",
                return_value=(_ToyEngine(), checkpoint),
            ), mock.patch(
                "src.generation.gen3_molexar.build_pocket_graph",
                return_value=({}, pocket),
            ):
                first = root / "first"
                second = root / "second"
                generate_gen3_samples(root / "model", root / "target.pdb", first, n=8, seed=20260801)
                generate_gen3_samples(root / "model", root / "target.pdb", second, n=8, seed=20260801)
            self.assertEqual((first / RAW_SAMPLES_NAME).read_bytes(), (second / RAW_SAMPLES_NAME).read_bytes())
            self.assertEqual((first / MOLECULES_NAME).read_bytes(), (second / MOLECULES_NAME).read_bytes())
            summary = json.loads((first / SAMPLING_NAME).read_text())
            self.assertEqual(summary["stage"], "gen3_sampling")
            self.assertEqual(summary["counts"]["raw_samples"], 8)
            self.assertTrue(summary["interpretation"]["target_aware"])
            self.assertFalse(summary["interpretation"]["replacement_or_top_up"])
            self.assertFalse(summary["interpretation"]["active_ligand_or_docking_reward_used"])


if __name__ == "__main__":
    unittest.main()
