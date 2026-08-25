from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.generation.gen2_warmmolgenone import (
    DEFAULT_CONFIG_PATH,
    MODEL_NAME,
    MOLECULES_NAME,
    RAW_SAMPLES_NAME,
    SAMPLING_NAME,
    Gen2CheckpointError,
    Gen2ConfigurationError,
    generate_gen2_samples,
    load_gen2_config,
    load_model_bundle,
    sample_smiles,
    verify_checkpoint,
)


class _ProteinTokenizer:
    def __call__(self, sequence: str, *, return_tensors: str):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[len(sequence)]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }


class _SmilesTokenizer:
    bos_token_id = 0
    eos_token_id = 2

    def batch_decode(self, rows, *, skip_special_tokens: bool):
        assert skip_special_tokens
        alphabet = {3: "C", 4: "N", 5: "O"}
        return ["".join(alphabet.get(int(token), "") for token in row) for row in rows]


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        count = int(kwargs["num_return_sequences"])
        middle = torch.randint(3, 6, (count, 4), device=self.anchor.device)
        start = torch.zeros((count, 1), dtype=torch.long, device=self.anchor.device)
        end = torch.full((count, 1), 2, dtype=torch.long, device=self.anchor.device)
        return torch.cat((start, middle, end), dim=1)


class Gen2ConfigurationAndSamplingTests(unittest.TestCase):
    def test_locked_config_sequence_and_registered_design_validate(self):
        specification = load_gen2_config()
        self.assertEqual(specification["arm"]["name"], MODEL_NAME)
        self.assertEqual(len(specification["target"]["sequence"]), 359)
        self.assertEqual(specification["sampling"]["raw_samples"], 10000)
        self.assertEqual(specification["sampling"]["docking_subsample"], 1000)
        self.assertFalse(specification["sampling"]["replacement_or_top_up"])

    def test_changed_sequence_is_rejected_by_hash(self):
        specification = load_gen2_config()
        specification["target"]["sequence"] = (
            "A" + specification["target"]["sequence"][1:]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(specification), encoding="utf-8")
            with self.assertRaisesRegex(Gen2ConfigurationError, "SHA-256"):
                load_gen2_config(path)

    def test_nonofficial_checkpoint_is_rejected_before_model_load(self):
        specification = load_gen2_config()
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "pytorch_model.bin").write_bytes(b"not official")
            with self.assertRaisesRegex(Gen2CheckpointError, "byte count mismatch"):
                verify_checkpoint(model_dir, specification)

    def test_downloaded_official_checkpoint_and_tokenizers_load_when_present(self):
        repository_root = DEFAULT_CONFIG_PATH.parents[1]
        model_dir = repository_root / "Models & Miscellaneous"
        decoder_tokenizer_dir = (
            model_dir / "PubChem10M_SMILES_BPE_450k"
        )
        if not (model_dir / "pytorch_model.bin").is_file():
            self.skipTest("local official WarmMolGenOne checkpoint is not present")
        if not decoder_tokenizer_dir.is_dir():
            self.skipTest("local official decoder tokenizer is not present")
        model, protein_tokenizer, smiles_tokenizer, provenance = load_model_bundle(
            model_dir,
            decoder_tokenizer_dir,
            device="cpu",
            specification=load_gen2_config(),
        )
        self.assertEqual(model.config.encoder.num_hidden_layers, 12)
        self.assertEqual(model.config.decoder.num_hidden_layers, 6)
        self.assertEqual(len(protein_tokenizer), 10261)
        self.assertEqual(len(smiles_tokenizer), 7924)
        self.assertEqual(
            provenance["weights"]["sha256"],
            load_gen2_config()["model"]["weights_sha256"],
        )

    def test_sampler_is_deterministic_and_retains_exact_count(self):
        specification = load_gen2_config()
        sampling = dict(specification["sampling"])
        sampling["batch_size"] = 3
        sampling["max_length_decoder_tokens"] = 8
        first = sample_smiles(
            _ToyModel(), _ProteinTokenizer(), _SmilesTokenizer(),
            specification["target"]["sequence"], n=7, seed=20260801,
            sampling=sampling,
        )
        second = sample_smiles(
            _ToyModel(), _ProteinTokenizer(), _SmilesTokenizer(),
            specification["target"]["sequence"], n=7, seed=20260801,
            sampling=sampling,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertTrue(all(row["terminated_by_eos"] for row in first))

    def test_generated_transport_is_reproducible_with_mocked_bundle(self):
        specification = load_gen2_config()
        provenance = {
            "declared_model": specification["model"],
            "weights": {"sha256": specification["model"]["weights_sha256"]},
        }
        bundle = (_ToyModel(), _ProteinTokenizer(), _SmilesTokenizer(), provenance)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "src.generation.gen2_warmmolgenone.load_model_bundle",
                return_value=bundle,
            ):
                first = root / "first"
                second = root / "second"
                generate_gen2_samples(
                    root / "model", root / "tokenizer", first,
                    n=8, seed=20260801,
                )
                generate_gen2_samples(
                    root / "model", root / "tokenizer", second,
                    n=8, seed=20260801,
                )
            self.assertEqual((first / RAW_SAMPLES_NAME).read_bytes(), (second / RAW_SAMPLES_NAME).read_bytes())
            self.assertEqual((first / MOLECULES_NAME).read_bytes(), (second / MOLECULES_NAME).read_bytes())
            summary = json.loads((first / SAMPLING_NAME).read_text())
            self.assertEqual(summary["stage"], "gen2_sampling")
            self.assertEqual(summary["counts"]["raw_samples"], 8)
            self.assertTrue(summary["interpretation"]["target_aware"])
            self.assertFalse(summary["interpretation"]["replacement_or_top_up"])
            self.assertEqual(summary["parameters"]["seed"], 20260801)
            self.assertEqual(summary["parameters"]["raw_samples"], 8)


if __name__ == "__main__":
    unittest.main()
