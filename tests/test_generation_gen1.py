from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.generation.gen1_analysis import (
    build_reference_index,
    write_gen1_analysis,
)
from src.generation.gen1_guacamol import (
    ARCHITECTURE,
    MODEL_NAME,
    MOLECULES_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    RAW_SAMPLES_NAME,
    SAMPLING_NAME,
    Gen1CheckpointError,
    GuacaMolSmilesLSTM,
    GuacaMolVocabulary,
    generate_gen1_samples,
    load_checkpoint,
    sample_smiles,
)
from src.generation.run_gen1_pipeline import Gen1CampaignError, run_gen1_campaign
from src.harness import config, intake, runtime


class Gen1CheckpointAndSamplingTests(unittest.TestCase):
    def _tiny_model(self) -> GuacaMolSmilesLSTM:
        torch.manual_seed(7)
        return GuacaMolSmilesLSTM(
            input_size=47,
            hidden_size=8,
            output_size=47,
            num_layers=1,
            rnn_dropout=0.0,
        ).eval()

    def test_vocabulary_matches_checkpoint_dimensions_and_multichar_decode(self):
        vocabulary = GuacaMolVocabulary()
        self.assertEqual(len(vocabulary.char_idx), ARCHITECTURE["input_size"])
        self.assertEqual(vocabulary.begin_idx, 1)
        self.assertEqual(vocabulary.end_idx, 2)
        self.assertEqual(
            vocabulary.decode_tokens([14, 15, 7, 3, 43, 46]),
            "BrClSiSe@@se",
        )

    def test_action_sampler_is_deterministic_and_retains_exact_count(self):
        model = self._tiny_model()
        vocabulary = GuacaMolVocabulary()
        first = sample_smiles(
            model,
            vocabulary,
            n=5,
            seed=20260801,
            max_length=12,
            batch_size=3,
        )
        second = sample_smiles(
            model,
            vocabulary,
            n=5,
            seed=20260801,
            max_length=12,
            batch_size=3,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(
            all(row["terminated_by_eos"] != row["hit_max_length"] for row in first)
        )

    def test_generated_transport_is_byte_identical_for_same_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"unit-test")
            provenance = {
                "declared_source": {"description": "synthetic unit test"},
                "weights": runtime.file_record(checkpoint),
                "architecture": ARCHITECTURE,
                "training": {},
            }
            model = self._tiny_model()
            with mock.patch(
                "src.generation.gen1_guacamol.load_checkpoint",
                return_value=(model, GuacaMolVocabulary(), provenance),
            ):
                first = root / "first"
                second = root / "second"
                generate_gen1_samples(checkpoint, first, n=8, seed=20260801)
                generate_gen1_samples(checkpoint, second, n=8, seed=20260801)

            self.assertEqual(
                (first / RAW_SAMPLES_NAME).read_bytes(),
                (second / RAW_SAMPLES_NAME).read_bytes(),
            )
            self.assertEqual(
                (first / MOLECULES_NAME).read_bytes(),
                (second / MOLECULES_NAME).read_bytes(),
            )
            summary = json.loads((first / SAMPLING_NAME).read_text())
            self.assertEqual(summary["counts"]["raw_samples"], 8)
            self.assertFalse(summary["interpretation"]["validity_filtering"])
            self.assertFalse(summary["interpretation"]["replacement_or_top_up"])

    def test_nonofficial_file_is_rejected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model_final_0.473.pt"
            checkpoint.write_bytes(b"not the official checkpoint")
            with self.assertRaisesRegex(Gen1CheckpointError, "byte count mismatch"):
                load_checkpoint(checkpoint)

    def test_downloaded_official_checkpoint_loads_strictly_when_present(self):
        checkpoint = (
            config.REPO_ROOT
            / "Models & Miscellaneous"
            / "model_final_0.473.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("local official checkpoint is not present")
        model, vocabulary, provenance = load_checkpoint(checkpoint)
        self.assertEqual(len(vocabulary.char_idx), 47)
        self.assertEqual(tuple(model.encoder.weight.shape), (47, 1024))
        self.assertEqual(tuple(model.decoder.weight.shape), (47, 1024))
        self.assertEqual(
            provenance["weights"]["sha256"], OFFICIAL_CHECKPOINT_SHA256
        )

    def test_smaller_nonvalidation_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Gen1CampaignError, "require --validation"):
                run_gen1_campaign(
                    Path(temporary) / "checkpoint.pt",
                    Path(temporary) / "output",
                    raw_n=10,
                    dock_n=5,
                )


class Gen1FullCohortAnalysisTests(unittest.TestCase):
    def _actives(self, root: Path) -> Path:
        path = root / "actives.csv"
        path.write_text(
            "molecule_chembl_id,canonical_smiles\nACTIVE_CCO,CCO\n",
            encoding="utf-8",
        )
        return path

    def _samples(self, root: Path) -> Path:
        sample_dir = root / "samples"
        sample_dir.mkdir()
        rows = [
            (1, "GEN1_S1_00001", "CCO", "CCO", "", True, False),
            (2, "GEN1_S1_00002", "CCO", "CCO", "", True, False),
            (3, "GEN1_S1_00003", "not_smiles", "not_smiles", "", True, False),
            (
                4,
                "GEN1_S1_00004",
                "",
                "__GEN1_EMPTY_SMILES__",
                "empty_smiles",
                True,
                False,
            ),
            (5, "GEN1_S1_00005", "c1ccccc1", "c1ccccc1", "", True, False),
        ]
        raw_path = sample_dir / RAW_SAMPLES_NAME
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_index",
                    "molecule_id",
                    "raw_smiles",
                    "transport_smiles",
                    "transport_encoding",
                    "token_count",
                    "character_length",
                    "terminated_by_eos",
                    "hit_max_length",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            for index, molecule_id, raw, transport, encoding, eos, maximum in rows:
                writer.writerow(
                    {
                        "sample_index": index,
                        "molecule_id": molecule_id,
                        "raw_smiles": raw,
                        "transport_smiles": transport,
                        "transport_encoding": encoding,
                        "token_count": len(raw),
                        "character_length": len(raw),
                        "terminated_by_eos": eos,
                        "hit_max_length": maximum,
                    }
                )
        smi_path = sample_dir / MOLECULES_NAME
        smi_path.write_text(
            "".join(
                f"{transport} {molecule_id}\n"
                for _, molecule_id, _, transport, *_ in rows
            ),
            encoding="utf-8",
        )
        sampling = {
            "schema_version": 1,
            "stage": "gen1_sampling",
            "model_name": MODEL_NAME,
            "parameters": {"seed": 1},
            "counts": {
                "raw_samples": 5,
                "terminated_by_eos": 5,
                "hit_max_length": 0,
                "empty_raw_smiles": 1,
            },
            "raw_character_length": {
                "mean": 4.8,
                "median": 3.0,
                "minimum": 0,
                "maximum": 11,
            },
            "checkpoint": {
                "weights": {"sha256": "synthetic"},
                "declared_source": {"description": "synthetic"},
            },
            "outputs": {
                "raw_samples_csv": runtime.file_record(raw_path),
                "molecules_smi": runtime.file_record(smi_path),
            },
        }
        runtime.write_json_atomic(sample_dir / SAMPLING_NAME, sampling)
        return sample_dir

    def test_full_report_counts_invalid_duplicate_scaffold_and_active_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = self._samples(root)
            intake_dir = root / "intake"
            intake.run_intake(sample_dir / MOLECULES_NAME, intake_dir)
            report = write_gen1_analysis(
                sample_dir,
                intake_dir,
                self._actives(root),
                root / "analysis",
            )

            metrics = json.loads((report.parent / "metrics.json").read_text())
            quality = metrics["generation_quality"]
            self.assertEqual(quality["valid_structures"], 3)
            self.assertEqual(quality["unique_valid_parents"], 2)
            self.assertEqual(quality["accepted_unique_parents"], 2)
            self.assertEqual(quality["exact_known_active_parent_rediscoveries"], 1)
            self.assertEqual(quality["unique_nonempty_bemis_murcko_scaffolds"], 1)
            self.assertEqual(quality["acyclic_accepted_parents"], 1)
            self.assertFalse(quality["training_novelty"]["available"])
            self.assertIn(
                "Gen1: target-unaware GuacaMol SMILES-LSTM", report.read_text()
            )

    def test_reference_index_enables_parent_and_scaffold_novelty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "guacamol_v1_train.smiles"
            dataset.write_text("CCO\n", encoding="utf-8")
            index_dir = root / "reference"
            build_reference_index(
                dataset,
                index_dir,
                source_description="Synthetic GuacaMol-shaped source",
            )
            sample_dir = self._samples(root)
            intake_dir = root / "intake"
            intake.run_intake(sample_dir / MOLECULES_NAME, intake_dir)
            report = write_gen1_analysis(
                sample_dir,
                intake_dir,
                self._actives(root),
                root / "analysis",
                reference_index_dir=index_dir,
            )
            novelty = json.loads(
                (report.parent / "metrics.json").read_text()
            )["generation_quality"]["training_novelty"]
            self.assertTrue(novelty["available"])
            self.assertEqual(novelty["novel_unique_parents"], 1)
            self.assertEqual(novelty["novel_nonempty_scaffolds"], 1)


if __name__ == "__main__":
    unittest.main()
