from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.generation.run_all_arms import AllArmsCampaignError, run_all_arms


class AllArmsPreflightTests(unittest.TestCase):
    def test_existing_output_is_rejected_before_any_arm_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(AllArmsCampaignError, "already exists"):
                run_all_arms(output)

    def test_missing_inputs_are_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            missing = root / "missing"
            with self.assertRaisesRegex(AllArmsCampaignError, "inputs are missing"):
                run_all_arms(
                    output,
                    baseline_smi=missing,
                    gen1_checkpoint=missing,
                    gen2_model_dir=missing,
                    gen2_tokenizer_dir=missing,
                    gen3_model_dir=missing,
                    molexar_python=missing,
                )
            self.assertFalse(output.exists())

    def test_four_arm_handoff_and_comparison_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.smi"
            gen1 = root / "gen1.pt"
            gen2 = root / "gen2"
            gen2_tokenizer = root / "gen2_tokenizer"
            gen3 = root / "gen3"
            molexar_python = root / "python"
            baseline.write_text("C BASELINE_1\n", encoding="utf-8")
            gen1.write_bytes(b"checkpoint")
            gen2.mkdir()
            (gen2 / "pytorch_model.bin").write_bytes(b"checkpoint")
            gen2_tokenizer.mkdir()
            (gen2_tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
            gen3.mkdir()
            (gen3 / "pytorch_model.bin").write_bytes(b"checkpoint")
            molexar_python.write_text("", encoding="utf-8")

            def fake_baseline(*args, **kwargs):
                run_dir = Path(kwargs["analysis_outdir"]) / "run"
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
                report = run_dir / "report.html"
                report.write_text("baseline", encoding="utf-8")
                return report

            def fake_campaign(*args, **kwargs):
                campaign_dir = Path(args[-1])
                campaign_dir.mkdir(parents=True)
                metrics = campaign_dir / "metrics.json"
                metrics.write_text("{}\n", encoding="utf-8")
                (campaign_dir / "campaign_summary.json").write_text(
                    json.dumps(
                        {"outputs": {"candidate_metrics": {"path": str(metrics)}}}
                    ),
                    encoding="utf-8",
                )
                report = campaign_dir / "report.html"
                report.write_text("campaign", encoding="utf-8")
                return report

            def fake_comparison(metrics_paths, outdir):
                self.assertEqual(len(metrics_paths), 4)
                self.assertTrue(all(Path(path).is_file() for path in metrics_paths))
                outdir = Path(outdir)
                outdir.mkdir(parents=True)
                report = outdir / "report.html"
                report.write_text("comparison", encoding="utf-8")
                return report

            output = root / "output"
            with mock.patch(
                "src.generation.run_all_arms.run_candidate_pipeline",
                side_effect=fake_baseline,
            ), mock.patch(
                "src.generation.run_all_arms.run_gen1_campaign",
                side_effect=fake_campaign,
            ), mock.patch(
                "src.generation.run_all_arms.run_gen2_campaign",
                side_effect=fake_campaign,
            ), mock.patch(
                "src.generation.run_all_arms.run_gen3_campaign",
                side_effect=fake_campaign,
            ), mock.patch(
                "src.generation.run_all_arms.write_candidate_comparison",
                side_effect=fake_comparison,
            ):
                report = run_all_arms(
                    output,
                    baseline_smi=baseline,
                    gen1_checkpoint=gen1,
                    gen2_model_dir=gen2,
                    gen2_tokenizer_dir=gen2_tokenizer,
                    gen3_model_dir=gen3,
                    molexar_python=molexar_python,
                )
            self.assertEqual(report, output / "four_arm_comparison/report.html")
            summary = json.loads((output / "all_arms_summary.json").read_text())
            self.assertEqual(summary["stage"], "registered_all_arms_campaign")
            self.assertEqual(set(summary["arms"]), {
                "naive_property_matched",
                "gen1_guacamol",
                "gen2_warmmolgenone",
                "gen3_molexar",
            })


if __name__ == "__main__":
    unittest.main()
