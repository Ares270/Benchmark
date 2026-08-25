from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.benchmark import (
    BenchmarkConsoleError,
    build_run_command,
    configure_run,
    launch,
    status,
)


class BenchmarkCommandTests(unittest.TestCase):
    def test_gen3_pilot_maps_only_to_existing_validation_flags(self):
        command = build_run_command(
            "gen3", Path("results/pilot"),
            pilot=True, raw_n=25, dock_n=5, workers=3,
        )
        self.assertIn("src.generation.run_gen3_pipeline", command)
        self.assertEqual(command[command.index("--raw-n") + 1], "25")
        self.assertEqual(command[command.index("--dock-n") + 1], "5")
        self.assertEqual(command[command.index("--workers") + 1], "3")
        self.assertIn("--validation", command)

    def test_registered_gen1_does_not_expose_count_overrides(self):
        command = build_run_command(
            "gen1", Path("results/registered"),
            pilot=False, raw_n=100, dock_n=10, workers=4,
        )
        self.assertNotIn("--validation", command)
        self.assertNotIn("--raw-n", command)
        self.assertNotIn("--dock-n", command)

    def test_count_overrides_require_pilot_label(self):
        with self.assertRaisesRegex(BenchmarkConsoleError, "pilot-only"):
            build_run_command(
                "gen1", Path("results/mislabelled"),
                pilot=False, raw_n=50, dock_n=5, workers=4,
            )

    def test_all_arm_pilot_is_rejected(self):
        with self.assertRaisesRegex(BenchmarkConsoleError, "registered-only"):
            build_run_command(
                "all", Path("results/all"),
                pilot=True, raw_n=10, dock_n=2, workers=1,
            )

    def test_invalid_counts_are_rejected_before_launch(self):
        with self.assertRaisesRegex(BenchmarkConsoleError, "1 <= dock_n <= raw_n"):
            build_run_command(
                "gen1", Path("results/bad"),
                pilot=True, raw_n=5, dock_n=6, workers=1,
            )


class BenchmarkLaunchTests(unittest.TestCase):
    def test_dry_run_neither_starts_process_nor_creates_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "dry.console.log"
            with mock.patch("src.benchmark.subprocess.Popen") as popen:
                code = launch([sys.executable, "-c", "print('no')"], dry_run=True, log_path=log)
            self.assertEqual(code, 0)
            self.assertFalse(log.exists())
            popen.assert_not_called()

    def test_live_output_is_also_written_to_durable_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "run.console.log"
            captured = io.StringIO()
            with redirect_stdout(captured):
                code = launch(
                    [sys.executable, "-c", "print('pipeline-output')"],
                    dry_run=False,
                    log_path=log,
                )
            self.assertEqual(code, 0)
            self.assertIn("pipeline-output", captured.getvalue())
            payload = log.read_text(encoding="utf-8")
            self.assertIn("pipeline-output", payload)
            self.assertIn("Exit code: 0", payload)


class BenchmarkConfigureTests(unittest.TestCase):
    @staticmethod
    def _answers(values: list[str]):
        iterator = iter(values)
        return lambda _prompt: next(iterator)

    def test_registered_builder_defaults_to_all_and_hides_scientific_knobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            command, outdir, start = configure_run(
                no_start=True,
                input_fn=self._answers(["", "", str(Path(temporary) / "registered"), ""]),
                output_fn=lambda _line: None,
            )
            self.assertFalse(start)
            self.assertEqual(outdir, Path(temporary) / "registered")
            self.assertIn("src.generation.run_all_arms", command)
            self.assertNotIn("--seed", command)
            self.assertNotIn("--raw-n", command)

    def test_pilot_builder_maps_counts_to_existing_validation_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            command, _, start = configure_run(
                no_start=True,
                input_fn=self._answers(
                    ["2", "1", str(Path(temporary) / "pilot"), "3", "25", "5"]
                ),
                output_fn=lambda _line: None,
            )
            self.assertFalse(start)
            self.assertIn("src.generation.run_gen3_pipeline", command)
            self.assertEqual(command[command.index("--raw-n") + 1], "25")
            self.assertEqual(command[command.index("--dock-n") + 1], "5")
            self.assertEqual(command[command.index("--workers") + 1], "3")


class BenchmarkStatusTests(unittest.TestCase):
    def test_pipeline_status_finds_sibling_candidate_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = root / "pipeline"
            analysis = root / "analysis"
            pipeline.mkdir()
            analysis.mkdir()
            report = analysis / "report.html"
            report.write_text("report", encoding="utf-8")
            (pipeline / "pipeline_summary.json").write_text(
                json.dumps(
                    {
                        "name": "baseline_test",
                        "stage": "candidate_pipeline",
                        "funnel": {"submitted": 5, "successfully_scored": 4},
                        "outputs": {"candidate_report": str(report)},
                    }
                ),
                encoding="utf-8",
            )
            captured = io.StringIO()
            with redirect_stdout(captured):
                status(pipeline)
            self.assertIn("State: COMPLETE", captured.getvalue())
            self.assertIn(str(report), captured.getvalue())


if __name__ == "__main__":
    unittest.main()
