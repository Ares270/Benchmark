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
    launch_sequence,
    status,
    watch_pane_block,
    watch_pane_command,
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
            printed: list[str] = []
            plan = configure_run(
                no_start=True,
                input_fn=self._answers(
                    ["", "", str(Path(temporary) / "registered"), "", ""]
                ),
                output_fn=printed.append,
            )
            self.assertFalse(plan.start)
            self.assertFalse(plan.watch)
            self.assertEqual(plan.outdir, Path(temporary) / "registered")
            self.assertEqual(len(plan.commands), 1)
            command = plan.commands[0]
            self.assertIn("src.generation.run_all_arms", command)
            self.assertNotIn("--seed", command)
            self.assertNotIn("--raw-n", command)

    def test_four_arm_builder_states_that_no_pilot_version_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            printed: list[str] = []
            configure_run(
                no_start=True,
                input_fn=self._answers(
                    ["", "", str(Path(temporary) / "registered"), "", ""]
                ),
                output_fn=printed.append,
            )
            notice = "\n".join(printed)
            self.assertIn("cannot be run as a pilot", notice)
            self.assertIn("20260801", notice)
            self.assertIn("10,000", notice)
            self.assertIn("1,000", notice)

    def test_pilot_builder_maps_counts_to_existing_validation_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = configure_run(
                no_start=True,
                input_fn=self._answers(
                    ["2", "1", str(Path(temporary) / "pilot"), "3", "25", "5", ""]
                ),
                output_fn=lambda _line: None,
            )
            self.assertFalse(plan.start)
            command = plan.commands[0]
            self.assertIn("src.generation.run_gen3_pipeline", command)
            self.assertEqual(command[command.index("--raw-n") + 1], "25")
            self.assertEqual(command[command.index("--dock-n") + 1], "5")
            self.assertEqual(command[command.index("--workers") + 1], "3")

    def test_labelled_builder_chains_cohort_and_run_from_one_name(self):
        printed: list[str] = []
        plan = configure_run(
            no_start=True,
            input_fn=self._answers(
                ["3", "labelled_console_unittest", "5", "", "", "", "", ""]
            ),
            output_fn=printed.append,
        )
        self.assertFalse(plan.start)
        self.assertEqual(plan.outdir, Path("results/labelled_console_unittest"))
        build, dock = plan.commands
        self.assertIn("src.harness.build_cohort", build)
        self.assertEqual(build[build.index("--n-actives") + 1], "5")
        self.assertEqual(build[build.index("--seed") + 1], "42")
        self.assertNotIn("--decoys-per-active", build)
        self.assertIn("src.harness.run_local", dock)
        self.assertEqual(dock[dock.index("--name") + 1], "labelled_console_unittest")
        self.assertEqual(dock[dock.index("--missing-policy") + 1], "error")
        self.assertIn("results/labelled_console_unittest", build)
        self.assertIn("results/labelled_console_unittest", dock)
        review = "\n".join(printed)
        # 5 actives + 5 x 50 assigned decoys; per-active is not a total.
        self.assertIn("Molecules to be docked: 255", review)
        self.assertIn("all assigned decoys (50 per active)", review)

    def test_labelled_builder_counts_requested_decoys_per_active(self):
        printed: list[str] = []
        plan = configure_run(
            no_start=True,
            input_fn=self._answers(
                ["3", "labelled_console_unittest2", "10", "3", "7", "2", "2", ""]
            ),
            output_fn=printed.append,
        )
        build, dock = plan.commands
        self.assertEqual(build[build.index("--decoys-per-active") + 1], "3")
        self.assertEqual(build[build.index("--seed") + 1], "7")
        self.assertEqual(dock[dock.index("--workers") + 1], "2")
        self.assertEqual(dock[dock.index("--missing-policy") + 1], "exclude")
        self.assertIn("Molecules to be docked: 40", "\n".join(printed))

    def test_labelled_builder_refuses_a_name_the_analysis_would_reject(self):
        with self.assertRaisesRegex(BenchmarkConsoleError, "Run name"):
            configure_run(
                no_start=True,
                input_fn=self._answers(["3", "not a valid name!"]),
                output_fn=lambda _line: None,
            )

    def test_builder_offers_the_monitor_pane_as_its_own_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            printed: list[str] = []
            plan = configure_run(
                no_start=True,
                input_fn=self._answers(
                    ["", "2", str(Path(temporary) / "watched"), "", "y"]
                ),
                output_fn=printed.append,
            )
            self.assertTrue(plan.watch)
            self.assertIn("Monitor pane", "\n".join(printed))
            self.assertNotIn("&&", "\n".join(printed).split("Monitor pane")[0])


class BenchmarkWatchPaneTests(unittest.TestCase):
    def test_absent_windows_terminal_falls_back_to_a_pasteable_command(self):
        with mock.patch("src.benchmark.shutil.which", return_value=None):
            argv = watch_pane_command(Path("results/run"))
        self.assertIsNone(argv)
        block = watch_pane_block(Path("results/run"), argv)
        self.assertIn("python -m src.benchmark watch results/run", block[1])

    def test_pane_command_quotes_paths_and_keeps_the_shell_alive(self):
        with mock.patch("src.benchmark.shutil.which", return_value="/mnt/c/wt.exe"):
            argv = watch_pane_command(Path("results/a run"))
        self.assertIsNotNone(argv)
        self.assertEqual(argv[:4], ["/mnt/c/wt.exe", "-w", "0", "split-pane"])
        inner = argv[-1]
        self.assertIn("'results/a run'", inner)
        self.assertTrue(inner.endswith("; exec bash"))


class BenchmarkSequenceTests(unittest.TestCase):
    def test_failed_first_stage_stops_the_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "chain.console.log"
            marker = Path(temporary) / "second-stage-ran"
            captured = io.StringIO()
            with redirect_stdout(captured):
                code = launch_sequence(
                    [
                        [sys.executable, "-c", "raise SystemExit(3)"],
                        [
                            sys.executable,
                            "-c",
                            f"open({str(marker)!r}, 'w').close()",
                        ],
                    ],
                    dry_run=False,
                    log_path=log,
                )
            self.assertEqual(code, 3)
            self.assertFalse(marker.exists())
            self.assertIn("the remaining stages were not started", captured.getvalue())

    def test_dry_run_prints_the_pane_without_opening_it(self):
        captured = io.StringIO()
        with mock.patch("src.benchmark.open_watch_pane") as opener:
            with redirect_stdout(captured):
                code = launch_sequence(
                    [[sys.executable, "-c", "print('no')"]],
                    dry_run=True,
                    watch_dir=Path("results/run"),
                )
        self.assertEqual(code, 0)
        opener.assert_not_called()
        self.assertIn("Monitor pane", captured.getvalue())
        self.assertIn("Dry run only", captured.getvalue())


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
