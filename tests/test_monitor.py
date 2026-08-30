from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.monitor import (
    MonitorError,
    latest_run,
    render_snapshot,
    snapshot_run,
    watch_run,
)


class MonitorTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_completed_pipeline_snapshot_uses_durable_stage_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            screening = root / "screening_5"
            self._write(screening / "subsample.json", {"selected_rows": 5})
            self._write(
                screening / "intake/summary.json",
                {"counts": {"accepted_for_preparation": 4}},
            )
            self._write(screening / "gate/gate_summary.json", {"n_passed": 3})
            self._write(
                screening / "prepared/_prep_summary.json",
                {"counts": {"successful_pdbqt_available": 3}},
            )
            self._write(
                screening / "docking/_dock_summary.json",
                {"counts": {"ok": 2, "cached": 1}},
            )
            self._write(
                screening / "pipeline_summary.json",
                {"funnel": {"submitted": 5, "successfully_scored": 3}},
            )

            snapshot = snapshot_run(root)
            self.assertTrue(snapshot["complete"])
            self.assertEqual(snapshot["arms"][0]["stages"][1]["current"], 4)
            self.assertEqual(snapshot["arms"][0]["stages"][4]["current"], 3)
            rendered = render_snapshot(snapshot)
            self.assertIn("COMPLETE", rendered)
            self.assertIn("3 scored", rendered)
            self.assertIn("does not stop the benchmark run", rendered)

    def test_four_arm_monitor_spins_only_on_first_incomplete_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "four"
            (root / "naive_property_matched").mkdir(parents=True)
            (root / "gen1_guacamol").mkdir()
            snapshot = snapshot_run(root)
            rendered = render_snapshot(snapshot, frame=1)
            arm_lines = [line for line in rendered.splitlines() if "Baseline" in line or "Gen1" in line]
            self.assertTrue(arm_lines[0].lstrip().startswith("/"))
            self.assertTrue(arm_lines[1].lstrip().startswith("·"))


class CohortMonitorTests(unittest.TestCase):
    """The labelled layout: build_cohort writes the cohort, run_local docks it."""

    @staticmethod
    def _cohort(root: Path, *, actives: int, decoys: int, selection: bool = True) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "actives.smi").write_text(
            "".join(f"C{index} A{index}\n" for index in range(actives)),
            encoding="utf-8",
        )
        (root / "decoys.smi").write_text(
            "".join(f"C{index} D{index}\n" for index in range(decoys)),
            encoding="utf-8",
        )
        if selection:
            (root / "cohort.json").write_text(
                json.dumps({"selection": {"n_actives": actives, "n_decoys": decoys}}),
                encoding="utf-8",
            )
        return root

    @staticmethod
    def _prepared(root: Path, cohort: str, count: int, *, summary: bool) -> None:
        directory = root / "ligands" / cohort
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{cohort[0].upper()}{index}.pdbqt").touch()
        if summary:
            (directory / "_prep_summary.json").write_text(
                json.dumps({"counts": {"successful_pdbqt_available": count}}),
                encoding="utf-8",
            )

    @staticmethod
    def _docked(root: Path, cohort: str, count: int, *, summary: bool) -> None:
        directory = root / "docking" / cohort
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{cohort[0].upper()}{index}_out.pdbqt").touch()
        if summary:
            (directory / "_dock_summary.json").write_text(
                json.dumps({"counts": {"ok": count, "cached": 0}}),
                encoding="utf-8",
            )

    @staticmethod
    def _stages(snapshot: dict) -> dict[str, dict]:
        return {stage["label"]: stage for stage in snapshot["arms"][0]["stages"]}

    def test_cohort_generative_and_campaign_layouts_are_told_apart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = self._cohort(root / "labelled", actives=2, decoys=4)
            generative = root / "generative"
            (generative / "screening_10" / "intake").mkdir(parents=True)
            campaign = root / "campaign"
            (campaign / "gen1_guacamol").mkdir(parents=True)
            self.assertEqual(snapshot_run(cohort)["mode"], "cohort")
            self.assertEqual(snapshot_run(generative)["mode"], "single")
            self.assertEqual(snapshot_run(campaign)["mode"], "all")

    def test_cohort_directory_without_cohort_json_is_still_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "legacy", actives=3, decoys=9,
                                selection=False)
            snapshot = snapshot_run(root)
            self.assertEqual(snapshot["mode"], "cohort")
            # Totals fall back to counting the .smi files that were docked.
            self.assertEqual(self._stages(snapshot)["Prep actives"]["total"], 3)
            self.assertEqual(self._stages(snapshot)["Prep decoys"]["total"], 9)

    def test_partial_run_marks_one_stage_current_and_the_rest_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "live", actives=10, decoys=300)
            self._prepared(root, "actives", 10, summary=True)
            self._docked(root, "actives", 10, summary=True)
            self._prepared(root, "decoys", 300, summary=True)
            self._docked(root, "decoys", 12, summary=False)
            snapshot = snapshot_run(root)
            stages = self._stages(snapshot)
            self.assertFalse(snapshot["complete"])
            self.assertEqual(stages["Cohort"]["state"], "complete")
            self.assertEqual(stages["Dock actives"]["state"], "complete")
            self.assertEqual(stages["Dock decoys"]["state"], "current")
            self.assertEqual(stages["Analyze"]["state"], "pending")
            # The decoy stage is the point: prepared 300/300, docked 12/300.
            self.assertEqual(
                (stages["Prep decoys"]["current"], stages["Prep decoys"]["total"]),
                (300, 300),
            )
            self.assertEqual(
                (stages["Dock decoys"]["current"], stages["Dock decoys"]["total"]),
                (12, 300),
            )
            rendered = render_snapshot(snapshot)
            self.assertIn("LABELLED RUN", rendered)
            self.assertIn("12 scored", rendered)

    def test_live_files_carry_progress_before_a_summary_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "live", actives=8, decoys=16)
            self._prepared(root, "actives", 5, summary=False)
            stages = self._stages(snapshot_run(root))
            self.assertEqual(stages["Prep actives"]["state"], "current")
            self.assertEqual(stages["Prep actives"]["current"], 5)
            self.assertEqual(stages["Prep actives"]["total"], 8)
            self.assertEqual(stages["Prep actives"]["detail"], "5 PDBQT")

    def test_totals_come_from_cohort_json_before_the_smi_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "live", actives=4, decoys=8)
            (root / "cohort.json").write_text(
                json.dumps({"selection": {"n_actives": 4, "n_decoys": 200}}),
                encoding="utf-8",
            )
            stages = self._stages(snapshot_run(root))
            self.assertEqual(stages["Prep decoys"]["total"], 200)
            self.assertEqual(stages["Cohort"]["detail"], "4 actives + 200 decoys")

    def test_malformed_cohort_json_degrades_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "live", actives=2, decoys=6)
            (root / "cohort.json").write_text("{not json", encoding="utf-8")
            stages = self._stages(snapshot_run(root))
            self.assertEqual(stages["Prep actives"]["total"], 2)
            self.assertEqual(stages["Prep decoys"]["total"], 6)

            (root / "cohort.json").write_text("{}", encoding="utf-8")
            stages = self._stages(snapshot_run(root))
            self.assertEqual(stages["Prep decoys"]["total"], 6)

    def test_empty_cohort_directory_shows_every_stage_as_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "starting"
            root.mkdir()
            (root / "cohort.json").write_text("{}", encoding="utf-8")
            stages = self._stages(snapshot_run(root))
            # No totals are known, so no bar may claim a fraction.
            self.assertEqual(stages["Prep actives"]["total"], 0)
            self.assertEqual(stages["Dock decoys"]["total"], 0)
            self.assertIn("·", render_snapshot(snapshot_run(root)))

    def test_legacy_lig_and_dock_directories_are_tolerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "legacy", actives=5, decoys=250,
                                selection=False)
            (root / "lig_actives").mkdir()
            for index in range(5):
                (root / "lig_actives" / f"A{index}.pdbqt").touch()
            (root / "dock_actives").mkdir()
            for index in range(3):
                (root / "dock_actives" / f"A{index}_out.pdbqt").touch()
            stages = self._stages(snapshot_run(root))
            self.assertEqual(stages["Prep actives"]["current"], 5)
            self.assertEqual(stages["Dock actives"]["current"], 3)

    def test_completed_cohort_run_reports_exit_state_from_the_console_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            root = self._cohort(results / "labelled", actives=2, decoys=4)
            (root / "run_local.json").write_text(
                json.dumps({"stage": "run_local", "name": "labelled"}),
                encoding="utf-8",
            )
            (results / "labelled.console.log").write_text(
                "Started: 2026-08-29T10:00:00+00:00\nDone\nExit code: 0\n",
                encoding="utf-8",
            )
            snapshot = snapshot_run(root)
            self.assertTrue(snapshot["complete"])
            self.assertEqual(snapshot["exit_code"], 0)
            self.assertFalse(snapshot["failed"])
            self.assertEqual(snapshot["latest_log"], "Exit code: 0")
            self.assertEqual(watch_run(root, once=True), 0)
            self.assertEqual(latest_run(results), root)

    def test_failed_cohort_run_still_exits_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            root = self._cohort(results / "labelled", actives=2, decoys=4)
            (results / "labelled.console.log").write_text(
                "Started: 2026-08-29T10:00:00+00:00\nboom\nExit code: 2\n",
                encoding="utf-8",
            )
            self.assertEqual(watch_run(root, once=True), 1)

    def test_watching_never_writes_to_the_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cohort(Path(temporary) / "live", actives=4, decoys=8)
            self._prepared(root, "actives", 2, summary=False)

            def listing() -> set[tuple[str, float]]:
                return {
                    (str(path.relative_to(root)), path.stat().st_mtime)
                    for path in root.rglob("*")
                }

            before = listing()
            render_snapshot(snapshot_run(root))
            watch_run(root, once=True)
            self.assertEqual(listing(), before)


class LatestRunTests(unittest.TestCase):
    @staticmethod
    def _touch(path: Path, when: float) -> None:
        """Age a run directory and everything the selector looks at inside it."""

        for entry in path.iterdir():
            os.utime(entry, (when, when))
        os.utime(path, (when, when))

    def test_newest_finished_run_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            older = results / "older"
            newer = results / "newer"
            for directory in (older, newer):
                directory.mkdir()
                (directory / "pipeline_summary.json").write_text("{}", encoding="utf-8")
            self._touch(older, 1_000.0)
            self._touch(newer, 2_000.0)
            self.assertEqual(latest_run(results), newer)

    def test_in_progress_partial_directory_is_selectable(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            finished = results / "finished"
            finished.mkdir()
            (finished / "metrics.json").write_text("{}", encoding="utf-8")
            partial = results / ".fresh_20260829_120000Z.partial"
            partial.mkdir()
            self._touch(finished, 1_000.0)
            self._touch(partial, 3_000.0)
            self.assertEqual(latest_run(results), partial)

    def test_directory_without_run_artifacts_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            (results / "notes").mkdir()
            (results / "benchmark_home.html").write_text("<p>", encoding="utf-8")
            with self.assertRaisesRegex(MonitorError, "No run directories"):
                latest_run(results)


if __name__ == "__main__":
    unittest.main()
