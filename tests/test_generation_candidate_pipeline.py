from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generation.run_candidate_pipeline import (
    CandidatePipelineError,
    create_submission,
)


class CandidatePipelineSubsampleTests(unittest.TestCase):
    def _source(self, root: Path, n: int = 20) -> Path:
        path = root / "source.smi"
        path.write_text(
            "".join(f"{'C' * (index + 1)} MOL_{index:03d}\n" for index in range(n)),
            encoding="utf-8",
        )
        return path

    def test_same_seed_writes_identical_raw_subsample_and_id_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            first_path, first = create_submission(
                source, root / "first", n=7, seed=20260801
            )
            second_path, second = create_submission(
                source, root / "second", n=7, seed=20260801
            )

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                first["outputs"]["subsample_ids"]["sha256"],
                second["outputs"]["subsample_ids"]["sha256"],
            )
            self.assertEqual(first["selected_rows"], 7)
            self.assertEqual(len(first_path.read_text().splitlines()), 7)

    def test_different_seed_changes_the_subsample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            first_path, _ = create_submission(
                source, root / "first", n=7, seed=20260801
            )
            second_path, _ = create_submission(
                source, root / "second", n=7, seed=20260802
            )
            self.assertNotEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_request_larger_than_source_refuses_before_creating_workdir(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, n=3)
            workdir = root / "work"
            with self.assertRaises(CandidatePipelineError):
                create_submission(source, workdir, n=4, seed=1)
            self.assertFalse(workdir.exists())


if __name__ == "__main__":
    unittest.main()
