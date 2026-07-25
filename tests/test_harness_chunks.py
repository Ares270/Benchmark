from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.harness import runtime
from src.harness.chunks import (
    ChunkManifestError,
    create_manifest,
    merge_chunks,
    verify_manifest,
)


class HarnessChunkTests(unittest.TestCase):
    def _source(self, root: Path, count: int = 5) -> Path:
        path = root / "accepted.smi"
        path.write_text(
            "".join(f"{'C' * index} mol{index}\n" for index in range(1, count + 1)),
            encoding="utf-8",
        )
        return path

    def test_create_is_self_contained_deterministic_and_slurm_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            manifest = create_manifest(
                self._source(root),
                bundle,
                chunk_size=2,
                cpus_per_task=6,
            )

            self.assertEqual(manifest["counts"], {"molecules": 5, "chunks": 3})
            self.assertEqual(
                [record["n_molecules"] for record in manifest["chunks"]],
                [2, 2, 1],
            )
            self.assertTrue((bundle / "source.smi").is_file())
            script = (bundle / "submit_slurm_array.sh").read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-2", script)
            self.assertIn("run " + "\\" + "\n", script)
            self.assertIn("#SBATCH --cpus-per-task=6", script)
            self.assertIn("BENCH_REPO_ROOT", script)
            verified = verify_manifest(bundle / "manifest.json")
            self.assertEqual(verified["status"], "pass")

    def test_verify_detects_tampered_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            create_manifest(self._source(root), bundle, chunk_size=2)
            chunk = bundle / "chunks" / "chunk_0000.smi"
            chunk.write_text(chunk.read_text(encoding="utf-8") + "C extra\n")

            with self.assertRaisesRegex(ChunkManifestError, "hash mismatch"):
                verify_manifest(bundle / "manifest.json")

    def test_merge_restores_source_order_and_makes_missing_rows_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            create_manifest(self._source(root, count=3), bundle, chunk_size=2)
            run_root = root / "run"

            chunk_rows = (
                [
                    {
                        "molecule_id": "mol1",
                        "score_kcal_mol": "-7.1000",
                        "status": "ok",
                        "reason": "",
                    }
                ],
                [
                    {
                        "molecule_id": "mol3",
                        "score_kcal_mol": "",
                        "status": "dock_failed",
                        "reason": "fixture",
                    }
                ],
            )
            for index, rows in enumerate(chunk_rows):
                chunk_root = run_root / f"chunk_{index:04d}"
                chunk_root.mkdir(parents=True)
                scores_path = chunk_root / "scores.csv"
                with scores_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "molecule_id",
                            "score_kcal_mol",
                            "status",
                            "reason",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                runtime.write_json_atomic(
                    chunk_root / "chunk_result.json",
                    {
                        "schema_version": 1,
                        "stage": "portable_docking_chunk_result",
                        "status": "complete",
                        "chunk_index": index,
                        "outputs": {"scores_csv": runtime.file_record(scores_path)},
                        "manifest": runtime.file_record(bundle / "manifest.json"),
                        "input_chunk": runtime.file_record(
                            bundle
                            / "chunks"
                            / f"chunk_{index:04d}.smi"
                        ),
                        "timing": {
                            "wall_seconds": 10.0 * (index + 1),
                            "estimated_requested_cpu_slot_hours": 0.5,
                        },
                    },
                )

            outdir = root / "merged"
            merge = merge_chunks(bundle / "manifest.json", run_root, outdir)
            scores = pd.read_csv(outdir / "scores.csv", keep_default_na=False)

            self.assertEqual(scores["molecule_id"].tolist(), ["mol1", "mol2", "mol3"])
            self.assertEqual(scores.loc[1, "status"], "missing_result")
            self.assertEqual(merge["counts"], {"total": 3, "ok": 1, "cached": 0, "failed": 2})
            summary = json.loads(
                (outdir / "_dock_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["stage"], "smina_docking")
            self.assertEqual(summary["timing"]["sum_chunk_wall_seconds"], 30.0)
            self.assertEqual(summary["timing"]["max_chunk_wall_seconds"], 20.0)
            self.assertEqual(
                summary["outputs"]["scores_csv"]["sha256"],
                runtime.sha256_file(outdir / "scores.csv"),
            )


if __name__ == "__main__":
    unittest.main()
