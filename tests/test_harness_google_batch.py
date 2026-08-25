from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.harness.chunks import create_manifest
from src.harness.google_batch import (
    GoogleBatchSpecError,
    build_job_spec,
    write_job_spec,
)


IMAGE = (
    "us-central1-docker.pkg.dev/example-project/dyrk1a/harness"
    "@sha256:" + "a" * 64
)
SERVICE_ACCOUNT = "dyrk1a-batch@example-project.iam.gserviceaccount.com"


class GoogleBatchSpecTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        source = root / "source.smi"
        source.write_text(
            "".join(f"{'C' * index} mol{index}\n" for index in range(1, 6)),
            encoding="utf-8",
        )
        bundle = root / "bundle"
        create_manifest(
            source,
            bundle,
            chunk_size=2,
            cpus_per_task=8,
            memory="8G",
        )
        return bundle / "manifest.json"

    def _build(self, manifest: Path, **overrides):
        arguments = {
            "image_uri": IMAGE,
            "bucket": "example-project-dyrk1a-validation",
            "bundle_prefix": "bundles/decoys",
            "run_prefix": "runs/validation-v1/decoys",
            "service_account": SERVICE_ACCOUNT,
            "job_name": "dyrk1a-decoys-v1",
            "parallelism": 3,
        }
        arguments.update(overrides)
        return build_job_spec(manifest, **arguments)

    def test_spec_maps_verified_chunks_to_digest_pinned_spot_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._build(self._manifest(Path(temporary)))

        group = spec["taskGroups"][0]
        task = group["taskSpec"]
        runnable = task["runnables"][0]
        policy = spec["allocationPolicy"]["instances"][0]["policy"]
        self.assertEqual(group["taskCount"], 3)
        self.assertEqual(group["parallelism"], 3)
        self.assertEqual(task["computeResource"], {
            "cpuMilli": 8000,
            "memoryMib": 8192,
        })
        self.assertEqual(task["maxRetryCount"], 2)
        self.assertEqual(task["maxRunDuration"], "21600s")
        self.assertEqual(runnable["container"]["imageUri"], IMAGE)
        self.assertEqual(
            runnable["environment"]["variables"]["BENCH_BUNDLE_PREFIX"],
            "bundles/decoys",
        )
        self.assertEqual(policy["machineType"], "c3d-highcpu-90")
        self.assertEqual(policy["provisioningModel"], "SPOT")
        self.assertEqual(
            spec["allocationPolicy"]["serviceAccount"]["email"],
            SERVICE_ACCOUNT,
        )

    def test_smoke_override_can_only_select_a_manifest_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary))
            smoke = self._build(
                manifest,
                task_count=1,
                parallelism=1,
                job_name="dyrk1a-decoys-smoke",
            )
            self.assertEqual(smoke["taskGroups"][0]["taskCount"], 1)
            with self.assertRaisesRegex(GoogleBatchSpecError, "within 1-3"):
                self._build(manifest, task_count=4)

    def test_rejects_mutable_image_and_resource_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary))
            with self.assertRaisesRegex(GoogleBatchSpecError, "immutable"):
                self._build(
                    manifest,
                    image_uri=(
                        "us-central1-docker.pkg.dev/example/repo/harness:latest"
                    ),
                )
            with self.assertRaisesRegex(
                GoogleBatchSpecError, "differs from manifest"
            ):
                self._build(manifest, workers=4, cpu_milli=4000)
            with self.assertRaisesRegex(
                GoogleBatchSpecError, "bucket-relative prefix"
            ):
                self._build(manifest, run_prefix="/runs/validation-v1/decoys")

    def test_writer_refuses_to_replace_a_reviewed_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            out = root / "job.json"
            write_job_spec(out, self._build(manifest))
            with self.assertRaisesRegex(GoogleBatchSpecError, "already exists"):
                write_job_spec(out, self._build(manifest))


if __name__ == "__main__":
    unittest.main()
