"""Generate immutable Google Cloud Batch specs for verified docking bundles.

This module does not submit jobs or mutate cloud state. It converts one already
verified :mod:`src.harness.chunks` manifest into a Batch JSON document whose
container is pinned by digest and whose task indices map one-to-one to chunk
indices.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

from src.harness import runtime
from src.harness.chunks import ChunkManifestError, verify_manifest


DEFAULT_MACHINE_TYPE = "c3d-highcpu-90"
DEFAULT_PARALLELISM = 11                        # 11 parallel tasks
DEFAULT_WORKERS = 8                             # 8 vCPUs per task, 8 GiB RAM per task
DEFAULT_CPU_MILLI = 8_000   
DEFAULT_MEMORY_MIB = 8_192
DEFAULT_MAX_RETRY_COUNT = 2
DEFAULT_MAX_RUN_DURATION_S = 21_600
DEFAULT_REGION = "us-central1"
GCS_MOUNT_PATH = "/mnt/disks/benchmark"
CONTAINER_ENTRYPOINT = "/usr/local/bin/_entrypoint.sh"
CONTAINER_COMMAND = "/opt/benchmark/cloud/google_batch/run_chunk.sh"

_IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_JOB_NAME = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SERVICE_ACCOUNT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9.-]+[.]gserviceaccount[.]com$"
)


class GoogleBatchSpecError(ValueError):
    """Raised when a cloud job would be ambiguous or irreproducible."""


def _normalized_gcs_prefix(value: str, label: str) -> str:
    value = str(value).strip()
    if (
        not value
        or value.startswith("gs://")
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise GoogleBatchSpecError(
            f"{label} must be a bucket-relative prefix without gs://"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise GoogleBatchSpecError(f"{label} is not a safe relative GCS prefix")
    normalized = str(path)
    if normalized != value:
        raise GoogleBatchSpecError(
            f"{label} must already be normalized: {value!r} != {normalized!r}"
        )
    return normalized


def _manifest_memory_mib(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([GM])", str(value).strip().upper())
    if match is None:
        raise GoogleBatchSpecError(
            f"Unsupported manifest memory format {value!r}; expected e.g. 8G"
        )
    amount = int(match.group(1))
    return amount * (1024 if match.group(2) == "G" else 1)






def _load_verified_manifest(path: Path) -> dict:
    path = Path(path)
    try:
        verify_manifest(path)                   # re-hash and validate the manifest
    except ChunkManifestError as error:
        raise GoogleBatchSpecError(str(error)) from error
    return json.loads(path.read_text(encoding="utf-8"))








# loada - validate - validate some more - assemble a dict #

def build_job_spec(
    manifest_path: Path,
    *,
    image_uri: str,
    bucket: str,
    bundle_prefix: str,
    run_prefix: str,
    service_account: str,
    job_name: str,
    region: str = DEFAULT_REGION,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    parallelism: int = DEFAULT_PARALLELISM,
    workers: int = DEFAULT_WORKERS,
    cpu_milli: int = DEFAULT_CPU_MILLI,
    memory_mib: int = DEFAULT_MEMORY_MIB,
    max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
    max_run_duration_s: int = DEFAULT_MAX_RUN_DURATION_S,
    task_count: int | None = None,
) -> dict:
    """Return one Google Batch job document without submitting it."""

    manifest = _load_verified_manifest(manifest_path)
    total_chunks = int(manifest["counts"]["chunks"])
    declared_workers = int(manifest["parameters"]["cpus_per_task"])
    declared_memory = _manifest_memory_mib(manifest["parameters"]["memory"])

    if not _IMAGE_DIGEST.fullmatch(str(image_uri)):
        raise GoogleBatchSpecError(
            "image_uri must use an immutable @sha256:<64 lowercase hex> digest"
        )
    if not _BUCKET_NAME.fullmatch(str(bucket)):
        raise GoogleBatchSpecError(f"Invalid Cloud Storage bucket name: {bucket!r}")
    if not _SERVICE_ACCOUNT.fullmatch(str(service_account)):
        raise GoogleBatchSpecError(
            f"Invalid Google service-account email: {service_account!r}"
        )
    if not _JOB_NAME.fullmatch(str(job_name)):
        raise GoogleBatchSpecError(
            "job_name must be 1-63 lowercase letters, numbers, or hyphens, "
            "start with a letter, and end with a letter or number"
        )
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]", str(region)):
        raise GoogleBatchSpecError(f"Invalid Google Cloud region: {region!r}")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,62}", str(machine_type)):
        raise GoogleBatchSpecError(f"Invalid machine type: {machine_type!r}")

    bundle_prefix = _normalized_gcs_prefix(bundle_prefix, "bundle_prefix")
    run_prefix = _normalized_gcs_prefix(run_prefix, "run_prefix")
    if bundle_prefix == run_prefix or run_prefix.startswith(bundle_prefix + "/"):
        raise GoogleBatchSpecError(
            "run_prefix must not overwrite or nest inside the immutable bundle"
        )

    if workers != declared_workers:
        raise GoogleBatchSpecError(
            f"workers={workers} differs from manifest cpus_per_task="
            f"{declared_workers}"
        )
    if cpu_milli != workers * 1000:
        raise GoogleBatchSpecError(
            f"cpu_milli must equal workers * 1000 ({workers * 1000})"
        )
    if memory_mib < declared_memory:
        raise GoogleBatchSpecError(
            f"memory_mib={memory_mib} is below manifest request {declared_memory}"
        )
    if max_retry_count < 0 or max_retry_count > 10:
        raise GoogleBatchSpecError("max_retry_count must be between 0 and 10")
    if max_run_duration_s < 3600:
        raise GoogleBatchSpecError("max_run_duration_s must be at least 3600")

    resolved_task_count = total_chunks if task_count is None else int(task_count)
    if resolved_task_count < 1 or resolved_task_count > total_chunks:
        raise GoogleBatchSpecError(
            f"task_count must be within 1-{total_chunks}, got "
            f"{resolved_task_count}"
        )
    if parallelism < 1 or parallelism > resolved_task_count:
        raise GoogleBatchSpecError(
            f"parallelism must be within 1-{resolved_task_count}, got "
            f"{parallelism}"
        )

    variables = {
        "BENCH_BUNDLE_PREFIX": bundle_prefix,
        "BENCH_RUN_PREFIX": run_prefix,
        "BENCH_WORKERS": str(workers),
        "BENCH_JOB_NAME": job_name,
        "BENCH_REGION": region,
        "BENCH_IMAGE_URI": image_uri,
    }
    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": image_uri,
                                "entrypoint": CONTAINER_ENTRYPOINT,
                                "commands": [
                                    "/bin/bash",
                                    CONTAINER_COMMAND,
                                ],
                            },
                            "environment": {"variables": variables},
                        }
                    ],
                    "computeResource": {
                        "cpuMilli": cpu_milli,
                        "memoryMib": memory_mib,
                    },
                    "maxRetryCount": max_retry_count,
                    "maxRunDuration": f"{max_run_duration_s}s",
                    "volumes": [
                        {
                            "gcs": {"remotePath": bucket},
                            "mountPath": GCS_MOUNT_PATH,
                        }
                    ],
                },
                "taskCount": resolved_task_count,
                "parallelism": parallelism,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "policy": {
                        "machineType": machine_type,
                        "provisioningModel": "SPOT",
                        "reservation": "NO_RESERVATION",
                    }
                }
            ],
            "serviceAccount": {"email": service_account},
        },
        "labels": {
            "project": "dyrk1a",
            "protocol": "validation-v1",
            "cohort": job_name[:63],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }





# Refuses to overwrite an existing file and writes atomically #

def write_job_spec(path: Path, spec: dict) -> dict:
    """Write a Batch spec atomically and refuse to replace an existing file."""

    path = Path(path)
    if path.exists():
        raise GoogleBatchSpecError(f"Output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_json_atomic(path, spec)
    return runtime.file_record(path)
















def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a digest-pinned Google Cloud Batch job for a verified "
            "portable docking manifest; does not submit or spend money"
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--bundle-prefix", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cpu-milli", type=int, default=DEFAULT_CPU_MILLI)
    parser.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    parser.add_argument(
        "--max-retry-count", type=int, default=DEFAULT_MAX_RETRY_COUNT
    )
    parser.add_argument(
        "--max-run-duration-s", type=int, default=DEFAULT_MAX_RUN_DURATION_S
    )
    parser.add_argument(
        "--task-count",
        type=int,
        help="override for a smoke run; cannot exceed manifest chunk count",
    )
    args = parser.parse_args()

    try:
        spec = build_job_spec(
            args.manifest,
            image_uri=args.image,
            bucket=args.bucket,
            bundle_prefix=args.bundle_prefix,
            run_prefix=args.run_prefix,
            service_account=args.service_account,
            job_name=args.job_name,
            region=args.region,
            machine_type=args.machine_type,
            parallelism=args.parallelism,
            workers=args.workers,
            cpu_milli=args.cpu_milli,
            memory_mib=args.memory_mib,
            max_retry_count=args.max_retry_count,
            max_run_duration_s=args.max_run_duration_s,
            task_count=args.task_count,
        )
        record = write_job_spec(args.out, spec)
    except GoogleBatchSpecError as error:
        parser.error(str(error))

    print(f"Wrote {args.out} ({record['sha256']})")
    print(
        f"Review, then submit: gcloud batch jobs submit {args.job_name} "
        f"--location {args.region} --config {args.out}"
    )


if __name__ == "__main__":
    main()
