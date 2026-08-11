#!/usr/bin/env bash
set -Eeuo pipefail
umask 0022

required=(
  BATCH_TASK_INDEX
  BENCH_BUNDLE_PREFIX
  BENCH_RUN_PREFIX
  BENCH_WORKERS
  BENCH_JOB_NAME
  BENCH_IMAGE_URI
)
for variable_name in "${required[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "ERROR: required environment variable is empty: ${variable_name}" >&2
    exit 2
  fi
done
if [[ ! "${BATCH_TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: BATCH_TASK_INDEX must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${BENCH_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BENCH_WORKERS must be a positive integer" >&2
  exit 2
fi

readonly mount_path="/mnt/disks/benchmark"
readonly manifest_path="${mount_path}/${BENCH_BUNDLE_PREFIX}/manifest.json"
readonly run_root="${mount_path}/${BENCH_RUN_PREFIX}"
readonly retry_attempt="${BATCH_TASK_RETRY_ATTEMPT:-0}"

echo "job=${BENCH_JOB_NAME} task=${BATCH_TASK_INDEX} retry=${retry_attempt}"
echo "image=${BENCH_IMAGE_URI}"
echo "manifest=${manifest_path}"
echo "run_root=${run_root}"
python --version
smina --version

if [[ ! -f "${manifest_path}" ]]; then
  echo "ERROR: mounted manifest does not exist: ${manifest_path}" >&2
  exit 2
fi
mkdir -p "${run_root}"
cd /opt/benchmark
python -m src.harness.chunks run \
  "${manifest_path}" \
  "${BATCH_TASK_INDEX}" \
  "${run_root}" \
  --workers "${BENCH_WORKERS}"
