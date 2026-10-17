#!/bin/bash
# Full TRAIN targets only, inside a separately guarded serial CPU allocation.
set -euo pipefail
target_root=<cluster-work>/fusion/full-targets-045-20261010
python=<cluster-work>/fusion/nn-008/venv/bin/python
test -n "${SLURM_JOB_ID:-}"
mkdir "$target_root/allocated.lock"
printf '%s\n' "$SLURM_JOB_ID" > "$target_root/job-id.txt"
trap 'status=$?; printf "%s\n" "$status" > "$target_root/step-exit.txt"' EXIT
scontrol update "JobId=$SLURM_JOB_ID" Requeue=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$target_root/repo"
sha256sum --check source-sha256.txt
sha256sum --check "$target_root/input-sha256.txt"
"$python" -u -m research.full_train_targets "$target_root"
