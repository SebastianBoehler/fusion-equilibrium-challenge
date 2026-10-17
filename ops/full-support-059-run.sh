#!/bin/bash
set -euo pipefail
support_root=<cluster-work>/fusion/full-support-059-20261011
python=<cluster-work>/fusion/nn-008/venv/bin/python
test -n "${SLURM_JOB_ID:-}"
mkdir "$support_root/allocated.lock"
printf '%s\n' "$SLURM_JOB_ID" > "$support_root/job-id.txt"
trap 'status=$?; printf "%s\n" "$status" > "$support_root/step-exit.txt"' EXIT
scontrol update "JobId=$SLURM_JOB_ID" Requeue=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$support_root/repo"
sha256sum --check source-sha256.txt
"$python" -u -m research.full_constraint_support "$support_root"
