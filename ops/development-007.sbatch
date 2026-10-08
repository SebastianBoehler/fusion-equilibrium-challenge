#!/bin/bash
#SBATCH --job-name=fusion-dev007-20261008
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=<cluster-work>/fusion/development-007-20261008/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/development-007-20261008/logs/%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1
root=<cluster-work>/fusion/development-007-20261008
previous=<cluster-work>/fusion/development-006-20261008
pilot=<cluster-work>/fusion/breadth-004r1-20261008
python=<cluster-work>/fusion/development-005r1-20261008/venv/bin/python
cd "$root/repo"
"$python" -c 'import resource; print({"worker_file_limits": resource.getrlimit(resource.RLIMIT_NOFILE)})'
"$python" -u -m participant.development_compare \
  --manifest "$pilot/data.json" --pilot "$pilot/result" \
  --raw "$root/raw" --cache "$root/cache" --out "$root/result" \
  --resume-from "$previous"
