#!/bin/bash
#SBATCH --job-name=fusion-dev005r1-20261008
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=<cluster-work>/fusion/development-005r1-20261008/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/development-005r1-20261008/logs/%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
root=<cluster-work>/fusion/development-005r1-20261008
pilot=<cluster-work>/fusion/breadth-004r1-20261008
base_python=<cluster-work>/learn2design/exp123-gentle-refinement-20260916-235735-fecfe302/venv/bin/python
"$base_python" -m venv "$root/venv"
"$root/venv/bin/python" -m pip install --disable-pip-version-check -r "$root/repo/ops/cpu-requirements.txt"
cd "$root/repo"
"$root/venv/bin/python" -u -m participant.development_compare \
  --manifest "$pilot/data.json" --pilot "$pilot/result" \
  --raw "$root/raw" --cache "$root/cache" --out "$root/result"
