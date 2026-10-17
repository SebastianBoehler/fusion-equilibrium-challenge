#!/bin/bash
# Two new critical-level arms, serially, after successful CPU validation.
#SBATCH --job-name=fusion-critical031
#SBATCH --partition=a100-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/critical-031-v2-20261009/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/critical-031-v2-20261009/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_root=<cluster-work>/fusion/critical-031-v2-20261009
neural_root=<cluster-work>/fusion/nn-008
contour_root=<cluster-work>/fusion/contour-024-20261009
python="$neural_root/venv/bin/python"
cd "$task_root/repo"
sha256sum --check source-sha256.txt
sha256sum --check "$task_root/input-sha256.txt"
test -s "$task_root/critical-validation.json"
nvidia-smi -L
for arm in critical-kappa critical-li-kappa; do
  "$python" -u -m participant.critical_tune --checkpoint "$task_root/parent-model.pt" \
    --data "$neural_root/packed" --contours "$contour_root/train-contours" \
    --critical "$task_root/train-critical" --validation "$task_root/critical-validation.json" \
    --out "$task_root/$arm" --arm "$arm" --seed 20261031
  "$python" -u -m participant.fast_score --packed "$neural_root/packed/development" \
    --predictions "$task_root/$arm/development.npz" \
    --reference-cache "$neural_root/packed/development-refs.pkl" --workers 12
  "$python" -u -m participant.full_regression_score --packed "$neural_root/packed/regression" \
    --split "$task_root/full-split.json" --checkpoint "$task_root/$arm/model.pt" \
    --out "$task_root/$arm/regression.scores.json" --workers 12
done
