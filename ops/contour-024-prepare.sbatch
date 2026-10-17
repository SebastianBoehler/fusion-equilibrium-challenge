#!/bin/bash
# Human-authorized repair follow-up; CPU preparation allocates no GPU.
#SBATCH --job-name=fusion-contour024-prep
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/contour-024-20261009/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/contour-024-20261009/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_root=<cluster-work>/fusion/contour-024-20261009
neural_root=<cluster-work>/fusion/nn-008
python="$neural_root/venv/bin/python"
cd "$task_root/repo"
sha256sum --check source-sha256.txt
sha256sum --check "$task_root/input-sha256.txt"
"$python" -u -m participant.scored_contour_targets --raw "$neural_root/raw" \
  --packed "$neural_root/packed/train" --split "$task_root/full-split.json" \
  --out "$task_root/train-contours" --workers 12
"$python" -u -m participant.validate_contour_pack --targets "$task_root/train-contours" \
  --packed "$neural_root/packed/train" --split "$task_root/full-split.json" \
  --out "$task_root/target-validation.json"
