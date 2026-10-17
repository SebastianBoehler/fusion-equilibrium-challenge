#!/bin/bash
#SBATCH --job-name=fusion-geometry010
#SBATCH --partition=a100-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/geometry-010-20261008/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/geometry-010-20261008/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_root=<cluster-work>/fusion/geometry-010-20261008
neural_root=<cluster-work>/fusion/nn-008
python="$neural_root/venv/bin/python"
cd "$task_root/repo"
nvidia-smi -L
"$python" -u -m participant.geometry_targets --raw "$neural_root/raw" \
  --packed "$neural_root/packed/train" --out "$task_root/train-geometry" --workers 12
"$python" -u -m participant.geometry_tune --checkpoint "$task_root/parent-model.pt" \
  --data "$neural_root/packed" --geometry "$task_root/train-geometry" \
  --out "$task_root/boundary-axis-010" --epochs 10 --lr 0.00002 \
  --boundary-weight 0.1 --axis-weight 0.1 --seed 20261009
"$python" -u -m participant.fast_score --packed "$neural_root/packed/development" \
  --predictions "$task_root/boundary-axis-010/development.npz" \
  --reference-cache "$neural_root/packed/development-refs.pkl" --workers 12
"$python" -u -m tests.nn_regression_check "$neural_root/packed/regression" \
  "$neural_root/../full-split.json" "$task_root/boundary-axis-010/model.pt"
