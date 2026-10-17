#!/bin/bash
# Prepared proposal only: do not submit without explicit human approval.
#SBATCH --job-name=fusion-li020
#SBATCH --partition=a100-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --dependency=afterany:2950589
#SBATCH --output=<cluster-work>/fusion/inductance-020-20261009/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/inductance-020-20261009/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_root=<cluster-work>/fusion/inductance-020-20261009
neural_root=<cluster-work>/fusion/nn-008
python="$neural_root/venv/bin/python"
cd "$task_root/repo"
sha256sum --check source-sha256.txt
nvidia-smi -L
"$python" -u -m participant.inductance_targets --raw "$neural_root/raw" \
  --packed "$neural_root/packed/train" --split "$task_root/full-split.json" \
  --out "$task_root/train-inductance" --workers 12
for arm in pixel pixel-li; do
  "$python" -u -m participant.inductance_tune --checkpoint "$task_root/parent-model.pt" \
    --data "$neural_root/packed" --targets "$task_root/train-inductance" \
    --out "$task_root/$arm" --arm "$arm" --seed 20261021
  "$python" -u -m participant.fast_score --packed "$neural_root/packed/development" \
    --predictions "$task_root/$arm/development.npz" \
    --reference-cache "$neural_root/packed/development-refs.pkl" --workers 12
  "$python" -u -m tests.nn_regression_check "$neural_root/packed/regression" \
    "$task_root/full-split.json" "$task_root/$arm/model.pt"
done
