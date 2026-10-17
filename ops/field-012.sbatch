#!/bin/bash
#SBATCH --job-name=fusion-field012
#SBATCH --partition=a100-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/field-012-20261008/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/field-012-20261008/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_root=<cluster-work>/fusion/field-012-20261008
neural_root=<cluster-work>/fusion/nn-008
geometry_root=<cluster-work>/fusion/geometry-010-20261008
python="$neural_root/venv/bin/python"
cd "$task_root/repo"
nvidia-smi -L
test -f "$geometry_root/train-geometry/index.json"
grid_shot=$("$python" -c 'import json, pathlib; root=pathlib.Path("<cluster-work>/fusion/nn-008"); name=json.loads((root/"packed/train/index.json").read_text())["names"][0]; paths=list((root/"raw").rglob(name+".parquet")); assert len(paths)==1, (name,len(paths)); print(paths[0])')
for representation in pca grid; do
  "$python" -u -m participant.field_tune --checkpoint "$task_root/parent-model.pt" \
    --data "$neural_root/packed" --geometry "$geometry_root/train-geometry" \
    --grid-shot "$grid_shot" --representation "$representation" \
    --out "$task_root/$representation-field" --epochs 10 --lr 0.00002 --seed 20261011
  "$python" -u -m participant.fast_score --packed "$neural_root/packed/development" \
    --predictions "$task_root/$representation-field/development.npz" \
    --reference-cache "$neural_root/packed/development-refs.pkl" --workers 12
  "$python" -u -m tests.nn_regression_check "$neural_root/packed/regression" \
    "$neural_root/../full-split.json" "$task_root/$representation-field/model.pt"
done
