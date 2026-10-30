#!/bin/bash
# Site-specific: set --account, --partition and --qos for your cluster on the sbatch command line.
#SBATCH --job-name=fusion-nn-train
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
# Train one DIII-D ensemble member per array task from repro/nn_members.json (0-10).
# Usage (from the repo root, after the prepare step in repro/README.md):
#   sbatch --array=0-10 --export=ALL,PACKED=<packed dir>,RUNS=<output dir> ops/nn-train.sbatch
set -euo pipefail
python=${PYTHON:-.venv/bin/python}
mapfile -t flags < <("$python" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
m = json.load(open("repro/nn_members.json"))["members"][int(sys.argv[1])]
print(m["name"])
for k, v in m["args"].items():
    if isinstance(v, bool):
        if v: print("--" + k.replace("_", "-"))
    else:
        print("--" + k.replace("_", "-")); print(v)
PY
)
name=${flags[0]}
echo "member=$name host=$(hostname) start=$(date -Is)"
"$python" -B -u -m participant.nn_train --data "$PACKED" --out "$RUNS/$name" "${flags[@]:1}"
echo "end=$(date -Is)"
