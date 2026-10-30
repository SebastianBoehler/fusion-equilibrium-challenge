#!/bin/bash
# Site-specific: set --account, --partition and --qos for your cluster on the sbatch command line.
#SBATCH --job-name=mast-gs
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
# Sharded MAST kinetic-GS run: array task k solves shards/<run>/shard-<k>.json.
# Usage: sbatch --array=0-3 --export=ALL,ROOT=<work dir with repo/ and shards/>,RUN=<name>,ARGS="<mast_gs_robust flags>" mast-gs-array.sbatch
set -euo pipefail
root=${ROOT:?set ROOT to the work dir holding repo/ and shards/}
python=${PYTHON:-$root/repo/.venv/bin/python}
cd "$root/repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
shard="$root/shards/$RUN/shard-$SLURM_ARRAY_TASK_ID.json"
echo "run=$RUN shard=$shard args=$ARGS host=$(hostname) start=$(date -Is)"
# shellcheck disable=SC2086
"$python" -B -u -m research.mast_gs_robust --shots "$shard" --out "run_data/$RUN" --workers 32 $ARGS
echo "end=$(date -Is)"
