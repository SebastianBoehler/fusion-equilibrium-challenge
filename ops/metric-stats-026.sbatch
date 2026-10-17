#!/bin/bash
# Independent read-only training statistics; no GPU or learned candidate.
#SBATCH --job-name=fusion-metrics026
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/metric-stats-026-20261009/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/metric-stats-026-20261009/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stats_root=<cluster-work>/fusion/metric-stats-026-20261009
python=<cluster-work>/fusion/nn-008/venv/bin/python
cd "$stats_root"
sha256sum --check source-sha256.txt
"$python" -u training_metric_stats.py \
  <cluster-work>/fusion/contour-024-20261009 \
  "$stats_root/metric-stats.json"
