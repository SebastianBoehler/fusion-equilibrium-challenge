#!/bin/bash
# CPU-only finite-measurement cost fixture; no optimizer or GPU request.
#SBATCH --job-name=fusion-finitecost037
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=<cluster-work>/fusion/finite-cost-037-20261010/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/finite-cost-037-20261010/logs/%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cost_root=<cluster-work>/fusion/finite-cost-037-20261010
python=<cluster-work>/fusion/nn-008/venv/bin/python
cd "$cost_root/repo"
sha256sum --check source-sha256.txt
sha256sum --check "$cost_root/plan-sha256.txt"
lscpu
"$python" -u research/finite_cluster_cost.py "$cost_root"
