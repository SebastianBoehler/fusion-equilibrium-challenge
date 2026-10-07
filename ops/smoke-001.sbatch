#!/bin/bash
#SBATCH --job-name=fusion-smoke001-20261007
#SBATCH --partition=cpu-galvani
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=<cluster-work>/fusion/smoke-001-20261007/logs/%j.out
#SBATCH --error=<cluster-work>/fusion/smoke-001-20261007/logs/%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
root=<cluster-work>/fusion/smoke-001-20261007
base_python=<cluster-work>/learn2design/exp123-gentle-refinement-20260916-235735-fecfe302/venv/bin/python
"$base_python" -m venv "$root/venv"
"$root/venv/bin/python" -m pip install --disable-pip-version-check \
  numpy==2.4.6 scipy==1.17.1 pyarrow==24.0.0 scikit-image==0.26.0
cd "$root/repo"
"$root/venv/bin/python" -u -m participant.smoke \
  --manifest protocols/smoke-001-data.json --out "$root/result"
