#!/bin/bash
# Runs only within the one scheduled CPU-only srun allocation.
set -euo pipefail
cost_root=<cluster-work>/fusion/finite-cost-044-srun-20261010
python=<cluster-work>/fusion/nn-008/venv/bin/python
test -n "${SLURM_JOB_ID:-}"
mkdir "$cost_root/allocated.lock"
printf '%s\n' "$SLURM_JOB_ID" > "$cost_root/job-id.txt"
trap 'status=$?; printf "%s\n" "$status" > "$cost_root/step-exit.txt"' EXIT
scontrol update "JobId=$SLURM_JOB_ID" Requeue=0
scontrol show job "$SLURM_JOB_ID" -o > "$cost_root/allocation-state.txt"
python3 - "$cost_root" <<'PY'
import json,sys,re
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1])
value=dict(re.findall(r'(\w+)=([^\s]+)',(root/'allocation-state.txt').read_text()))
checks={'partition':value.get('Partition')=='cpu-galvani',
        'time':value.get('TimeLimit')=='00:10:00','cpus':value.get('NumCPUs')=='16',
        'nodes':value.get('NumNodes')=='1','no_requeue':value.get('Requeue')=='0',
        'no_gpu':'gres/gpu' not in value.get('AllocTRES',''),
        'memory': 'mem=32G' in value.get('AllocTRES','') or 'mem=32768M' in value.get('AllocTRES','')}
with (root/'allocation-receipt.json').open('x') as f:
    json.dump({'observed_at':datetime.now(timezone.utc).isoformat(),'job':value,'checks':checks},f,indent=2)
if not all(checks.values()):raise ValueError('Actual044 CPU allocation/resources differ')
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$cost_root/repo"
sha256sum --check source-sha256.txt
sha256sum --check "$cost_root/plan-sha256.txt"
lscpu
"$python" -u -m research.finite_cluster_revision "$cost_root"
