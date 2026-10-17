#!/bin/bash
# One persistent normal-queue srun client; no allocation retry.
set -euo pipefail
cost_root=<cluster-work>/fusion/finite-cost-044-srun-20261010
mkdir "$cost_root/srun-dispatch.lock"
squeue -u <user> -h -o '%i|%j|%T' > "$cost_root/pre-dispatch-queue.txt"
python3 - "$cost_root/pre-dispatch-queue.txt" <<'PY'
import sys
from pathlib import Path
if any('|fusion-' in s for s in Path(sys.argv[1]).read_text().splitlines()):
    raise ValueError('Another Fusion allocation is queued or running')
PY
nohup srun --job-name=fusion-finitecost044 --account=<account> --partition=cpu-galvani \
    --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G \
    --time=00:10:00 --kill-on-bad-exit=1 --input=/dev/null \
    --output="$cost_root/logs/%j.out" --error="$cost_root/logs/%j.err" \
    bash "$cost_root/repo/ops/finite-cost-044-run.sh" \
    > "$cost_root/srun-client.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$cost_root/srun-client.pid"
date -Is > "$cost_root/client-dispatch-time.txt"
