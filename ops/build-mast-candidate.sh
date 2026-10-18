#!/bin/zsh
# Build an upload-ready MAST-only candidate from a finished mast_gs_robust run (public split).
# Usage: ops/build-mast-candidate.sh <solve-dir> <candidate-id> <flux|share> <name-slug>
# Output: run_data/submission-mast-<id>/fusion-mast-<slug>-<id>-mastonly.zip (+ receipts)
set -euo pipefail
cd "${0:A:h}/.."
solve=$1 id=$2 betan=$3 slug=$4
out=run_data/submission-mast-$id
n=$(ls "$solve"/shots | grep -vc partial)
[[ $n == 1206 ]] || { echo "incomplete run: $n/1206 shots in $solve"; exit 1; }
PYTHONPATH=. .venv/bin/python -B -m research.mast_final --solve-dir "$solve" --config mast_public_test \
  --expected 1206 --betan "$betan" --out "$out"
PYTHONPATH=. .venv/bin/python -B -m participant.final_archive \
  --member "$out/mast_public_test.npz" 1206 --out "$out/fusion-mast-$slug-$id-mastonly.zip"
