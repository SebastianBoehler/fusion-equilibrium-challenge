#!/bin/zsh
# Candidate L2 (23 Oct): matched shots re-solved with FAIR-MAST level-2 PF currents (release class frozen),
# then E2 magnetics TTA; unmatched shots = v5. betaN channel from v5.
set -e
cd "${0:A:h}/../.."
until ! pgrep -f "research/fairmast_pf.py" >/dev/null; do sleep 30; done
echo "pf done $(date)"; ls run_data/final/mast-pf | wc -l
PY=.venv/bin/python
export OMP_NUM_THREADS=1 PYTHONPATH=.
MAST_DATA_DIR=run_data/final/mast $PY -B -m research.build_l2_inputs run_data/final/mast run_data/mag-tta/match-private-accepted-b.json run_data/final/mast-pf run_data/final/mast-l2
export MAST_DATA_DIR=run_data/final/mast-l2
SH=run_data/final/mast-l2/l2-shots.json
base=(--shots $SH --workers 13 --passive ALLgf)
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2 $base --te-power 0.75
echo "base done $(date)"
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2-te05 $base --te-power 0.5
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2-te10 $base --te-power 1.0
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2-te15 $base --te-power 1.5
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2-e10 $base --te-power 0.75 --edge-current 0.1
$PY -B -m research.mast_gs_robust --out run_data/final/mast-solve-l2-e20 $base --te-power 0.75 --edge-current 0.2
echo "variants done $(date)"
$PY -B -m research.mag_tta_apply run_data/final/mast-l2 run_data/final/mast-mag run_data/mag-tta/match-private-accepted-b.json \
  run_data/final/mast-solve-l2-tta run_data/final/mast-solve-l2 run_data/final/mast-solve-l2-te05 run_data/final/mast-solve-l2-te10 \
  run_data/final/mast-solve-l2-te15 run_data/final/mast-solve-l2-e10 run_data/final/mast-solve-l2-e20 --max-dist 0.003 --workers 13 \
  --fallback run_data/final/mast-solve-v5
$PY -B -m research.mast_final --solve-dir run_data/final/mast-solve-l2-tta --config mast_private_test --expected 1205 \
  --betan flux-kin --out run_data/final/mast-pred-l2-tta --workers 13
$PY -B -m research.combine_members --psirz run_data/final/mast-pred-l2-tta/mast_private_test.npz \
  --q95 run_data/final/mast-pred-l2-tta/mast_private_test.npz --betan run_data/final/mast-pred-v5/mast_private_test.npz \
  --shots 1205 --out run_data/final/mast-pred-l2-tta-c/mast_private_test.npz
mkdir -p run_data/final/submission-3-l2-tta
$PY -B -m participant.final_archive --member run_data/final/d3d-pred/diii_d_private_test.npz 879 \
  --member run_data/final/mast-pred-l2-tta-c/mast_private_test.npz 1205 --out run_data/final/submission-3-l2-tta/fusion-final-3-l2-tta.zip
echo "ALL-DONE $(date)"
