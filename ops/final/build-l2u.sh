#!/bin/zsh
# Candidate L2-u (23 Oct): level-2-current re-solve ONLY for matched shots of the centred (Z=0) class;
# shifted-class and unmatched shots = v5 exactly (inputs and flux). Loops: L2 better on 78% of centred, 41% of shifted.
set -e
cd "${0:A:h}/../.."
PY=.venv/bin/python
rm -rf run_data/final/mast-l2u run_data/final/mast-solve-l2u
mkdir -p run_data/final/mast-l2u/inputs run_data/final/mast-solve-l2u/shots
cp run_data/final/mast/data.json run_data/final/mast-l2u/data.json
$PY -B -c "
import json, numpy as np, shutil, os
from pathlib import Path
D=Path('run_data/final/mast'); S=json.loads((D/'data.json').read_text())['shots']
l2=set(json.load(open('run_data/final/mast-l2/l2-shots.json'))); used=[]
for i,e in enumerate(S):
    n=Path(e['path']).stem; li=Path('run_data/final/mast-l2/inputs')/f'{n}.npz'
    use = i in l2 and not bool(np.load(li)['shifted_override'])
    src = li if use else (D/'inputs'/f'{n}.npz').resolve()
    os.symlink(src.resolve(), Path('run_data/final/mast-l2u/inputs')/f'{n}.npz')
    shutil.copyfile(Path('run_data/final/mast-solve-l2' if use else 'run_data/final/mast-solve-v5')/'shots'/f'{i:04d}.npz', Path('run_data/final/mast-solve-l2u/shots')/f'{i:04d}.npz')
    if use: used.append(i)
json.dump(used, open('run_data/final/mast-l2u/l2u-shots.json','w')); print('L2 shots used:', len(used))
"
export MAST_DATA_DIR=run_data/final/mast-l2u OMP_NUM_THREADS=1 PYTHONPATH=.
$PY -B -m research.mast_final --solve-dir run_data/final/mast-solve-l2u --config mast_private_test --expected 1205 \
  --betan flux-kin --out run_data/final/mast-pred-l2u --workers 13
mkdir -p run_data/final/submission-3-l2u
$PY -B -m participant.final_archive --member run_data/final/d3d-pred/diii_d_private_test.npz 879 \
  --member run_data/final/mast-pred-l2u/mast_private_test.npz 1205 --out run_data/final/submission-3-l2u/fusion-final-3-l2u.zip
echo "ALL-DONE $(date)"
