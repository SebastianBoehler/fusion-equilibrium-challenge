"""Score a neural checkpoint on the two frozen 8-shot regression groups separately."""
import json
import sys
from pathlib import Path

import numpy as np

from participant import fast_score
from participant.nn_submit import Predictor


def main(packed: Path, split: Path, checkpoints: list[Path]):
    index, shots = fast_score.load_truth(packed)
    x = np.load(packed / 'x.npy', mmap_mode='r')
    model = Predictor(checkpoints)
    offsets = index['offsets']
    preds = []
    for i in range(len(shots)):
        p = model.predict_inputs(np.asarray(x[offsets[i]:offsets[i + 1]]))
        preds.append({'psirz': p['psirz'].astype(np.float32).astype(np.float64),
                      'q95': p['q95'].astype(np.float32).astype(np.float64),
                      'betaN': p['betaN'].astype(np.float32).astype(np.float64)})
    groups = json.loads(split.read_text())['regression']
    for name, entries in groups.items():
        wanted = [Path(e['path']).stem for e in entries]
        pick = [index['names'].index(w) for w in wanted]
        sub = [shots[i] for i in pick]
        refs = fast_score.references(sub, None, 8)
        s = fast_score.composite(sub, refs, [preds[i] for i in pick], 8)
        print(name, {k: s[k] for k in ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency')})


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
