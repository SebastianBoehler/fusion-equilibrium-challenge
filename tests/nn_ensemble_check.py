"""Score single checkpoints and their average on packed development shots at upload precision."""
import sys
from pathlib import Path

import numpy as np

from participant import fast_score
from participant.nn_submit import Predictor


def predictions(packed, index, checkpoints):
    x = np.load(packed / 'x.npy', mmap_mode='r')
    model = Predictor(checkpoints)
    offsets, out = index['offsets'], []
    for i in range(len(offsets) - 1):
        p = model.predict_inputs(np.asarray(x[offsets[i]:offsets[i + 1]]))
        out.append({'psirz': p['psirz'].astype(np.float32).astype(np.float64),
                    'q95': p['q95'].astype(np.float32).astype(np.float64),
                    'betaN': p['betaN'].astype(np.float32).astype(np.float64)})
    return out


def main(packed: Path, refs_cache: Path, groups: list[list[Path]]):
    index, shots = fast_score.load_truth(packed)
    refs = fast_score.references(shots, refs_cache, 12)
    for checkpoints in groups:
        s = fast_score.composite(shots, refs, predictions(packed, index, checkpoints), 12)
        print('+'.join(p.parent.name for p in checkpoints),
              {k: s[k] for k in ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency')}, flush=True)
        print('   ', s['r2_cons_each'], flush=True)


if __name__ == '__main__':
    # usage: packed refs.pkl a.pt [b.pt ...] ; each ','-joined argument is one averaged group
    main(Path(sys.argv[1]), Path(sys.argv[2]), [[Path(p) for p in g.split(',')] for g in sys.argv[3:]])
