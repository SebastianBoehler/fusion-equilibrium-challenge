"""Where does the ensemble's development error live? Per shot and per discharge phase."""
import json
import sys
from pathlib import Path

import numpy as np

from participant.nn_submit import Predictor

# rich.py feature layout: 84 current features, then t, t - t_start, t_stop - t (seconds)
T_START, T_STOP = 85, 86


def main(packed: Path, checkpoints: list[Path]):
    index = json.loads((packed / 'index.json').read_text())
    x = np.load(packed / 'x.npy', mmap_mode='r')
    psi = np.load(packed / 'psi.npy', mmap_mode='r')
    scal = np.load(packed / 'scalars.npy', mmap_mode='r')
    model = Predictor(checkpoints)
    o = index['offsets']
    sse, phase, shot_sse, q_err, b_err, ip = [], [], [], [], [], []
    for i in range(len(o) - 1):
        xi = np.asarray(x[o[i]:o[i + 1]])
        p = model.predict_inputs(xi)
        err = ((p['psirz'].reshape(len(xi), -1).astype(np.float16).astype(float)
                - np.asarray(psi[o[i]:o[i + 1]], dtype=float)) ** 2).sum(1)
        sse.append(err)
        shot_sse.append(err.sum())
        phase.append(np.where(xi[:, T_START] < 0.6, 0, np.where(xi[:, T_STOP] < 0.4, 2, 1)))
        q_err.append(p['q95'] - scal[o[i]:o[i + 1], 0])
        b_err.append(p['betaN'] - scal[o[i]:o[i + 1], 1])
        ip.append(np.abs(xi[:, 20]))
    sse, phase = np.concatenate(sse), np.concatenate(phase)
    q_err, b_err, ip = map(np.concatenate, (q_err, b_err, ip))
    shot_sse = np.array(shot_sse)
    names = ['ramp-up (<0.6 s after Ip>100 kA)', 'flat-top', 'ramp-down (<0.4 s before end)']
    print('frames', len(sse))
    for k, n in enumerate(names):
        m = phase == k
        print(f'{n:34s} frames {m.mean():.3f}  psi SSE share {sse[m].sum() / sse.sum():.3f}  '
              f'q95 SSE share {np.nansum(q_err[m] ** 2) / np.nansum(q_err ** 2):.3f}  '
              f'betaN SSE share {np.nansum(b_err[m] ** 2) / np.nansum(b_err ** 2):.3f}')
    order = np.argsort(shot_sse)[::-1]
    for top in (1, 5, 10, 26):
        print(f'top {top:2d} shots ({top / len(shot_sse):.1%}) hold {shot_sse[order[:top]].sum() / shot_sse.sum():.3f} of psi SSE')
    print('worst shots:', [index['names'][j] for j in order[:5]])
    low = ip < 600
    print(f'|Ip|<600 kA frames {low.mean():.3f} hold psi SSE share {sse[low].sum() / sse.sum():.3f}')
    big = q_err ** 2
    ok = np.isfinite(big)
    qs = np.sort(big[ok])[::-1]
    print(f'q95: top 1% of frames hold {qs[:max(1, len(qs) // 100)].sum() / qs.sum():.3f} of SSE')


if __name__ == '__main__':
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
