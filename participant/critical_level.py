"""Training-target stencils for active events in the unchanged DIII-D scorer."""

import numpy as np

from . import fast_score

from lcfs import _best_contour_at
from o_point import find_o_point


def stencil(field):
    field = np.asarray(field, dtype=float)
    r, z, coarse, mask_f = fast_score._geometry()
    sign = fast_score.official.AXIS_SIGN['DIII-D']
    phi = field * sign
    _, _, iz, ir, _ = find_o_point(phi, r, z, coarse)
    border = np.concatenate([phi[0], phi[-1], phi[:, 0], phi[:, -1]])
    low = float(border.min())
    lo, hi = low, float(phi[iz, ir]) - 1e-9
    if hi <= lo:
        raise ValueError('Undefined critical flux bracket')
    best = _best_contour_at(phi, hi, iz, ir, mask_f)
    if best is None:
        raise ValueError('Undefined initial closed surface')
    for _ in range(fast_score.official.N_ITER):
        mid = .5 * (lo + hi)
        contour = _best_contour_at(phi, mid, iz, ir, mask_f)
        if contour is None:
            lo = mid
        else:
            hi, best = mid, contour
    level = hi / sign
    tolerance = 3 * (float(phi[iz, ir]) - low) / 2 ** fast_score.official.N_ITER + 1e-12
    candidates = []
    for row, col in best:
        if abs(row - round(row)) < 1e-8:
            a = (int(round(row)), int(np.floor(col)))
            b = (a[0], min(a[1] + 1, 64))
        elif abs(col - round(col)) < 1e-8:
            a = (int(np.floor(row)), int(round(col)))
            b = (min(a[0] + 1, 64), a[1])
        else:
            raise ValueError('Marching-squares point is not on a grid edge')
        ma, mb = mask_f[a], mask_f[b]
        if ma == mb:
            continue
        t = (.5 - ma) / (mb - ma)
        if 0 <= t <= 1:
            value = field[a] * (1 - t) + field[b] * t
            candidates.append((abs(value - level), 2,
                               [a[0] * 65 + a[1], b[0] * 65 + b[1]], [1 - t, t]))
    for node in np.argwhere(abs(field - level) <= tolerance):
        if np.linalg.norm(best - node, axis=1).min() < 1.5:
            i = int(node[0] * 65 + node[1])
            candidates.append((abs(field[tuple(node)] - level), 1, [i, i], [1., 0.]))
    if not candidates:
        raise ValueError('No supported active critical event')
    # Preserve the validated prototype's envelope-before-node tie convention.
    gap, kind, indices, weights = min(candidates, key=lambda value:
                                    (value[0], 0 if value[1] == 2 else 1, value[2], value[3]))
    if gap > tolerance:
        raise ValueError('Active critical event differs from bisection level')
    return {'indices': [int(i) for i in indices], 'weights': [float(w) for w in weights],
            'kind': kind, 'level': float(level), 'event_level_gap': float(gap),
            'bisection_tolerance': float(tolerance)}
