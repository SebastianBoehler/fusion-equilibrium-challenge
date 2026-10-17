"""Layout-independent actuator-history and spatial Thomson features for neural models.

Only allowed inputs are read: coil and plasma currents, Thomson profiles with their
chord geometry, and the EFIT timestamps. Every Thomson subsystem is resampled onto a
fixed spatial grid so that the 22 released chord layouts share one feature space.
"""

import numpy as np

from vendor.fusion_starter.data_fixes import fix_d3d_ip_times

from .data import SIGNALS

VERSION = 'rich-v1'
CORE_Z = np.linspace(0.0, 0.85, 18)
EDGE_R = np.linspace(1.67, 2.27, 13)
# Window edges (ms, relative to each EFIT time) for causal and acausal current means.
WINDOWS = ((-1.0, 1.0), (-5.0, -1.0), (1.0, 5.0), (-20.0, -5.0), (5.0, 20.0), (-100.0, -20.0))
INPUT_COLUMNS = ('source', 'efit_times', 'magnetics_time', 'magnetics_plasma_current_times',
                 *(f'magnetics_{s}' for s in SIGNALS),
                 *(f'thomson_{sys}_{v}' for sys in ('core', 'edge') for v in ('times', 'Te', 'ne')),
                 'thomson_chord_name', 'thomson_chord_R', 'thomson_chord_Z')


def _window_means(times, values, centers):
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid current time series')
    integral = np.concatenate([[0.0], np.cumsum(np.diff(times) * (values[1:] + values[:-1]) / 2)])

    def mean(a, b):
        lo = np.clip(centers + a, times[0], times[-1])
        hi = np.clip(centers + b, times[0], times[-1])
        width = hi - lo
        area = np.interp(hi, times, integral) - np.interp(lo, times, integral)
        point = np.interp((lo + hi) / 2, times, values)
        return np.where(width > 1e-6, area / np.maximum(width, 1e-6), point)

    m = {w: mean(*w) for w in WINDOWS}
    value = m[(-1.0, 1.0)]
    return np.column_stack([value,
                            m[(1.0, 5.0)] - m[(-5.0, -1.0)],
                            m[(5.0, 20.0)] - m[(-20.0, -5.0)],
                            m[(-100.0, -20.0)] - value])


def current_features(row):
    centers = np.asarray(row['efit_times'], dtype=np.float64)
    shared = np.asarray(row['magnetics_time'], dtype=np.float64)
    ip_times = np.asarray(fix_d3d_ip_times(row), dtype=np.float64)
    blocks = []
    for signal in SIGNALS:
        times = ip_times if signal == 'plasma_current' else shared
        blocks.append(_window_means(times, np.asarray(row[f'magnetics_{signal}'], dtype=np.float64), centers))
    # Order: 21 values, then 21 short slopes, 21 long slopes, 21 slow lags.
    stacked = np.stack(blocks, axis=1)
    ip = np.asarray(row['magnetics_plasma_current'], dtype=np.float64)
    live = np.isfinite(ip) & np.isfinite(ip_times) & (np.abs(ip) > 100.0)
    if live.any():
        start, stop = ip_times[live].min(), ip_times[live].max()
    else:
        start, stop = centers.min(), centers.max()
    timing = np.column_stack([centers / 1000.0, (centers - start) / 1000.0, (stop - centers) / 1000.0])
    return np.concatenate([stacked.transpose(0, 2, 1).reshape(len(centers), -1), timing], axis=1)


def _aligned_profiles(times, values, centers, half_width):
    """Mean of valid samples within +-half_width of each EFIT time, else nearest within 3x."""
    times = np.asarray(times, dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]
    result = np.full((len(centers), values.shape[1]), np.nan)
    lo = np.searchsorted(times, centers - half_width, side='left')
    hi = np.searchsorted(times, centers + half_width, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            block = values[a:b]
            with np.errstate(all='ignore'):
                count = np.isfinite(block).sum(axis=0)
                result[i] = np.where(count > 0, np.nansum(block, axis=0) / np.maximum(count, 1), np.nan)
        else:
            j = int(np.clip(np.searchsorted(times, centers[i]), 1, len(times) - 1)) if len(times) > 1 else 0
            k = j - 1 if j > 0 and abs(times[j - 1] - centers[i]) < abs(times[j] - centers[i]) else j
            if len(times) and abs(times[k] - centers[i]) <= 3 * half_width:
                result[i] = values[k]
    return result


def _grid(profile, coordinate, grid):
    """Interpolate one profile onto a fixed grid; zero outside the valid measured span."""
    valid = np.isfinite(profile)
    out = np.zeros(len(grid))
    mask = np.zeros(len(grid))
    if valid.sum() >= 2:
        c, p = coordinate[valid], profile[valid]
        order = np.argsort(c)
        c, p = c[order], p[order]
        inside = (grid >= c[0]) & (grid <= c[-1])
        out[inside] = np.interp(grid[inside], c, p)
        mask[inside] = 1.0
    return out, mask


def thomson_features(row):
    centers = np.asarray(row['efit_times'], dtype=np.float64)
    names = np.asarray(row['thomson_chord_name']).astype(str)
    chord_r = np.asarray(row['thomson_chord_R'], dtype=np.float64)
    chord_z = np.asarray(row['thomson_chord_Z'], dtype=np.float64)
    layouts = {'core': (np.char.startswith(names, 'TS_core'), chord_z, CORE_Z, 10.0),
               'edge': (np.char.startswith(names, 'TS_tangential'), chord_r, EDGE_R, 30.0)}
    columns = []
    for system, (selected, coordinate, grid, half_width) in layouts.items():
        te = np.asarray(row[f'thomson_{system}_Te'], dtype=np.float64)
        ne = np.asarray(row[f'thomson_{system}_ne'], dtype=np.float64)
        position = coordinate[selected]
        if te.shape != ne.shape or te.shape[1] != len(position):
            raise ValueError(f'Thomson {system} channels do not match chord geometry')
        times = row[f'thomson_{system}_times']
        te_t = _aligned_profiles(times, te, centers, half_width) / 1e3
        ne_t = _aligned_profiles(times, ne, centers, half_width) / 1e19
        pe_t = te_t * ne_t
        grids = {name: np.zeros((len(centers), len(grid))) for name in ('te', 'ne', 'pe', 'mask')}
        for i in range(len(centers)):
            grids['te'][i], grids['mask'][i] = _grid(te_t[i], position, grid)
            grids['ne'][i], _ = _grid(ne_t[i], position, grid)
            grids['pe'][i], _ = _grid(pe_t[i], position, grid)
        with np.errstate(all='ignore'):
            peak_ne = np.nanmax(np.where(np.isfinite(ne_t), ne_t, -np.inf), axis=1)
            peak_te = np.nanmax(np.where(np.isfinite(te_t), te_t, -np.inf), axis=1)
            weight = np.where(np.isfinite(pe_t), pe_t, 0.0)
            total = weight.sum(axis=1)
            centroid = (weight * position).sum(axis=1) / np.where(total > 0, total, 1.0)
            spread = np.sqrt((weight * (position - centroid[:, None]) ** 2).sum(axis=1)
                             / np.where(total > 0, total, 1.0))
            dense = np.isfinite(ne_t) & (ne_t > 0.3 * peak_ne[:, None])
            far = np.where(dense, position, -np.inf).max(axis=1)
            near = np.where(dense, position, np.inf).min(axis=1)
            count = np.isfinite(ne_t).sum(axis=1) / max(len(position), 1)
        summary = np.column_stack([peak_ne, peak_te, total / max(len(position), 1), centroid, spread,
                                   far, near, count])
        summary[~np.isfinite(summary)] = 0.0
        columns += [grids['te'], grids['ne'], grids['pe'], grids['mask'], summary]
    return np.concatenate(columns, axis=1)


def rich_features(row):
    if row['source'] != 'DIII-D':
        raise ValueError('Rich features currently support DIII-D only')
    result = np.concatenate([current_features(row), thomson_features(row)], axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite rich features')
    return result.astype(np.float32)


def feature_count():
    return 21 * 4 + 3 + 2 * 8 + 4 * (len(CORE_Z) + len(EDGE_R))


def canonicalize_currents(x):
    """Exact sign symmetry: flip all toroidal currents so Ip > 0; Bt sign does not enter psi.

    The released flux is stored with the magnetic axis as its minimum for both Ip
    signs, so after this flip reversed-current shots share the normal mapping.
    Columns follow current_features: four 21-signal blocks (value, short slope,
    long slope, slow lag) in SIGNALS order, ECOILA=0, bcoil=1, F-coils, Ip=20.
    """
    x = np.array(x, dtype=np.float32, copy=True)
    ip_sign = np.where(x[:, 20] < 0, -1.0, 1.0).astype(np.float32)
    bt_sign = np.where(x[:, 1] < 0, -1.0, 1.0).astype(np.float32)
    for block in range(4):
        base = 21 * block
        toroidal = [base] + list(range(base + 2, base + 21))
        x[:, toroidal] *= ip_sign[:, None]
        x[:, base + 1] *= bt_sign
    return x
