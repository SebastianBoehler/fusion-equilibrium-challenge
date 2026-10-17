"""Fixed geometry-only rectangular unit-coil quadrature; no predictions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

from participant.artifacts import digest, write_json
from participant.mast_green import greens
from participant.mast_inputs import COLUMNS, SIGNALS

ROOT = Path('run_data/mast-011')
GEOMETRY = ('coil_input_column', 'coil_R', 'coil_Z', 'coil_width', 'coil_height',
            'efit_grid_R', 'efit_grid_Z')


def require(value, message):
    if not value:
        raise ValueError(message)


def identities(plan, freeze):
    require(digest(ROOT / 'predeclared-plan.json') == freeze['plan_sha256'], 'Plan changed')
    for name, expected in (plan['source_sha256'] | freeze['source_sha256']
                           | plan['input_cache_sha256']).items():
        require(digest(Path(name)) == expected, f'Frozen identity changed: {name}')


def budget(started, plan):
    require(time.monotonic() - started <= plan['resources']['hard_local_wall_seconds'],
            'Original local wall cap exceeded')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == 'darwin' else rss * 1024
    require(rss <= plan['resources']['max_process_RSS_bytes'], 'Process RSS cap exceeded')
    return rss


def controls(orders):
    checks = {}
    for order in orders:
        nodes, weights = leggauss(order)
        x, y = np.meshgrid(nodes, nodes)
        w = np.outer(weights, weights) / 4
        constant, quadratic = float(w.sum()), float((w * (x*x + y*y)).sum())
        require(abs(constant - 1) <= 1e-14 and abs(quadratic - 2/3) <= 1e-14,
                'Polynomial quadrature control failed')
        checks[str(order)] = {'constant': constant, 'quadratic': quadratic}
    return checks


def response(rc, zc, width, height, r, z, nodes, weights, started, plan):
    out = np.zeros(len(r), dtype=np.float64)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            budget(started, plan)
            out += weights[i] * weights[j] * greens(
                (rc + a*width/2)[None], (zc + b*height/2)[None],
                r[:, None], z[:, None]).sum(axis=1)
    return out


def audit(entry, plan, started):
    cache = Path('run_data/mast-001/inputs') / (Path(entry['path']).stem + '.npz')
    with np.load(cache, allow_pickle=False) as data:
        require(set(data.files) == set(COLUMNS) | {'revision', 'source_hash'}
                and str(data['source']) == 'MAST'
                and str(data['revision']) == plan['dataset_revision']
                and str(data['source_hash']) == entry['sha256'], 'Cache provenance changed')
        row = {key: np.asarray(data[key]) for key in GEOMETRY}
    names = row['coil_input_column'].astype(str)
    rc, zc, width, height = [row[key].astype(np.float64) for key in GEOMETRY[1:5]]
    require(names.ndim == 1 and all(a.shape == names.shape for a in (rc, zc, width, height)),
            'Coil geometry shapes differ')
    require(all(np.isfinite(a).all() for a in (rc, zc, width, height))
            and np.all(rc-width/2 > 0) and np.all(width >= 0) and np.all(height >= 0),
            'Invalid coil rectangle geometry')
    axes = [row[key].astype(np.float64) for key in GEOMETRY[5:]]
    require(all(a.shape == (65,) and np.isfinite(a).all() and np.all(np.diff(a) > 0)
                for a in axes) and np.all(axes[0] > 0), 'Invalid permitted grid geometry')
    rr, zz = np.meshgrid(axes[0][[0,16,32,48,64]], axes[1][[0,16,32,48,64]])
    r, z = rr.ravel(), zz.ravel()
    inside = ((np.abs(r[:,None]-rc) <= width/2)
              & (np.abs(z[:,None]-zc) <= height/2))
    item = {'shot': Path(entry['path']).stem, 'query_R': r.tolist(), 'query_Z': z.tolist(),
            'query_inside_rectangle_pairs': np.argwhere(inside).tolist(), 'groups': {}}
    if inside.any():
        item.update(geometry_support_passed=False, quadrature_reached=False)
        return item
    for name in np.unique(names):
        require(name.removeprefix('magnetics_') in SIGNALS, 'Unsupported original coil column')
        pick = names == name
        arrays = [a[pick] for a in (rc, zc, width, height)]
        normalization = 1 if name == 'magnetics_sol_current' else int(pick.sum())
        values = {}
        for order in plan['orders']:
            nodes, weights = leggauss(order)
            values[str(order)] = response(*arrays, r, z, nodes, weights/2, started, plan)/normalization
        baseline = response(*arrays, r, z, np.array([-.5,.5]), np.array([.5,.5]),
                            started, plan)/normalization
        require(all(np.isfinite(v).all() for v in (*values.values(), baseline)),
                'Nonfinite actual unit-coil response')
        absolute = float(np.sum(np.abs(values['8']-values['16'])))
        denominator = float(np.sum(np.abs(values['16'])))
        require(denominator > 0, 'Zero reference response')
        relative = absolute/denominator
        item['groups'][name] = {'coil_elements': int(pick.sum()), 'normalization': normalization,
            'responses': {key: value.tolist() for key,value in values.items()},
            'original_fourpoint_response': baseline.tolist(), '8vs16_absolute_L1': absolute,
            '8vs16_relative_L1': relative, 'convergence_passed': relative <= 1e-6 or absolute <= 1e-14,
            'fourpoint_vs16_relative_L1': float(np.sum(np.abs(baseline-values['16'])))/denominator}
    item.update(geometry_support_passed=True, quadrature_reached=True)
    return item


def main():
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    identities(plan, freeze)
    (ROOT/'running.lock').mkdir()
    started = time.monotonic()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': freeze['plan_sha256'], 'shots': [], 'MAST_labels_read': False,
              'actual_currents_pressure_predictions_read_or_generated': False,
              'optimizer_steps': 0, 'GPU_count': 0, 'SSH_attempts': 0}
    try:
        result['polynomial_controls'] = controls(plan['orders'])
        for entry in plan['entries']:
            item = audit(entry, plan, started)
            result['shots'].append(item)
            write_json(ROOT/'quadrature-check.json', result)
            if not item['geometry_support_passed']:
                break
        identities(plan, freeze)
        support = len(result['shots']) == 20 and all(s['geometry_support_passed'] for s in result['shots'])
        result.update(complete=True, geometry_support_passed=support,
            quadrature_convergence_passed=support and all(g['convergence_passed']
                for s in result['shots'] for g in s['groups'].values()),
            sources_inputs_unchanged=True, maximum_process_RSS_bytes=budget(started, plan),
            model_replacement_ready=False, score_gain=False)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic()-started
        write_json(ROOT/'quadrature-check.json', result)
    print(json.dumps({key:value for key,value in result.items() if key != 'shots'}, indent=2))


if __name__ == '__main__':
    main()
