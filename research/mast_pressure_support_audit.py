"""Predeclared007 temporal-support audit, without pressure extrapolation."""

import json
from pathlib import Path
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_green import E_CHARGE
from participant.mast_inputs import COLUMNS


ROOT = Path('run_data/mast-007')


def audit_shot(path, entry, plan):
    if digest(path) != plan['input_cache_sha256'][str(path)]:
        raise ValueError('Pressure input cache hash changed')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(COLUMNS) | {'source_hash', 'revision'}:
            raise ValueError('Pressure cache contains unexpected columns')
        if (str(data['source_hash']) != entry['sha256']
                or str(data['revision']) != plan['dataset_revision']
                or str(data['source']) != 'MAST'):
            raise ValueError('Pressure input provenance mismatch')
        radius = np.asarray(data['thomson_core_R'], dtype=float)
        times = np.asarray(data['thomson_core_times'], dtype=float)
        centers = np.asarray(data['efit_times'], dtype=float)
        te = np.asarray(data['thomson_core_Te'], dtype=float)
        ne = np.asarray(data['thomson_core_ne'], dtype=float)
    if (radius.ndim != 1 or times.ndim != 1 or centers.ndim != 1
            or te.ndim != 2 or te.shape != ne.shape
            or te.shape != (len(times), len(radius) + 1)
            or not np.isfinite(radius).all() or np.any(radius <= 0)
            or len(np.unique(radius)) != len(radius)
            or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0)):
        raise ValueError('Invalid MAST pressure coordinates or ghost-channel alignment')
    te, ne = te[:, 1:], ne[:, 1:]
    with np.errstate(over='ignore', invalid='ignore'):
        pressure = te * ne * E_CHARGE
    supported = np.zeros(len(centers), dtype=np.int64)
    channel_intervals = []
    for channel in range(len(radius)):
        valid = (np.isfinite(times) & np.isfinite(pressure[:, channel])
                 & (te[:, channel] > 0) & (ne[:, channel] > 0))
        valid_times = times[valid]
        if len(valid_times) < 2:
            channel_intervals.append(None)
            continue
        if np.any(np.diff(valid_times) <= 0):
            raise ValueError('Nonmonotone valid Thomson sample times')
        supported += (centers >= valid_times[0]) & (centers <= valid_times[-1])
        channel_intervals.append([float(valid_times[0]), float(valid_times[-1])])
    missing = np.flatnonzero(supported < plan['minimum_bracketing_channels_per_frame'])
    return {'source_path': entry['path'], 'frames': len(centers),
            'channels': len(radius), 'bracketing_channel_counts': supported.tolist(),
            'unsupported_frame_indices': missing.tolist(),
            'unsupported_frame_times_ms': centers[missing].tolist(),
            'valid_channel_time_intervals_ms': channel_intervals,
            'supported_frames': int(len(centers) - len(missing))}


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen pressure source changed: {name}')
    if digest(Path('run_data/mast-001/data.json')) != plan['manifest_sha256']:
        raise ValueError('Frozen MAST manifest changed')
    (ROOT / 'running.lock').mkdir()
    result = {'complete': False, 'MAST_allowed_inputs_read': True,
              'MAST_labels_demos_envelopes_read': False, 'GPU_count': 0,
              'optimizer_steps': 0, 'predictor_inference': False,
              'candidate_predictions_generated': False, 'shots': [],
              'plan_sha256': digest(ROOT / 'predeclared-plan.json')}
    output = ROOT / 'support-audit.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        for entry in plan['entries']:
            path = Path('run_data/mast-001/inputs') / (Path(entry['path']).stem + '.npz')
            result['shots'].append(audit_shot(path, entry, plan))
            if time.monotonic() - started > plan['hard_local_cpu_wall_seconds']:
                raise TimeoutError('007 input audit CPU cap exceeded')
        for name, expected in plan['source_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Frozen pressure source changed: {name}')
        total = sum(s['frames'] for s in result['shots'])
        supported = sum(s['supported_frames'] for s in result['shots'])
        result.update(complete=True, checks_passed=total == supported,
                      frames=total, supported_frames=supported,
                      unsupported_frames=total - supported, frozen_sources_unchanged=True,
                      source_sha256={str(Path(__file__).relative_to(Path.cwd())):
                                     digest(Path(__file__))})
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic() - started
        write_json(output, result)


if __name__ == '__main__':
    main()
