"""Fixed allowed-current time support; no induction solver or prediction."""

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_inputs import COLUMNS, SIGNALS

ROOT = Path('run_data/mast-010')


def identities(plan, freeze):
    if digest(ROOT / 'predeclared-plan.json') != freeze['plan_sha256']:
        raise ValueError('Induction support plan changed')
    for name, expected in (plan['source_sha256'] | freeze['source_sha256']
                           | plan['input_cache_sha256']).items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen source/input changed: {name}')


def audit_shot(entry, plan):
    path = Path('run_data/mast-001/inputs') / (Path(entry['path']).stem + '.npz')
    with np.load(path, allow_pickle=False) as data:
        if (set(data.files) != set(COLUMNS) | {'revision', 'source_hash'}
                or str(data['source']) != 'MAST'
                or str(data['revision']) != plan['dataset_revision']
                or str(data['source_hash']) != entry['sha256']):
            raise ValueError('Allowed MAST cache provenance/columns changed')
        centers = np.asarray(data['efit_times'], dtype=np.float64)
        times = np.asarray(data['magnetics_time'], dtype=np.float64)
        if (centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all()
                or np.any(np.diff(centers) <= 0) or times.ndim != 1):
            raise ValueError('Frame/current time schema or order changed')
        result = {'shot': Path(entry['path']).stem, 'frames': len(centers), 'signals': {}}
        for signal in SIGNALS:
            current = np.asarray(data['magnetics_' + signal], dtype=np.float64)
            if current.shape != times.shape:
                raise ValueError(f'Current time shape mismatch: {signal}')
            finite = np.isfinite(times) & np.isfinite(current)
            axis = times[finite]
            if len(axis) < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f'Insufficient/unsorted finite current samples: {signal}')
            supported = (centers >= axis[0]) & (centers <= axis[-1])
            failed = np.flatnonzero(~supported)
            # Select actual adjacent sample intervals, including an exact endpoint.
            left = np.searchsorted(axis, centers[supported], side='right') - 1
            left[left == len(axis) - 1] -= 1
            intervals = axis[left + 1] - axis[left]
            result['signals'][signal] = {
                'finite_samples': len(axis), 'first_sample_ms': float(axis[0]),
                'last_sample_ms': float(axis[-1]),
                'samples_before_first_frame': int(np.sum(axis < centers[0])),
                'unsupported_indices': failed.tolist(),
                'unsupported_times_ms': centers[failed].tolist(),
                'bracketing_sample_intervals_ms': intervals.tolist(),
                'supported_frames': int(supported.sum()),
            }
    return result


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT / 'source-freeze.json').read_text())
    identities(plan, freeze)
    (ROOT / 'running.lock').mkdir()
    started = time.monotonic()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': freeze['plan_sha256'], 'shots': [],
              'MAST_labels_demos_envelopes_read': False, 'predictions_generated': False,
              'pressure_evaluated': False, 'optimizer_steps': 0, 'GPU_count': 0,
              'induced_currents_initialized_or_solved': False}
    output = ROOT / 'support-audit.json'
    with output.open('x') as stream:
        json.dump(result, stream)
    try:
        maximum_rss = 0
        for entry in plan['entries']:
            result['shots'].append(audit_shot(entry, plan))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            maximum_rss = max(maximum_rss, rss if sys.platform == 'darwin' else rss * 1024)
            if time.monotonic() - started > plan['resources']['hard_wall_seconds']:
                raise TimeoutError('Original local CPU cap exceeded')
            if maximum_rss > plan['resources']['max_process_RSS_bytes']:
                raise MemoryError('Original process RSS cap exceeded')
            write_json(output, result)
        identities(plan, freeze)
        frames = sum(s['frames'] for s in result['shots'])
        if frames != plan['expected_frames'] or len(result['shots']) != 20:
            raise ValueError('Original fixed shot/frame count changed')
        failures = sum(len(v['unsupported_indices']) for s in result['shots']
                       for v in s['signals'].values())
        result.update(complete=True, frames=frames, signal_frame_checks=frames * len(SIGNALS),
                      unsupported_signal_frames=failures, time_support_passed=failures == 0,
                      sources_inputs_unchanged=True, maximum_process_RSS_bytes=maximum_rss,
                      vessel_model_ready=False, score_gain=False)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic() - started
        write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'shots'}, indent=2))


if __name__ == '__main__':
    main()
