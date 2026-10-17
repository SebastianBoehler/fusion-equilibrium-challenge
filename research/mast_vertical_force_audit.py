"""One frozen mast009 CPU diagnostic; no targets, inference or model changes."""

import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_inputs import COLUMNS
from research.mast_vertical_force import baseline_current, coil_derivatives, vertical_force
from research.mast_vertical_force_controls import check_controls

ROOT = Path('run_data/mast-009')


def identities(plan, freeze):
    for name, expected in plan['source_sha256'].items() | freeze['source_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if digest(ROOT / 'predeclared-plan.json') != freeze['plan_sha256']:
        raise ValueError('Predeclared force plan changed')
    if digest(Path('run_data/mast-001/data.json')) != plan['manifest_sha256']:
        raise ValueError('Frozen MAST manifest changed')
    for name, expected in plan['input_cache_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen allowed-input cache changed: {name}')


def row_from_cache(entry, plan):
    path = Path('run_data/mast-001/inputs') / (Path(entry['path']).stem + '.npz')
    if digest(path) != plan['input_cache_sha256'][str(path)]:
        raise ValueError('Input cache identity changed before read')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(COLUMNS) | {'source_hash', 'revision'}:
            raise ValueError('Unexpected cache columns')
        if (str(data['source_hash']) != entry['sha256']
                or str(data['revision']) != plan['dataset_revision']
                or str(data['source']) != 'MAST'):
            raise ValueError('MAST allowed-input provenance mismatch')
        row = {name: data[name] for name in COLUMNS}
    times = np.asarray(row['efit_times'])
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid exact frame order')
    return row


def shot_diagnostic(entry, row, gates, budget):
    currents, weights, sr, sz, r0, variance = baseline_current(row)
    columns = coil_derivatives(row, sr, sz, gates['derivative_h_m'], budget)
    derivative = np.zeros((3,) + weights.shape)
    response_checks = []
    for column, response in columns:
        reference_scale = np.abs(response[0]).sum()
        errors = np.abs(response[1:] - response[0]).sum(axis=1)
        bound = (gates['derivative_response_relative_L1_tolerance'] * reference_scale
                 + gates['derivative_absolute_Tm_per_A_tolerance'] * len(sr))
        response_checks.append({'signal_index': column, 'reference_L1_Tm_per_A':
                                float(reference_scale), 'errors_L1_Tm_per_A': errors.tolist(),
                                'allowed_error_L1_Tm_per_A': float(bound),
                                'passed': bool(np.all(errors <= bound))})
        derivative += currents[None, :, column, None] * 1000 * response[:, None, :]
    force, scale, residual = vertical_force(weights, derivative[0])
    paired_forces = [vertical_force(weights, d)[0] for d in derivative[1:]]
    errors = np.abs(np.asarray(paired_forces) - force)
    bound = gates['force_relative_L1_tolerance'] * scale + gates['force_absolute_N_tolerance']
    numerical_pass = (all(c['passed'] for c in response_checks)
                      and bool(np.all(errors <= bound)))
    stationary = np.abs(residual) <= gates['stationary_external_residual_tolerance']
    budget()
    return {'path': entry['path'], 'frames': len(force),
            'times_ms': np.asarray(row['efit_times']).tolist(),
            'source_cells': len(sr), 'baseline_R0_m': r0.tolist(),
            'baseline_variance_m2': variance.tolist(), 'force_N': force.tolist(),
            'force_L1_scale_N': scale.tolist(), 'signed_normalized_residual': residual.tolist(),
            'central_difference_forces_N': [f.tolist() for f in paired_forces],
            'central_force_errors_N': errors.tolist(), 'response_checks': response_checks,
            'numerical_checks_passed': numerical_pass,
            'stationary_external_gate_passed_frames': int(stationary.sum()),
            'stationary_external_gate_failed_indices': np.flatnonzero(~stationary).tolist()}


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT / 'source-freeze.json').read_text())
    identities(plan, freeze)
    import scipy
    import participant.mast_green as green
    import participant.mast_inputs as inputs
    import research.mast_vertical_force as helper
    import research.mast_vertical_force_controls as controls
    runtime = {'python': sys.version.split()[0], 'executable': sys.executable,
               'numpy': np.__version__, 'scipy': scipy.__version__,
               'module_origins': {m.__name__: str(Path(m.__file__).absolute())
                                  for m in (np, scipy, green, inputs, helper, controls)}}
    if runtime != freeze['runtime']:
        raise ValueError('Frozen runtime or module origin changed')
    (ROOT / 'running.lock').mkdir()
    started = time.monotonic()
    maximum_rss = 0

    def budget():
        nonlocal maximum_rss
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        maximum_rss = max(maximum_rss, rss if sys.platform == 'darwin' else rss * 1024)
        if time.monotonic() - started > plan['gates']['hard_local_cpu_wall_seconds']:
            raise TimeoutError('Force audit exceeded fixed CPU wall cap')
        if maximum_rss > plan['gates']['max_rss_bytes']:
            raise MemoryError('Force audit exceeded fixed process RSS cap')

    result = {'complete': False, 'plan_sha256': freeze['plan_sha256'],
              'runtime': runtime, 'allowed_inputs_read': False,
              'MAST_labels_demos_envelopes_read': False, 'predictions_generated': False,
              'optimizer_steps': 0, 'GPU_count': 0, 'shots': []}
    output = ROOT / 'force-audit.json'
    with output.open('x') as stream:
        json.dump(result, stream)
    try:
        result['controls'] = check_controls(plan['gates'])
        write_json(output, result)
        budget()
        for entry in plan['entries']:
            budget()
            row = row_from_cache(entry, plan)
            result['allowed_inputs_read'] = True
            shot = shot_diagnostic(entry, row, plan['gates'], budget)
            result['shots'].append(shot)
            write_json(output, result)
            if not shot['numerical_checks_passed']:
                raise ValueError('Original numerical derivative/force gate failed')
        identities(plan, freeze)
        frames = sum(s['frames'] for s in result['shots'])
        if frames != plan['expected_frames'] or len(result['shots']) != 20:
            raise ValueError('Original frame/shot count changed')
        passed = sum(s['stationary_external_gate_passed_frames'] for s in result['shots'])
        residuals = np.concatenate([s['signed_normalized_residual'] for s in result['shots']])
        result.update(complete=True, numerical_checks_passed=True, frames=frames,
                      stationary_external_gate_passed_frames=passed,
                      stationary_external_gate_failed_frames=frames - passed,
                      max_absolute_normalized_residual=float(np.max(np.abs(residuals))),
                      median_absolute_normalized_residual=float(np.median(np.abs(residuals))),
                      stationary_external_hypothesis_passed=passed == frames,
                      sources_inputs_manifest_unchanged=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(actual_body_seconds=time.monotonic() - started,
                      maximum_process_RSS_bytes=maximum_rss)
        write_json(output, result)


if __name__ == '__main__':
    main()
