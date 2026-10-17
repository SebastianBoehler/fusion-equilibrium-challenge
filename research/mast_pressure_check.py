"""Predeclared003 analytic volume-pressure check; no MAST rows or labels."""

import json
from pathlib import Path
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_pressure import volume_beta_n


ROOT = Path('run_data/mast-003')


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    (ROOT / 'running.lock').mkdir()
    result = {'complete': False, 'MAST_data_read': False, 'GPU_count': 0,
              'optimizer_steps': 0, 'candidate_predictions_generated': False,
              'plan_sha256': digest(ROOT / 'predeclared-plan.json')}
    output = ROOT / 'functional-check.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        r, z = np.linspace(.8, 1.6, 401), np.linspace(-.7, .7, 401)
        rr, zz = np.meshgrid(r, z)
        radius = ((rr - 1.2) / .4) ** 2 + (zz / .7) ** 2
        core = radius <= 1
        normalization = (1., 1.2, .4, 1e6)
        uniform = np.full_like(rr, 1000.)
        _, average = volume_beta_n(r, z, uniform, core, *normalization)
        np.testing.assert_allclose(average, 1000., rtol=1e-12)
        pressure = 1000 * np.maximum(1 - radius, 0)
        reference, average = volume_beta_n(r, z, pressure, core, *normalization)
        result['parabolic_pressure_average'] = average
        result['parabolic_relative_error'] = abs(average / 500 - 1)
        np.testing.assert_allclose(average, 500., rtol=2e-4)
        cases = [(r, z, 3 * pressure, core, 1., 1.2, .4, 1e6, 3),
                 (r, z, pressure, core, 2., 1.2, .4, 1e6, .5),
                 (r, z, pressure, core, 1., 1.2, .4, 2e6, .5),
                 (3 * r, 3 * z, pressure, core, 3., 3.6, 1.2, 1e6, 3)]
        for *args, factor in cases:
            actual, _ = volume_beta_n(*args)
            np.testing.assert_allclose(actual, reference * factor, rtol=1e-12)
        invalid = [(r, z, -uniform, core, *normalization),
                   (r, z, uniform, np.zeros_like(core), *normalization),
                   (r[::-1], z, uniform, core, *normalization),
                   (r, z, uniform * np.nan, core, *normalization),
                   (r, z, uniform, core, 0., 1.2, .4, 1e6),
                   (r, z, uniform, core, 1., 1.2, .4, 0.)]
        for args in invalid:
            try:
                volume_beta_n(*args)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid pressure integral accepted')
        if time.monotonic() - started > plan['hard_local_cpu_wall_seconds']:
            raise TimeoutError('003 local analytic cap exceeded')
        result.update(complete=True, checks_passed=True, rejected_controls=len(invalid),
                      source_sha256={str(p): digest(p) for p in
                          [Path('participant/mast_pressure.py'), Path(__file__).relative_to(Path.cwd())]})
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic() - started
        write_json(output, result)


if __name__ == '__main__':
    main()
