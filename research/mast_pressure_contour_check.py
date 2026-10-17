"""Predeclared004 analytic revision; rejected003 remains immutable."""

import json
from pathlib import Path
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_pressure_contour import contour_beta_n


ROOT = Path('run_data/mast-004')


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    if digest(Path('run_data/mast-003/functional-check.json')) != plan['rejected003_result_sha256']:
        raise ValueError('Rejected003 evidence changed')
    (ROOT / 'running.lock').mkdir()
    result = {'complete': False, 'MAST_data_read': False, 'GPU_count': 0,
              'optimizer_steps': 0, 'candidate_predictions_generated': False,
              'plan_sha256': digest(ROOT / 'predeclared-plan.json')}
    output = ROOT / 'functional-check.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        r, z = np.linspace(.7, 1.7, 401), np.linspace(-.8, .8, 401)
        rr, zz = np.meshgrid(r, z)
        theta = np.linspace(0, 2 * np.pi, 401)
        contour = np.column_stack((1.2 + .4 * np.cos(theta), .7 * np.sin(theta)))
        contour[-1] = contour[0]
        axis = np.array([1.2, 0.])
        uniform = np.full_like(rr, 1000.)
        _, average = contour_beta_n(r, z, uniform, contour, axis, 1., .4, 1e6)
        np.testing.assert_allclose(average, 1000., rtol=1e-12)
        pressure = 1000 * (1 - ((rr - 1.2) / .4) ** 2 - (zz / .7) ** 2)
        reference, average = contour_beta_n(r, z, pressure, contour, axis, 1., .4, 1e6)
        result['parabolic_pressure_average'] = average
        result['parabolic_relative_error'] = abs(average / 500 - 1)
        np.testing.assert_allclose(average, 500., rtol=2e-4)
        cases = [(r, z, 3 * pressure, contour, axis, 1., .4, 1e6, 3),
                 (r, z, pressure, contour, axis, 2., .4, 1e6, .5),
                 (r, z, pressure, contour, axis, 1., .4, 2e6, .5),
                 (3 * r, 3 * z, pressure, 3 * contour, 3 * axis, 3., 1.2, 1e6, 3)]
        for *args, factor in cases:
            value, _ = contour_beta_n(*args)
            np.testing.assert_allclose(value, reference * factor, rtol=1e-12)
        invalid = [(r, z, -uniform, contour, axis, 1., .4, 1e6),
                   (r, z, uniform, contour[::-1], axis, 1., .4, 1e6),
                   (r, z, uniform, contour * 2, axis, 1., .4, 1e6),
                   (r, z, uniform, contour[:-1], axis, 1., .4, 1e6),
                   (r, z, uniform * np.nan, contour, axis, 1., .4, 1e6),
                   (r, z, uniform, contour, axis, 0., .4, 1e6)]
        for args in invalid:
            try:
                contour_beta_n(*args)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid pressure contour accepted')
        if time.monotonic() - started > plan['hard_local_cpu_wall_seconds']:
            raise TimeoutError('004 local analytic cap exceeded')
        result.update(complete=True, checks_passed=True, rejected_controls=len(invalid),
                      source_sha256={str(p): digest(p) for p in
                          [Path('participant/mast_pressure_contour.py'), Path(__file__).relative_to(Path.cwd())]})
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic() - started
        write_json(output, result)


if __name__ == '__main__':
    main()
