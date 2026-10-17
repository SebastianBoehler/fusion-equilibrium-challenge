"""Predeclared006 analytic interpolation checks; no MAST data reader."""

import json
from pathlib import Path
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_pressure_profile import electron_pressure_on_flux


ROOT = Path('run_data/mast-006')


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    for name, expected in plan['frozen_source_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen pressure source changed: {name}')
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'functional-check.json'
    result = {'complete': False, 'MAST_data_read': False, 'GPU_count': 0,
              'optimizer_steps': 0, 'candidate_predictions_generated': False,
              'plan_sha256': digest(ROOT / 'predeclared-plan.json')}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        flux = np.array([0., .07, .23, .51, .79, 1.])
        query = np.linspace(0, 1, 1001)
        uniform = np.full_like(flux, 700.)
        np.testing.assert_allclose(electron_pressure_on_flux(flux, uniform, query),
                                   700., rtol=1e-12)
        pressure = 1200. - 600. * flux
        mapped = electron_pressure_on_flux(flux, pressure, query)
        np.testing.assert_allclose(mapped, 1200. - 600. * query, rtol=1e-12)
        np.testing.assert_allclose(electron_pressure_on_flux(flux, pressure, flux),
                                   pressure, rtol=1e-12)
        np.testing.assert_allclose(electron_pressure_on_flux(flux, 3 * pressure, query),
                                   3 * mapped, rtol=1e-12)
        nonmonotone = np.array([10., 60., 20., 70., 0., 15.])
        shaped = electron_pressure_on_flux(flux, nonmonotone, query)
        if shaped.min() < nonmonotone.min() or shaped.max() > nonmonotone.max():
            raise ValueError('Electron-pressure interpolation overshot measured range')
        invalid = [(np.array([0., .2, .2]), np.ones(3), query),
                   (flux[::-1], pressure, query),
                   (flux - .1, pressure, query),
                   (flux, -pressure, query),
                   (flux, pressure * np.nan, query),
                   (flux, pressure[:-1], query),
                   (flux, pressure, np.array([1.1])),
                   (flux, pressure, np.array([])),
                   (flux.astype(complex), pressure, query)]
        for args in invalid:
            try:
                electron_pressure_on_flux(*args)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid electron-pressure support accepted')
        for name, expected in plan['frozen_source_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Frozen pressure source changed: {name}')
        if time.monotonic() - started > plan['hard_local_cpu_wall_seconds']:
            raise TimeoutError('006 analytic CPU cap exceeded')
        result.update(complete=True, checks_passed=True, query_points=len(query),
                      rejected_controls=len(invalid), frozen_sources_unchanged=True,
                      maximum_affine_absolute_error=float(abs(mapped - (1200 - 600 * query)).max()),
                      source_sha256={str(p): digest(p) for p in
                          [Path('participant/mast_pressure_profile.py'),
                           Path(__file__).relative_to(Path.cwd())]})
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds'] = time.monotonic() - started
        write_json(output, result)


if __name__ == '__main__':
    main()
