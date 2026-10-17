"""Analytic field-line winding and real input-only surface extraction checks."""

from pathlib import Path

import numpy as np

from participant.mast_green import predict
from participant.mast_surface import predict_q95, safety_factor


def main():
    r, z = np.linspace(.2, 1.8, 65), np.linspace(-.8, .8, 65)
    rr, zz = np.meshgrid(r, z)
    psi = -(rr - 1) ** 2 - zz ** 2
    theta = np.linspace(0, 2 * np.pi, 4097)
    minor = .3
    contour = np.column_stack((1 + minor * np.cos(theta), minor * np.sin(theta)))
    expected = .4 / (2 * np.sqrt(1 - minor ** 2))
    q = safety_factor(psi, r, z, contour, .4)
    np.testing.assert_allclose(q, expected, rtol=1e-6)
    np.testing.assert_allclose(safety_factor(2 * psi, r, z, contour, .8), q, rtol=1e-12)
    np.testing.assert_allclose(safety_factor(-psi, r, z, contour, .4), q, rtol=1e-12)
    try:
        safety_factor(np.zeros_like(psi), r, z, contour, .4)
    except ValueError:
        pass
    else:
        raise AssertionError('Degenerate fields must raise an explicit error')
    paths = sorted(Path('run_data/mast-001/inputs').glob('*.npz'))[:20]
    if len(paths) != 20:
        raise RuntimeError('Twenty real input-only MAST shots are required')
    frames = 0
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as inputs:
            row = dict(inputs)
        prediction = predict(row)
        q95 = predict_q95(row, prediction['psirz'])
        assert q95.shape == prediction['q95'].shape and q95.dtype == np.float32
        assert np.isfinite(q95).all() and np.all(q95 > 0)
        if index == 0:
            contaminated = dict(row, efit_q95=np.full(len(q95), np.nan),
                                efit_psirz=np.full_like(prediction['psirz'], np.nan))
            np.testing.assert_array_equal(
                predict_q95(contaminated, prediction['psirz']), q95)
        frames += len(q95)
    print(f'Passed analytic winding, flux/field units, degenerate-field errors, '
          f'and target isolation on 20 real shots / {frames} frames; no MAST accuracy measured.')


if __name__ == '__main__':
    main()
