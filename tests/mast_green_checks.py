"""Physical units, coil multiplicity, and input-only MAST inference checks."""

from pathlib import Path

import numpy as np

from participant.mast_green import MU0, coil_kernel, greens, plasma_kernel, predict


def main():
    rc, zc, r, z = .8, -.4, 1.2, .6
    reference = greens(rc, zc, r, z)
    np.testing.assert_allclose(greens(r, z, rc, zc), reference, rtol=1e-13)
    np.testing.assert_allclose(greens(rc, -zc, r, -z), reference, rtol=1e-13)
    np.testing.assert_allclose(greens(3 * rc, 3 * zc, 3 * r, 3 * z),
                               3 * reference, rtol=1e-13)
    dipole = MU0 * rc ** 2 * r ** 2 / (4 * (r ** 2 + 100 ** 2) ** 1.5)
    np.testing.assert_allclose(greens(rc, 0, r, 100), dipole, rtol=2e-4)
    try:
        greens(rc, zc, rc, zc)
    except ValueError:
        pass
    else:
        raise AssertionError('A singular filament must raise an explicit error')

    kernel, sr, sz = plasma_kernel(tuple(np.linspace(.06, 2, 65)),
                                  tuple(np.linspace(-2, 2, 65)))
    current = np.exp(-((sr - .8) ** 2 + sz ** 2) / .18)
    psi = (kernel @ current).reshape(65, 65)
    np.testing.assert_allclose(psi, psi[::-1], rtol=1e-12, atol=1e-20)

    paths = sorted(Path('run_data/mast-001/inputs').glob('*.npz'))
    if len(paths) < 20:
        raise RuntimeError('Twenty real MAST input-only shots are required')
    frames = 0
    for path in paths[:20]:
        row = dict(np.load(path, allow_pickle=False))
        result = predict(row)
        n = len(row['efit_times'])
        for key, shape in (('psirz', (n, 65, 65)), ('q95', (n,)), ('betaN', (n,))):
            values = result[key]
            assert values.shape == shape and values.dtype == np.float32
            assert np.isfinite(values).all()
        if path == paths[0]:
            original = dict(coil_kernel(row))
            doubled = dict(row)
            for name in ('coil_input_column', 'coil_R', 'coil_Z', 'coil_width', 'coil_height'):
                doubled[name] = np.tile(row[name], 2)
            repeated = dict(coil_kernel(doubled))
            from participant.mast_inputs import SIGNALS
            for index in original:
                multiplier = 2 if SIGNALS[index] == 'sol_current' else 1
                np.testing.assert_allclose(repeated[index], multiplier * original[index],
                                           rtol=1e-12, atol=1e-15)
            # Unreleased targets cannot influence this predictor even if supplied.
            contaminated = dict(row, efit_psirz=np.full((n, 65, 65), np.nan),
                                efit_q95=np.full(n, np.nan), magnetics_dsep=np.full(n, np.nan))
            for key, value in predict(contaminated).items():
                np.testing.assert_array_equal(value, result[key])
        frames += n
    print(f'Passed physical identities, current units, label isolation, and '
          f'20 real input-only shots / {frames} frames; accuracy not measured.')


if __name__ == '__main__':
    main()
