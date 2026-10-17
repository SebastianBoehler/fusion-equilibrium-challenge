"""Independent manufactured references, executed before any MAST cache read."""

import numpy as np

from participant.mast_green import MU0
from research.mast_vertical_force import central_dz, flux_dz, vertical_force


def check_controls(gates):
    rc, zc = 1.7, .4
    r, z = np.array([.7, 1.1, .9]), np.array([.1, -.2, .6])
    exact = flux_dz(rc, zc, r, z)
    theta = (np.arange(gates['manufactured_biot_savart_angular_points']) + .5)
    theta *= 2 * np.pi / len(theta)
    distance = (r[:, None] ** 2 + rc ** 2 - 2 * r[:, None] * rc
                * np.cos(theta) + (z[:, None] - zc) ** 2)
    # Direct Biot-Savart radial field from dl cross displacement, no Green flux.
    br = MU0 / (4 * np.pi) * (2 * np.pi / len(theta)) * np.sum(
        rc * np.cos(theta) * (z[:, None] - zc) / distance ** 1.5, axis=1)
    np.testing.assert_allclose(exact, -r * br, rtol=1e-8, atol=1e-14)
    finite_errors = []
    for h in (gates['derivative_h_m'], gates['derivative_h2_m']):
        numerical = central_dz(rc, zc, r, z, h)
        np.testing.assert_allclose(exact, numerical, rtol=1e-5, atol=1e-12)
        finite_errors.append(float(np.max(np.abs(exact - numerical))))
    weights = np.array([[100., 200., 300.]])
    uniform = np.broadcast_to(-r * .02, weights.shape)
    f, scale, residual = vertical_force(weights, uniform)
    expected = -2 * np.pi * .02 * sum(float(i) * float(x)
                                     for i, x in zip(weights[0], r))
    np.testing.assert_allclose(f, expected, rtol=1e-8, atol=1e-8)
    assert f[0] < 0 and residual[0] == -1
    analytic = exact[None]
    for doubled_weights, doubled_field in ((2 * weights, analytic),
                                           (weights, 2 * analytic)):
        np.testing.assert_allclose(vertical_force(doubled_weights, doubled_field)[0],
                                   2 * vertical_force(weights, analytic)[0],
                                   rtol=1e-8, atol=1e-8)
    mirrored = flux_dz(rc, zc, r, np.zeros(3)) + flux_dz(rc, -zc, r, np.zeros(3))
    mirror_force = 2 * np.pi * np.sum(weights[0] * mirrored)
    assert abs(mirror_force) <= 1e-8
    invalid = [lambda: flux_dz(0., zc, r, z),
               lambda: flux_dz(rc, zc, np.array([np.nan]), zc),
               lambda: flux_dz(rc, zc, rc, zc),
               lambda: vertical_force(-weights, analytic),
               lambda: vertical_force(weights, analytic[:, :2]),
               lambda: vertical_force(weights, np.zeros_like(weights)),
               lambda: vertical_force(weights, np.full_like(weights, np.inf)),
               lambda: vertical_force(np.zeros_like(weights), analytic)]
    for fixture in invalid:
        try:
            fixture()
        except ValueError:
            continue
        raise AssertionError('Invalid manufactured fixture was accepted')
    return {'passed': True, 'independent_biot_savart_max_error_Tm_per_A':
            float(np.max(np.abs(exact + r * br))),
            'central_difference_max_errors_Tm_per_A': finite_errors,
            'uniform_Br_force_N': float(f[0]), 'mirror_force_N': float(mirror_force),
            'linearity_controls': 2, 'invalid_controls_rejected': len(invalid)}
