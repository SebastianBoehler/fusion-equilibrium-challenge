"""Manufactured q and additive field-decomposition controls for MAST015."""
import numpy as np
from research.mast_hybrid_q95 import gradient, integrate


def controls(plan):
    r = np.linspace(.4, 1.4, 65); z = np.linspace(-.5, .5, 65)
    rr, zz = np.meshgrid(r, z); r0, radius, c, fpol = .9, .2, .2, .7
    theta = np.linspace(0, 2*np.pi, 2049)
    contour = np.column_stack([r0+radius*np.cos(theta), radius*np.sin(theta)])
    contour[-1] = contour[0]
    mid = (contour[1:]+contour[:-1])/2
    psi = -c*((rr-r0)**2+zz**2)
    external = .07*rr**2+.03*rr*zz+.09*zz**2
    direct = np.stack([.14*mid[:, 0]+.03*mid[:, 1],
                       .03*mid[:, 0]+.18*mid[:, 1]])
    native = gradient(psi, r, z, mid)
    hybrid = gradient(psi-external, r, z, mid)+direct
    error = float(np.max(abs(native-hybrid)))
    if error > plan['gates']['manufactured_hybrid_gradient_absolute']:
        raise ValueError('Manufactured hybrid gradient/sign failed')
    q = integrate(contour, hybrid, fpol)
    exact = fpol/(2*c*np.sqrt(r0**2-radius**2))
    relative = abs(q/exact-1)
    if relative > plan['gates']['manufactured_circle_q_relative']:
        raise ValueError('Manufactured analytic circular q failed')
    if not np.isclose(integrate(contour, -hybrid, fpol), q, rtol=1e-12):
        raise ValueError('Manufactured simultaneous sign control failed')
    if not np.isclose(integrate(contour, 3*hybrid, fpol), q/3, rtol=1e-12):
        raise ValueError('Manufactured q scale control failed')
    bad = hybrid.copy(); bad[0, 0] = np.nan
    invalid = [(contour, bad, fpol), (contour, hybrid[:, :-1], fpol),
               (contour, hybrid*0, fpol), (contour, hybrid, 0),
               (contour, hybrid, np.nan), (contour*-1, hybrid, fpol)]
    for args in invalid:
        try: integrate(*args)
        except ValueError: pass
        else: raise AssertionError('Invalid q control accepted')
    return {'analytic_q': exact, 'actual_q': q, 'relative_error': relative,
            'hybrid_gradient_max_error': error, 'sign_scale_passed': True,
            'invalid_controls_rejected': len(invalid)}
