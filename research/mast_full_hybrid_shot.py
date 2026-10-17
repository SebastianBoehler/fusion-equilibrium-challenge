"""Per-shot full-audit computation using frozen physical q and current oracles."""
import numpy as np

from participant.mast_green import MU0, TF_TURNS
from participant.mast_inputs import SIGNALS
from participant.mast_surface import flux_surface, safety_factor
from research.mast_ampere_closure import expected_currents, rectangle_current
from research.mast_vertical_force import baseline_current
from research.mast_hybrid_q95 import gradient, integrate, external_grid, external_gradient
from research.mast_full_hybrid_runtime import array_hash, require


def audit(row, psi, oldq, beta, kernels, units, plan, deadline, emit):
    n = len(row['efit_times']); r, z = row['efit_grid_R'], row['efit_grid_Z']
    require(psi.shape == (n, 65, 65) and psi.dtype == np.float32
        and oldq.shape == beta.shape == (n,) and oldq.dtype == beta.dtype == np.float32
        and all(np.isfinite(a).all() for a in (psi, oldq, beta)), '016stored prediction schema changed')
    before = [array_hash(a) for a in (psi, beta)]
    currents, weights, _, _, center, _ = baseline_current(row)
    expected, scale, components = expected_currents(row, currents, weights)
    del weights
    direct_current = np.zeros(n)
    for name, group in units['groups'].items():
        require(group['direct_exact_passed'] and group['paired_passed'], '016unit gate not passed')
        direct_current += currents[:, SIGNALS.index(name.removeprefix('magnetics_'))]*1000*group['direct_orders_h_h2_A'][-1][-1]
    conductors = np.column_stack([np.interp(row['coil_Z'], z, np.arange(len(z))),
                                  np.interp(row['coil_R'], r, np.arange(len(r)))])
    fvac = abs(currents[:, SIGNALS.index('tf_current')])*1000*MU0*TF_TURNS/(2*np.pi)
    qvalues = np.empty(n, dtype=np.float32); failures = 0
    for frame in range(n):
        deadline(); contour = flux_surface(psi[frame], r, z, center[frame], conductors)
        mid = (contour[1:]+contour[:-1])/2
        nativeq = safety_factor(psi[frame], r, z, contour, fvac[frame])
        ext = external_grid(row, currents[frame], kernels)
        residual = psi[frame].astype(float)-ext
        plasma = gradient(residual, r, z, mid); native = gradient(psi[frame], r, z, mid)
        reconstructed = plasma+gradient(ext, r, z, mid)
        error = float(np.max(abs(reconstructed-native)))
        native_exact = np.float32(nativeq).tobytes() == oldq[frame].tobytes()
        reconstruction_ok = error <= 1e-10+1e-10*float(np.max(abs(native)))
        direct = external_gradient(row, currents[frame], mid, plan['radial_h_m'], deadline)
        qs = np.array([integrate(contour, plasma+v, fvac[frame]) for v in direct])
        qvalues[frame] = qs[1]
        require(np.isfinite(qvalues[frame]) and qvalues[frame]>0, '016float32 q export invalid')
        paired = float(abs(qs[0]/qs[1]-1))
        circulation = np.array([rectangle_current(residual, r, z, order) for order in (4, 8)])
        delta = float(abs(circulation[0]-circulation[1]))
        current = float(circulation[1]+direct_current[frame])
        current_error = float((current-expected[frame])/scale[frame])
        convergence_ok = delta/scale[frame] <= 1e-4 or delta <= .001
        gates = {'native_bit_equal': bool(native_exact), 'spline_reconstruction': reconstruction_ok,
                 'paired_q95': paired <= plan['gates']['direct_h_h2_q_relative'],
                 'paired_current': bool(convergence_ok),
                 'current_closure': abs(current_error) <= plan['gates']['full_actual_outer_current_closure_relative']}
        failures += not all(gates.values())
        emit({'index': frame, 'time_ms': float(row['efit_times'][frame]), 'gates': gates,
              'stored_q95': float(oldq[frame]), 'native_q95': nativeq, 'hybrid_h_h2_q95': qs.tolist(),
              'export_float32_q95': float(qvalues[frame]), 'paired_q_relative': paired,
              'spline_gradient_max_error': error, 'contour_sha256': array_hash(contour),
              'segments': len(mid), 'plasma_current_orders4_8_A': circulation.tolist(),
              'direct_external_current_A': float(direct_current[frame]),
              'hybrid_current_A': current, 'expected_current_A': float(expected[frame]),
              'expected_components_A': {k:float(v[frame]) for k,v in components.items()},
              'scale_current_A': float(scale[frame]), 'current_normalized_error': current_error})
        require(native_exact and reconstruction_ok, '016native/decomposition control failed')
    require(before == [array_hash(a) for a in (psi, beta)], '016frozen field/beta mutated')
    return qvalues, {'frames': n, 'failed_frames': int(failures), 'frozen_psi_betaN_sha256': before}
