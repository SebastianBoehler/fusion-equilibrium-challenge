"""Induced toroidal currents in passive conductors from measured current histories.

Passive conductors are toroidal filaments k with resistance R_k. With mutual inductances
M_ij = 2 pi G(i, j) (G = poloidal flux per radian per ampere, participant.mast_green.greens)
and self inductance mu0 R (ln(8R/a) - 7/4), the filament currents obey

    M dI/dt + R I = -d(Psi_drive)/dt,   Psi_drive = 2 pi sum_c G(c, k) I_c   (coils + plasma)

integrated with backward Euler on the measured 5 kHz magnetics time base. The induced flux
on the 65x65 grid is then sum_k I_k G(k, grid). No equilibrium data is used.
"""

import numpy as np

from participant.mast_green import MU0, greens


def inductance(r, z, radius):
    """Mutual/self inductance matrix (H) of toroidal filaments."""
    n = len(r)
    m = np.zeros((n, n))
    for i in range(n):
        others = np.arange(n) != i
        m[i, others] = 2 * np.pi * greens(r[others], z[others], r[i], z[i])
        m[i, i] = MU0 * r[i] * (np.log(8 * r[i] / radius[i]) - 1.75)
    return m


def flux_response(src_r, src_z, dst_r, dst_z):
    """Poloidal flux per radian at destinations per ampere in each source (dst x src)."""
    return greens(src_r[None, :], src_z[None, :], dst_r[:, None], dst_z[:, None])


def induced_currents(times, drive_flux, m, resistance, sample_times):
    """Backward-Euler solution of M dI/dt + R I = -dPsi/dt (Psi = 2 pi * drive_flux).

    times: (N,) s, drive_flux: (N, K) Wb/rad at each filament from all active currents.
    Returns (len(sample_times), K) filament currents in A.
    """
    psi = 2 * np.pi * drive_flux
    k = m.shape[0]
    current = np.zeros(k)
    out = np.zeros((len(sample_times), k))
    order = np.argsort(sample_times)
    j = 0
    cache = {}
    for step in range(1, len(times)):
        dt = times[step] - times[step - 1]
        if dt <= 0:
            continue
        key = round(dt, 9)
        if key not in cache:
            cache[key] = np.linalg.inv(m / dt + np.diag(resistance))
        rhs = m @ current / dt - (psi[step] - psi[step - 1]) / dt
        current = cache[key] @ rhs
        while j < len(order) and sample_times[order[j]] <= times[step]:
            out[order[j]] = current
            j += 1
    while j < len(order):
        out[order[j]] = current
        j += 1
    return out


def wall_filaments(r_in, r_out, z_half, thickness, resistivity, spacing):
    """Filaments along a rectangular vessel cross-section (outer wall, lids, inner wall)."""
    pts = []
    for z in np.arange(-z_half, z_half + 1e-9, spacing):        # outer cylinder
        pts.append((r_out, z, spacing))
    for z in np.arange(-z_half, z_half + 1e-9, spacing):        # inner (centre-column) can
        pts.append((r_in, z, spacing))
    for r in np.arange(r_in + spacing, r_out - 1e-9, spacing):  # top and bottom lids
        pts.append((r, z_half, spacing))
        pts.append((r, -z_half, spacing))
    r, z, width = (np.array(v, float) for v in zip(*pts))
    resistance = resistivity * 2 * np.pi * r / (width * thickness)
    return r, z, resistance, np.maximum(thickness, width) / 4
