"""Unrelaxed current profile: 1D cylindrical resistive diffusion driven by measured Ip(t).

Enclosed current I(rho, t) obeys dI/dt = rho/(mu0 a^2) d/drho( eta/rho dI/drho ), with I(0) = 0
and I(1) = Ip(t). eta is Spitzer (Zeff = 2) with a neoclassical trapped-particle factor, from
Thomson Te(rho, t). MAST shots (0.2-0.4 s) are much shorter than the resistive time
(3-9 s), so J != Te^p.

DIII-D check (labels as targets only): does the diffusion l_i track EFIT l_i better than
the relaxed Te^1 profile? Predeclared: the 16 local TRAIN shots in sorted order; gate =
diffusion's median within-shot Pearson r beats Te^1's by >= 0.2, and its pooled r is higher.
Cap: local CPU, under 30 min.
"""

import glob
import json

import numpy as np
import pyarrow.parquet as pq

from research.thomson_points import d3d_points

MU0 = 4e-7 * np.pi
RHO = np.linspace(0.0, 1.0, 41)


def resistivity(te_ev, rho, a, r0, zeff=2.0):
    te_kev = np.clip(te_ev, 10.0, None) / 1e3
    eps = np.clip(rho * a / r0, 0.0, 0.8)
    return 2.8e-8 * zeff / te_kev ** 1.5 / (1 - np.sqrt(eps)) ** 2


def step(current, eta, dt, a):
    """Implicit Euler for dI/dt = rho/(mu0 a^2) d/drho(eta/rho dI/drho), Dirichlet ends."""
    n, h = len(RHO), RHO[1] - RHO[0]
    mid = (RHO[1:] + RHO[:-1]) / 2
    k = (eta[1:] + eta[:-1]) / 2 / mid  # eta/rho on faces
    coef = RHO[1:-1] / (MU0 * a * a * h * h)
    lower, upper = coef * k[:-1], coef * k[1:]
    mat = np.diag(1 + dt * (lower + upper)) - dt * np.diag(upper[:-1], 1) - dt * np.diag(lower[1:], -1)
    rhs = current[1:-1].copy()
    rhs[-1] += dt * upper[-1] * current[-1]
    out = current.copy()
    out[1:-1] = np.linalg.solve(mat, rhs)
    return out


def internal_inductance(current):
    """Cylindrical l_i = 2 int I^2/rho drho / Ip^2."""
    integrand = np.where(RHO > 0, current ** 2 / np.where(RHO > 0, RHO, 1), 0.0)
    return 2 * np.trapezoid(integrand, RHO) / current[-1] ** 2


def relaxed(shape):
    """Enclosed current for j(rho) ~ shape, normalized to 1."""
    cum = np.concatenate(([0.0], np.cumsum((shape[1:] * RHO[1:] + shape[:-1] * RHO[:-1]) / 2 * np.diff(RHO))))
    return cum / cum[-1]


def evolve(t_ip, ip, t_frames, te_rho, a, r0, max_dt=2e-3):
    """Enclosed-current profiles at the frame times. Before the first frame, Te keeps the
    first frame's shape, scaled by Ip(t)/Ip(t0) (floor 0.2). Start: flat j at 5% of peak Ip."""
    start = t_ip[np.argmax(ip > 0.05 * ip.max())]
    current = relaxed(np.ones_like(RHO)) * np.interp(start, t_ip, ip)
    t, out = start, []
    ip0 = np.interp(t_frames[0], t_ip, ip)
    for k, tf in enumerate(t_frames):
        while t < tf - 1e-9:
            dt = min(max_dt, tf - t)
            t += dt
            if t < t_frames[0]:
                te = te_rho[0] * max(np.interp(t, t_ip, ip) / ip0, 0.2)
            else:
                j = min(np.searchsorted(t_frames, t), len(t_frames) - 1)
                te = te_rho[j]
            current[-1] = np.interp(t, t_ip, ip)
            current = step(current, resistivity(te, RHO, a, r0), dt, a)
        out.append(current.copy())
    return np.array(out)


def te_profiles(row):
    """Te on the RHO grid per frame: Thomson mapped by the EFIT flux, axis and LCFS (DIII-D only)."""
    from scipy.ndimage import map_coordinates
    pr, pz, te = d3d_points(row)
    psi = np.asarray(row["efit_psirz"], float)
    r, z = np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)
    to_idx = lambda rr, zz: [(np.asarray(zz) - z[0]) / (z[1] - z[0]), (np.asarray(rr) - r[0]) / (r[1] - r[0])]
    out = np.full((len(psi), len(RHO)), np.nan)
    for k, p in enumerate(psi):
        n = int(row["efit_lcfs_n"][k])
        lr, lz = np.asarray(row["efit_lcfs_r"][k], float)[:n], np.asarray(row["efit_lcfs_z"][k], float)[:n]
        if n < 4:
            continue
        lcfs = np.median(map_coordinates(p, to_idx(lr, lz), order=1))
        ax = map_coordinates(p, to_idx([row["efit_r_axis"][k]], [row["efit_z_axis"][k]]), order=1)[0]
        x = np.sqrt(np.clip((ax - map_coordinates(p, to_idx(pr, pz), order=1)) / (ax - lcfs), 0, None))
        ok = np.isfinite(te[k]) & (te[k] > 0) & (x < 1.0)
        if ok.sum() < 8:
            continue
        order = np.argsort(x[ok])
        out[k] = np.interp(RHO, x[ok][order], te[k][ok][order])
    return out


def d3d_check(limit=16, a=0.78, r0=1.7):
    cols = ["efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_li", "efit_lcfs_n", "efit_lcfs_r",
            "efit_lcfs_z", "efit_r_axis", "efit_z_axis", "magnetics_plasma_current",
            "magnetics_plasma_current_times", "thomson_core_times", "thomson_core_Te", "thomson_core_R",
            "thomson_edge_times", "thomson_edge_Te", "thomson_edge_spatial", "thomson_chord_R", "thomson_chord_Z",
            "thomson_chord_name"]
    files = sorted(glob.glob("run_data/full/raw/data/diii_d_train/*.parquet"))[:limit]
    li_true, li_diff, li_te1, shot_r = [], [], [], {"diffusion": [], "te1": []}
    for f in files:
        names = set(pq.read_schema(f).names)
        for row in pq.read_table(f, columns=[c for c in cols if c in names]).to_pylist():
            t = np.asarray(row["efit_times"], float) / 1e3
            t_ip = np.asarray(row["magnetics_plasma_current_times"], float) / 1e3
            ip = np.abs(np.asarray(row["magnetics_plasma_current"], float))
            te = te_profiles(row)
            ok = np.isfinite(te).all(axis=1) & np.isfinite(np.asarray(row["efit_li"], float))
            if ok.sum() < 10:
                continue
            te_f = te.copy()
            for j in range(len(te_f)):  # nearest valid profile for frames without Thomson
                if not np.isfinite(te_f[j]).all():
                    te_f[j] = te[np.nonzero(np.isfinite(te).all(axis=1))[0][np.abs(np.nonzero(np.isfinite(te).all(axis=1))[0] - j).argmin()]]
            prof = evolve(t_ip, ip, t, te_f, a, r0)
            d = np.array([internal_inductance(c) for c in prof])[ok]
            e = np.array([internal_inductance(relaxed(te_f[j])) for j in range(len(t))])[ok]
            y = np.asarray(row["efit_li"], float)[ok]
            li_true.append(y); li_diff.append(d); li_te1.append(e)
            shot_r["diffusion"].append(np.corrcoef(d, y)[0, 1]); shot_r["te1"].append(np.corrcoef(e, y)[0, 1])
    y, d, e = map(np.concatenate, (li_true, li_diff, li_te1))
    report = {"shots": len(li_true), "frames": int(len(y)),
              "pooled_r": {"diffusion": float(np.corrcoef(d, y)[0, 1]), "te1": float(np.corrcoef(e, y)[0, 1])},
              "median_within_r": {k: float(np.nanmedian(v)) for k, v in shot_r.items()},
              "li_p10_50_90": {"efit": np.percentile(y, [10, 50, 90]).round(3).tolist(),
                               "diffusion": np.percentile(d, [10, 50, 90]).round(3).tolist(),
                               "te1": np.percentile(e, [10, 50, 90]).round(3).tolist()}}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    d3d_check()
