"""Magnetics test-time adaptation (TTA): per-frame blend of profile variants + vertical shift.

For one shot, K solver runs that share the external field (same passive/solenoid config) give
plasma flux maps P_k = psi_k - ext. Per frame we choose convex weights w (w >= 0, sum 1) and a
rigid vertical shift dz of the blended plasma part, minimising the normalised residual to the
measured plasma contribution on a chosen set of magnetic sensors:
    cost = sum_j ((pred_j - meas_j) / s_j)^2 + LAM * |w - e_base|^2 + (dz / DZ_SIG)^2
Output psi = ext + shift(sum_k w_k P_k, dz). Hyperparameters are fixed a priori (22 Oct).
Sensors: flux loops (psi, Wb), CCBV/OBV (Bz) and OBR (BR) pickup coils (vacuum-calibrated as
in research.fairmast_probe_model). Magnetics are public FAIR-MAST diagnostics; never equilibrium.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, shift
from scipy.optimize import nnls

from research.fairmast_harness import external
from research.fairmast_probe_model import coil_fields, passive_fields, probe_geometry

LAM, DZ_SIG = 1.0, 0.03
SHIFTS = np.round(np.arange(-0.08, 0.0801, 0.01), 3)
PROBES = (("ccbv", 1), ("obv", 1), ("obr", 0))


def sensors(row, mag, pr, cfg, frames_ms):
    """{group: dict(kind, rows, cols, R, meas_plasma (T,n))}; groups 'CC', 'P' loops and probes."""
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    ext = external(row, cfg, r, z)
    t = frames_ms / 1000
    out = {}
    geo = list(mag["flux_loop_geometry_channel"])
    names = list(mag["flux_loop_channel"])
    idx = [geo.index(c) for c in names]
    fr, fz = mag["flux_loop_r"][idx].astype(float), mag["flux_loop_z"][idx].astype(float)
    rows, cols = (fz - z[0]) / (z[1] - z[0]), (fr - r[0]) / (r[1] - r[0])
    ext_l = np.array([map_coordinates(m, [rows, cols], order=1) for m in ext]) * 2 * np.pi
    meas = np.array([np.interp(t, mag["mag_time"], ch) for ch in mag["flux_loop_flux"]]).T
    bad = ~np.isfinite(meas).all(axis=0) | (np.nanstd(meas, axis=0) < 1e-4)
    bad |= np.array([n.upper().startswith("CC04") for n in names])
    for grp in ("CC", "P"):
        sel = np.array([n.upper().startswith(grp) for n in names]) & ~bad
        if sel.sum():
            out[grp] = dict(kind="psi", rows=rows[sel], cols=cols[sel], R=fr[sel], meas=(meas - ext_l)[:, sel],
                            names=np.array(names)[sel])
    if pr is None:
        return out, ext
    tp = pr["time"] * 1000
    ip = np.interp(tp, row["magnetics_time"], row["magnetics_plasma_current"])
    vac = tp[(tp > -80) & (tp < -5) & (np.abs(ip) < 5)]
    if len(vac) < 20:
        return out, ext
    sol, groups = cfg.get("sol_scale", 0.5), cfg.get("passive")
    for arr, comp in PROBES:
        R, Z = probe_geometry(pr, arr)
        ext_f = [a + b for a, b in zip(coil_fields(row, sol, R, Z, frames_ms), passive_fields(row, groups, sol, R, Z, frames_ms))]
        vac_f = [a + b for a, b in zip(coil_fields(row, sol, R, Z, vac), passive_fields(row, groups, sol, R, Z, vac))]
        mv = np.array([np.interp(vac, tp, ch) for ch in pr[f"{arr}_field"]]).T
        m = np.array([np.interp(frames_ms, tp, ch) for ch in pr[f"{arr}_field"]]).T
        cal = np.full(m.shape, np.nan)
        for j in range(len(R)):
            x, y = vac_f[comp][:, j], mv[:, j]
            ok = np.isfinite(y)
            if ok.sum() < 20 or np.std(x[ok]) < 1e-3:
                continue
            gain, off = np.polyfit(x[ok], y[ok], 1)
            if abs(gain) > 0.2:
                cal[:, j] = (m[:, j] - off) / gain
        keep = np.isfinite(cal).all(axis=0)
        if keep.sum():
            out[arr] = dict(kind="bz" if comp else "br", rows=((Z - z[0]) / (z[1] - z[0]))[keep],
                            cols=((R - r[0]) / (r[1] - r[0]))[keep], R=R[keep], meas=(cal - ext_f[comp])[:, keep],
                            names=np.array([f"{arr}:{c}" for c in pr[f"{arr}_channel"]])[keep])
    return out, ext


def predict(plasma, r, z, s):
    """Plasma contribution at the sensors of group s from one (65,65) plasma flux map."""
    if s["kind"] == "psi":
        return map_coordinates(plasma, [s["rows"], s["cols"]], order=1) * 2 * np.pi
    dz_, dr_ = np.gradient(plasma, z, r)
    g, sign = (dr_, 1.0) if s["kind"] == "bz" else (dz_, -1.0)
    return sign * map_coordinates(g, [s["rows"], s["cols"]], order=1) / s["R"]


def fit_frame(maps, r, z, groups, fit, scale, base=0):
    """maps: (K,65,65) plasma parts. Returns (w, dz, cost)."""
    K = len(maps)
    e0 = np.eye(K)[base]
    best = (e0, 0.0, np.inf)
    for dz in SHIFTS:
        sh = [shift(m, (dz / (z[1] - z[0]), 0), order=1, mode="nearest") for m in maps]
        A, b = [], []
        for g in fit:
            if g not in groups:
                continue
            P = np.array([predict(m, r, z, groups[g]) for m in sh]).T     # (n, K)
            A.append(P / scale[g][:, None]); b.append(groups[g]["meas_t"] / scale[g])
        if not A:
            return e0, 0.0, np.nan
        A, b = np.vstack(A), np.concatenate(b)
        ok = np.isfinite(b)
        A, b = A[ok], b[ok]
        big = 1e3
        Aa = np.vstack([A, np.sqrt(LAM) * np.eye(K), big * np.ones((1, K))])
        ba = np.concatenate([b, np.sqrt(LAM) * e0, [big]])
        w, _ = nnls(Aa, ba)
        cost = float(np.sum((A @ w - b) ** 2) + LAM * np.sum((w - e0) ** 2) + (dz / DZ_SIG) ** 2)
        if cost < best[2]:
            best = (w, float(dz), cost)
    return best


def blend(maps, w, dz, z):
    m = np.tensordot(w, maps, axes=1)
    return shift(m, (dz / (z[1] - z[0]), 0), order=1, mode="nearest")


def load_maps(runs, i):
    psis = [np.load(Path(run) / "shots" / f"{i:04d}.npz") for run in runs]
    cfg = json.loads(str(psis[0]["meta"]))["config"]
    return np.stack([p["psirz"].astype(float) for p in psis]), cfg


# ---- E3: per-frame vacuum multipole correction (Delta* v = 0), fitted to the residual ----------
R0 = 0.9
VAC_RIDGE = 1.0


def _vac(Rq, Zq):
    """(psi, dpsi/dR, dpsi/dZ) of 5 vacuum multipoles at points; columns (n, 5)."""
    x, y = Rq / R0, Zq / R0
    psi = np.stack([np.ones_like(x), x**2, x**2 * y, x**4 - 4 * x**2 * y**2, 4 * x**2 * y**3 - 3 * x**4 * y], 1)
    dx = np.stack([0 * x, 2 * x, 2 * x * y, 4 * x**3 - 8 * x * y**2, 8 * x * y**3 - 12 * x**3 * y], 1) / R0
    dy = np.stack([0 * x, 0 * x, x**2, -8 * x**2 * y, 12 * x**2 * y**2 - 3 * x**4], 1) / R0
    return psi, dx, dy


def vac_at(s, r, z):
    Rq = np.interp(s["cols"], np.arange(len(r)), r)
    Zq = np.interp(s["rows"], np.arange(len(z)), z)
    psi, dx, dy = _vac(Rq, Zq)
    if s["kind"] == "psi":
        return psi * 2 * np.pi
    return dx / Rq[:, None] if s["kind"] == "bz" else -dy / Rq[:, None]


def vac_fit(resid, groups, fit, scale, r, z):
    """Ridge coefficients c (5,) of the vacuum basis on the fit groups' residual (meas - pred)."""
    A, b = [], []
    for g in fit:
        if g in groups:
            A.append(vac_at(groups[g], r, z) / scale[g][:, None]); b.append(resid[g] / scale[g])
    if not A:
        return np.zeros(5)
    A, b = np.vstack(A), np.concatenate(b)
    ok = np.isfinite(b)
    A, b = A[ok], b[ok]
    norm = np.sqrt((A ** 2).mean(0)) + 1e-12
    An = A / norm
    c = np.linalg.solve(An.T @ An + VAC_RIDGE * np.eye(5), An.T @ b)
    return c / norm


def vac_map(c, r, z):
    RR, ZZ = np.meshgrid(r, z)
    return (_vac(RR.ravel(), ZZ.ravel())[0] @ c).reshape(RR.shape)


# ---- E4: per-frame passive-cluster current correction (localised physical sources) ------------
N_CL, CL_RIDGE, _CL = 16, 1.0, {}


def _clusters():
    if "fr" not in _CL:
        from scipy.cluster.vq import kmeans2
        from research.mast_passive import geometry_filaments
        fr, fz, _, _ = geometry_filaments("ALLgf")
        _, lab = kmeans2(np.column_stack([fr, fz]), N_CL, seed=0, minit="++")
        _CL.update(fr=fr, fz=fz, lab=lab)
    return _CL["fr"], _CL["fz"], _CL["lab"]


def _cl_psi(Rq, Zq):
    """Flux per radian at points (n, N_CL) for 1 A per cluster spread over its filaments."""
    from participant.mast_green import greens
    fr, fz, lab = _clusters()
    g = greens(fr[None, :], fz[None, :], Rq[:, None], Zq[:, None])        # (n, F)
    out = np.zeros((len(Rq), N_CL))
    for k in range(N_CL):
        sel = lab == k
        if sel.any():
            out[:, k] = g[:, sel].mean(1)
    return out


def cl_at(s, r, z, h=1e-4):
    Rq = np.interp(s["cols"], np.arange(len(r)), r)
    Zq = np.interp(s["rows"], np.arange(len(z)), z)
    if s["kind"] == "psi":
        return _cl_psi(Rq, Zq) * 2 * np.pi
    if s["kind"] == "bz":
        return (_cl_psi(Rq + h, Zq) - _cl_psi(Rq - h, Zq)) / (2 * h) / Rq[:, None]
    return -(_cl_psi(Rq, Zq + h) - _cl_psi(Rq, Zq - h)) / (2 * h) / Rq[:, None]


def cl_fit(resid, groups, fit, scale, r, z, cache):
    A, b = [], []
    for g in fit:
        if g in groups:
            if g not in cache:
                cache[g] = cl_at(groups[g], r, z)
            A.append(cache[g] / scale[g][:, None]); b.append(resid[g] / scale[g])
    if not A:
        return np.zeros(N_CL)
    A, b = np.vstack(A), np.concatenate(b)
    ok = np.isfinite(b)
    A, b = A[ok], b[ok]
    norm = np.sqrt((A ** 2).mean(0)) + 1e-12
    An = A / norm
    c = np.linalg.solve(An.T @ An + CL_RIDGE * np.eye(N_CL), An.T @ b)
    return c / norm


def cl_map(c, r, z):
    RR, ZZ = np.meshgrid(r, z)
    return (_cl_psi(RR.ravel(), ZZ.ravel()) @ c).reshape(RR.shape)
