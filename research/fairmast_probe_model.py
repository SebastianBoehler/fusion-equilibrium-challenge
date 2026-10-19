"""Pickup-coil validation with exact external fields and vacuum-phase probe calibration.

Predicted field at each probe = external part (PF coils + solenoid at sol_scale + passive
filaments), evaluated exactly from Green's-function derivatives at the probe position, plus
the plasma part from the solved flux minus the external flux on the grid (smooth near the
probes). B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR, psi in Wb/rad.

Calibration (instrument only, no plasma, no labels): in the pre-breakdown window
(-80..-5 ms, |Ip| < 5 kA, coils energized) each probe is fitted as
measured = gain * predicted_vacuum + offset. Gains near +1 validate the coil model and the
probe orientation; the offsets absorb integrator baselines. Plasma frames are then compared
after applying the inverse calibration. Probes with too little vacuum signal are skipped.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from participant.mast_green import greens
from participant.mast_inputs import SIGNALS
from research.fairmast_harness import external, r2
from research.mast_passive import geometry_filaments, passive_flux

H = 1e-4  # m, finite-difference step for the field


def _psi_points(rc, zc, w, h, R, Z):
    """Flux per radian per ampere at points (P,) from coil elements with 4-point quadrature."""
    out = np.zeros(len(R))
    for a in (-.25, .25):
        for b in (-.25, .25):
            out += greens((rc + a * w)[None, :], (zc + b * h)[None, :], R[:, None], Z[:, None]).sum(axis=1) / 4
    return out


def _field(psi_fn, R, Z):
    """(B_R, B_Z) per unit source from a flux function by central differences."""
    br = -(psi_fn(R, Z + H) - psi_fn(R, Z - H)) / (2 * H) / R
    bz = (psi_fn(R + H, Z) - psi_fn(R - H, Z)) / (2 * H) / R
    return br, bz


def coil_fields(row, sol_scale, R, Z, times_ms):
    """(B_R, B_Z) of all PF coils and the solenoid at probes, shape (T, P)."""
    names = np.asarray(row["coil_input_column"]).astype(str)
    t = np.asarray(row["magnetics_time"], float)
    br, bz = np.zeros((len(times_ms), len(R))), np.zeros((len(times_ms), len(R)))
    for name in np.unique(names):
        sel = names == name
        fn = lambda rr, zz: _psi_points(row["coil_R"][sel], row["coil_Z"][sel], row["coil_width"][sel],
                                        row["coil_height"][sel], rr, zz)
        kr, kz = _field(fn, R, Z)
        scale = sol_scale if name == "magnetics_sol_current" else 1 / sel.sum()
        cur = np.interp(times_ms, t, np.asarray(row[name], float)) * 1000 * scale
        br += np.outer(cur, kr)
        bz += np.outer(cur, kz)
    return br, bz


def passive_fields(row, groups, sol_scale, R, Z, times_ms):
    if not groups:
        return 0.0, 0.0
    fr, fz, _, _ = geometry_filaments(groups)
    _, currents = passive_flux(dict(row, efit_times=np.asarray(times_ms, np.float32)), groups, sol_scale=sol_scale)
    fn = lambda rr, zz: greens(fr[None, :], fz[None, :], rr[:, None], zz[:, None])  # (P, K)
    kr = -(fn(R, Z + H) - fn(R, Z - H)) / (2 * H) / R[:, None]
    kz = (fn(R + H, Z) - fn(R - H, Z)) / (2 * H) / R[:, None]
    return currents @ kr.T, currents @ kz.T


def probe_geometry(pr, arr):
    geo = [g.lower() for g in pr[f"{arr}_geometry_channel"]]
    idx = [geo.index(c.lower()) for c in pr[f"{arr}_channel"]]
    return pr[f"{arr}_r"][idx].astype(float), pr[f"{arr}_z"][idx].astype(float)


def evaluate(run, data=Path("run_data/fairmast/data"), probes=Path("run_data/fairmast/probes"), shots_limit=None):
    shots = json.loads((data / "data.json").read_text())["shots"]
    res = {a: {"pred": [], "meas": [], "gain": []} for a in ("ccbv", "obv", "obr")}
    for i, e in enumerate(shots[:shots_limit]):
        f, pf = run / "shots" / f"{i:04d}.npz", probes / f"{e['shot']}.npz"
        if not (f.exists() and pf.exists()):
            continue
        stored = np.load(f)
        cfg = json.loads(str(stored["meta"]))["config"]
        sol, groups = cfg.get("sol_scale", 0.5), cfg.get("passive")
        name = Path(e["path"]).stem
        row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
        frames = np.load(data / "magnetics" / f"{name}.npz")["frames_ms"].astype(float)
        pr = np.load(pf)
        tp = pr["time"] * 1000
        ip = np.interp(tp, row["magnetics_time"], row["magnetics_plasma_current"])
        vac = tp[(tp > -80) & (tp < -5) & (np.abs(ip) < 5)]
        r, z = row["efit_grid_R"], row["efit_grid_Z"]
        plasma = stored["psirz"].astype(float) - external(row, cfg, r, z)
        dpz, dpr = np.gradient(plasma, z, r, axis=(1, 2))
        for arr, comp in (("ccbv", 1), ("obv", 1), ("obr", 0)):
            R, Z = probe_geometry(pr, arr)
            rows, cols = (Z - z[0]) / (z[1] - z[0]), (R - r[0]) / (r[1] - r[0])
            g = dpr if comp else dpz
            bp = np.array([(1 if comp else -1) * map_coordinates(x, [rows, cols], order=1) / R for x in g])
            ext_f = [a + b for a, b in zip(coil_fields(row, sol, R, Z, frames), passive_fields(row, groups, sol, R, Z, frames))]
            pred = ext_f[comp] + bp
            meas = np.array([np.interp(frames, tp, ch) for ch in pr[f"{arr}_field"]]).T
            if len(vac) < 20:
                continue
            vac_f = [a + b for a, b in zip(coil_fields(row, sol, R, Z, vac), passive_fields(row, groups, sol, R, Z, vac))]
            mv = np.array([np.interp(vac, tp, ch) for ch in pr[f"{arr}_field"]]).T
            cal = np.full(meas.shape, np.nan)
            gains = np.full(len(R), np.nan)
            for j in range(len(R)):
                x, y = vac_f[comp][:, j], mv[:, j]
                ok = np.isfinite(y)
                if ok.sum() < 20 or np.std(x[ok]) < 1e-3:
                    continue
                gain, off = np.polyfit(x[ok], y[ok], 1)
                if abs(gain) > 0.2:
                    gains[j] = gain
                    cal[:, j] = (meas[:, j] - off) / gain
            res[arr]["pred"].append(pred); res[arr]["meas"].append(cal); res[arr]["gain"].append(gains)
    report = {}
    for arr, v in res.items():
        if not v["pred"]:
            continue
        p, m, g = (np.concatenate(v[k]) for k in ("pred", "meas", "gain"))
        per = [r2(p[:, j], m[:, j]) for j in range(p.shape[1]) if np.isfinite(m[:, j]).sum() > 50]
        report[arr] = {"r2_mean": float(np.nanmean(per)) if per else np.nan, "r2_median": float(np.nanmedian(per)) if per else np.nan,
                       "probes": len(per), "gain_p10_50_90": np.round(np.nanpercentile(g, [10, 50, 90]), 3).tolist()}
    return report


if __name__ == "__main__":
    import sys
    for run in sys.argv[1:]:
        print(Path(run).name, json.dumps(evaluate(Path(run))))
