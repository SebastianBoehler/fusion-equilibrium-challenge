"""MAST 024: q95 and betaN from kinetic-GS flux surfaces, validated on the DIII-D proxy.

q95 = F/(2 pi) * contour integral dl / (R |grad psi|) at psiN = 0.95, with F the vacuum
toroidal-field function from the measured TF current. DIII-D TF: 24 coils x 6 turns = 144.
Labels (efit_q95) are diagnostic targets only.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from skimage.measure import find_contours, points_in_poly

from participant.data import SIGNALS as D3D_SIGNALS, features
from participant.mast_green import MU0
from participant.mast_surface import safety_factor
from research.betan import (beta_n_electron, beta_n_equilibrium, beta_n_from_flux,
                            electron_pressure)
from research.gs_picard import find_axis, lcfs_level
from research.thomson_points import d3d_points
from research.gs_proxy_d3d import Profile, region, run_shot

D3D_TF_TURNS = 144


def q_at(phi, r, z, allowed, fpol, fraction=0.95):
    """q on the psiN = fraction surface of an axis-maximum flux map."""
    axis = find_axis(phi, allowed, (len(z) // 2, len(r) // 2))
    level, _ = lcfs_level(phi, allowed, axis)
    target = phi[axis] + fraction * (level - phi[axis])
    for c in find_contours(phi, target):
        if len(c) > 8 and np.allclose(c[0], c[-1]) and points_in_poly(
                np.array([axis], float), c)[0]:
            xy = np.column_stack((np.interp(c[:, 1], np.arange(len(r)), r),
                                  np.interp(c[:, 0], np.arange(len(z)), z)))
            return safety_factor(phi, r, z, xy, fpol)
    raise ValueError("No closed psiN surface around the axis")


def shot_q(path):
    import pyarrow.parquet as pq
    from participant.data import array_value
    res = run_shot((path, Profile(beta0=0.8, iterations=60, te_power=1.0), 10, "axis", True))
    table = pq.read_table(path, columns=["efit_times", "efit_grid_R", "efit_grid_Z",
                                         "magnetics_time", "magnetics_plasma_current",
                                         "magnetics_plasma_current_times",
                                         *(f"magnetics_{s}" for s in D3D_SIGNALS
                                           if s != "plasma_current"),
                                         "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
                                         *(f"thomson_{a}_{b}" for a in ("core", "edge")
                                           for b in ("times", "Te", "ne"))])
    row = {n: array_value(table[n][0]) for n in table.column_names}
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    bcoil = np.interp(np.asarray(row["efit_times"], float),
                      np.asarray(row["magnetics_time"], float),
                      np.asarray(row["magnetics_bcoil"], float))[::10]
    fvac = np.abs(bcoil) * 1000 * MU0 * D3D_TF_TURNS / (2 * np.pi)
    allowed = region(r, z, 1.0)
    ip = np.abs(features(row)[:, D3D_SIGNALS.index("plasma_current")][::10]) * 1000
    pr, pz, pe = electron_pressure(d3d_points(row, "Te"), d3d_points(row, "ne"))
    pe = pe[::10]
    out = []
    for t, pred in enumerate(res["pred"]):
        border = np.concatenate((pred[2], pred[-3], pred[:, 2], pred[:, -3]))
        phi = pred if pred[32, 32] > np.median(border) else -pred
        item = [res["row"]["efit_q95"][t], np.nan, res["row"]["efit_beta_n"][t], np.nan,
                res["row"]["efit_beta_n"][t], np.nan, res["row"]["efit_beta_n"][t], np.nan]
        if res["solved"][t] is not None:
            sphi, cells, info = res["solved"][t]
            try:
                item[5] = beta_n_equilibrium(sphi, cells, r, z, allowed, info["beta0"],
                                             info["r0"], fvac[t], ip[t])
            except ValueError:
                pass
            try:
                item[7] = beta_n_from_flux(sphi, r, z, allowed, fvac[t], ip[t])
            except ValueError:
                pass
        try:
            item[1] = abs(q_at(phi, r, z, allowed, fvac[t]))
        except ValueError:
            pass
        try:
            ok = np.isfinite(pe[t])
            item[3] = beta_n_electron(phi, r, z, allowed, (pr[ok], pz[ok], pe[t][ok]),
                                      fvac[t], ip[t])
        except ValueError:
            pass
        out.append(item)
    return out


def main() -> None:
    paths = sorted(Path("run_data/full/raw/data/diii_d_train").glob("*.parquet"))
    with ProcessPoolExecutor(2) as pool:
        pairs = np.array([x for chunk in pool.map(shot_q, paths) for x in chunk], float)
    report = {"frames": int(len(pairs))}
    for name, j in (("q95", 0), ("betaN_electron", 2), ("betaN_equilibrium", 4),
                    ("betaN_from_flux", 6)):
        ok = np.isfinite(pairs[:, j:j + 2]).all(axis=1)
        t, p = pairs[ok, j], pairs[ok, j + 1]
        ratio = float(np.median(p / t))
        report[name] = {"valid": int(ok.sum()),
                        "r2": float(1 - np.square(t - p).sum() / np.square(t - t.mean()).sum()),
                        "corr": float(np.corrcoef(t, p)[0, 1]), "ratio_median": ratio,
                        "r2_after_median_ratio": float(1 - np.square(t - p / ratio).sum()
                                                       / np.square(t - t.mean()).sum())}
    Path("run_data/mast-024").mkdir(exist_ok=True)
    Path("run_data/mast-024/q95-betan-proxy.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
