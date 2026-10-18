"""MAST 025: robust kinetic equilibria after the 985081 runaway diagnosis.

Changes against research/mast_gs (023):
- Axis anchor: Te median-filtered (9 channels), centroid of the region above 70% of the
  peak; kept only within MAST's axis range 0.75-1.10 m, else the shot median of valid
  anchors, else 0.92 m (MAST machine geometry, not equilibrium data).
- Allowed region additionally limited to R <= 1.6 m and |Z| <= 1.7 m (machine geometry).
- Pressure-share range narrowed to [-0.5, 2.0].
--vertical zero holds the current centroid at Z = 0 (MAST up-down symmetric operation)
instead of the external-force balance. Every frame is checked afterwards (axis R 0.7-1.15 m, |Z| < 0.3 m, closed LCFS); failing
frames keep the candidate-004 flux. Outputs per shot under <out>/shots/.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy import ndimage

from participant.mast_green import MU0, TF_TURNS, coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents
from research.betan import beta_n_equilibrium
from research.gs_picard import Profile, find_axis, lcfs_level, solve
from research.mast_gs import ROOT, SOL_SCALE, allowed_region

# Input directory (data.json + inputs/). Env var so spawned workers inherit it; Final runs
# point it at the private-test cache. The public default keeps scored runs reproducible.
DATA = Path(os.environ.get("MAST_DATA_DIR", ROOT / "mast-001"))
PUBLIC = DATA.resolve() == (ROOT / "mast-001").resolve()
from research.thomson_points import mast_points

PROFILE = Profile(beta0=0.8, iterations=60, te_power=1.0, beta0_range=(-0.5, 2.0))
PROFILES = {"force": PROFILE, "zero": Profile(beta0=0.8, iterations=60, te_power=1.0,
                                               beta0_range=(-0.5, 2.0), vertical="zero"),
            # EFIT++-style basis (literature review 1): J linear in (1 - psiN), no Te shape.
            "zero-linear": Profile(alpha_m=1.0, alpha_n=1.0, beta0=0.8, iterations=60,
                                   beta0_range=(-0.5, 2.0), vertical="zero")}
AXIS_RANGE, AXIS_DEFAULT = (0.75, 1.10), 0.92


def te_axis(pr, te):
    ok = np.isfinite(te)
    if ok.sum() < 15:
        return None
    order = np.argsort(pr[ok])
    x, y = pr[ok][order], ndimage.median_filter(te[ok][order], size=9, mode="nearest")
    top = y >= 0.7 * y.max()
    centre = float((x[top] * y[top]).sum() / y[top].sum())
    return centre if AXIS_RANGE[0] <= centre <= AXIS_RANGE[1] else None


R_MIN = {"default": 0.19, "inner": 0.175}


def region(row, inner="default"):
    """Allowed plasma region; "inner" moves the centre-column limit to just outside the
    solenoid conductors (0.175 m) so diverted plasmas are not wall-limited artificially."""
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    return (allowed_region(row, R_MIN[inner], 0.06 if inner == "default" else 0.02)
            & (r[None, :] <= 1.6)
            & (np.abs(z)[:, None] <= 1.7))


def efps_response(row, mode):
    """Grid flux per kA of efps_current flowing through a subset of P2 turns (both coils).

    efps is documented as a P2 link-board feed current (forum, 23 Sep). The winding it feeds
    is not documented, so `mode` selects a hypothesis: all P2 turns, inner (R < 0.5 m) or
    outer (R > 0.5 m) section. Each selected element carries efps (one turn each).
    """
    from participant.mast_green import greens
    r, z = np.meshgrid(row["efit_grid_R"], row["efit_grid_Z"])
    names = np.asarray(row["coil_input_column"]).astype(str)
    sel = np.isin(names, ["magnetics_p2u_current", "magnetics_p2l_current"])
    if mode == "inner":
        sel &= np.asarray(row["coil_R"]) < 0.5
    elif mode == "outer":
        sel &= np.asarray(row["coil_R"]) > 0.5
    rc, zc = np.asarray(row["coil_R"])[sel], np.asarray(row["coil_Z"])[sel]
    w, h = np.asarray(row["coil_width"])[sel], np.asarray(row["coil_height"])[sel]
    resp = np.zeros(r.shape)
    for a in (-.25, .25):
        for b in (-.25, .25):
            resp += greens((rc + a * w)[None, None, :], (zc + b * h)[None, None, :],
                           r[:, :, None], z[:, :, None]).sum(axis=2) / 4
    return 1000 * resp.ravel()


def fallback_psi(row, index):
    """Candidate-004 flux for a shot: from the built archive if present, else recomputed
    (002 Green model with each solenoid element at half the feed current, mast-021)."""
    archive = ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip"
    if PUBLIC and archive.exists():
        with zipfile.ZipFile(archive) as zf, zf.open("mast_public_test.npz") as inner, \
                zipfile.ZipFile(inner) as old:
            return np.lib.format.read_array(old.open(f"shot_{index:04d}_psirz.npy"))
    from participant.mast_green import predict
    currents = aligned_currents(row)
    kernel = dict(coil_kernel(row))
    try:
        psi = predict(row)["psirz"].astype(np.float64).reshape(-1, 65 * 65)
        psi -= 0.5 * currents[:, SIGNALS.index("sol_current"), None] * 1000 * kernel[SIGNALS.index("sol_current")][None]
    except ValueError:  # e.g. Ip <= 0 or unusable Thomson in some frame: coils-only external flux
        psi = sum(currents[:, c, None] * 1000 * (SOL_SCALE if SIGNALS[c] == "sol_current" else 1.0) * k[None]
                  for c, k in kernel.items())
    return psi.reshape(-1, 65, 65).astype(np.float32)


def plausible(phi, r, z, allowed):
    try:
        iz, ir = find_axis(phi, allowed, (len(z) // 2, int(np.argmin(np.abs(r - AXIS_DEFAULT)))))
        lcfs_level(phi, allowed, (iz, ir))
    except ValueError:
        return False
    return 0.7 <= r[ir] <= 1.15 and abs(z[iz]) < 0.45


def shifted_class(currents, ip, rule):
    """Downward-shifted (single-null) class from coil currents only.
    p2: lower divertor coil stronger, median (P2U-P2L)/(|P2U|+|P2L|) < -0.02 (run 032).
    p6: |median (P6U-P6L)/2 / Ip[kA]| x 1e3 > 4, outside the input noise band (run 042).
    union: either (candidate 030, official 996142)."""
    p2u = currents[:, SIGNALS.index("p2u_current")]
    p2l = currents[:, SIGNALS.index("p2l_current")]
    p2 = float(np.median((p2u - p2l) / (np.abs(p2u) + np.abs(p2l) + 1e-6))) < -0.02
    p6 = (currents[:, SIGNALS.index("p6u_current")] - currents[:, SIGNALS.index("p6l_current")]) / 2
    p6 = abs(float(np.median(p6 / np.clip(np.abs(ip / 1000), 1, None)))) * 1e3 > 4.0
    return {"p2": p2, "p6": p6, "union": p2 or p6}[rule]


SHIFTS_CM = np.arange(-3.0, 5.01, 1.0)  # scan range from the label-free distribution (p10 -1, p90 +3)


def symmetry_corrected(anchors, shot_anchor, prior, index, row, allowed, pr, te):
    """Per frame, the radial shift of the prior solved flux that minimizes the midplane Te
    inboard/outboard mismatch (Te is a flux function); 5-frame median of valid shifts,
    nearest fill; added to the Te-centroid anchor. Label-free test-time adaptation."""
    from research.flux_function_check import mismatch
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    old = np.load(Path(prior) / "shots" / f"{index:04d}.npz")["psirz"].astype(np.float64)
    deltas = np.full(len(anchors), np.nan)
    for t in range(0, len(anchors), 2):  # every second frame; median + nearest fill below
        m = []
        for s in SHIFTS_CM:
            try:
                phi = ndimage.shift(old[t], (0, s / 100 / (r[1] - r[0])), order=1, mode="nearest")
                m.append(mismatch(phi, r, z, allowed, pr, te[t]))
            except ValueError:
                m.append(np.nan)
        m = np.array(m)
        if np.isfinite(m).sum() >= 6:
            deltas[t] = SHIFTS_CM[np.nanargmin(m)] / 100
    good = np.nonzero(np.isfinite(deltas))[0]
    if len(good) < 3:
        return anchors
    smooth = ndimage.median_filter(deltas[good], size=5, mode="nearest")
    corr = smooth[np.abs(np.arange(len(anchors))[:, None] - good[None]).argmin(axis=1)]
    return [(a if a is not None else shot_anchor) + c for a, c in zip(anchors, corr)]


def run_shot(task):
    index, entry, out, vertical, vessel, inner, smooth_anchor, edge, te_power, nobz, efps = (
        tuple(task) + ("force", None, "default", False, False, 1.0, False, "none")[len(task) - 3:])
    config = {"vertical": vertical, "vessel": vessel, "inner": inner, "smooth_anchor": smooth_anchor,
              "edge": edge, "te_power": te_power, "edge_nobz": nobz, "efps": efps}
    prior = os.environ.get("MAST_ANCHOR_PRIOR")  # env so spawned workers inherit it
    if prior:
        config["anchor_prior"] = prior
    target = out / "shots" / f"{index:04d}.npz"
    if target.exists():
        meta = json.loads(str(np.load(target)["meta"]))
        if meta.get("config", config) != config:  # older files lack a config; never mix runs
            raise ValueError(f"{target} was made with {meta['config']}, not {config}")
        return meta
    row = dict(np.load(DATA / "inputs" / (Path(entry["path"]).stem + ".npz"),
                       allow_pickle=False))
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    currents = aligned_currents(row)
    ext = np.zeros((len(currents), r.size * z.size))
    for column, response in coil_kernel(row):
        scale = SOL_SCALE if SIGNALS[column] == "sol_current" else 1.0
        ext += currents[:, column, None] * 1000 * scale * response[None]
    if efps != "none":  # hypothesis: efps feeds part of the P2 windings
        sign = -1.0 if efps.startswith("neg-") else 1.0
        ext += sign * currents[:, SIGNALS.index("efps_current"), None] * efps_response(
            row, efps.removeprefix("neg-"))[None]
    if vessel is not None:  # induced passive currents (research/mast_passive.py)
        from research.mast_passive import passive_flux
        ext += passive_flux(row, vessel)[0]
    ext = ext.reshape(-1, len(z), len(r))
    ip = currents[:, SIGNALS.index("plasma_current")] * 1000
    pr, pz, te = mast_points(row)
    allowed = region(row, inner)
    anchors = [te_axis(pr, te[t]) for t in range(len(ip))]
    valid = [a for a in anchors if a is not None]
    shot_anchor = float(np.median(valid)) if valid else AXIS_DEFAULT
    if smooth_anchor and valid:  # E6: 5-frame running median of valid anchors, nearest fill
        idx = np.array([t for t, a in enumerate(anchors) if a is not None])
        vals = ndimage.median_filter(np.array([anchors[t] for t in idx]), size=5, mode="nearest")
        near = idx[np.abs(np.arange(len(anchors))[:, None] - idx[None]).argmin(axis=1)]
        anchors = list(np.interp(near, idx, vals))
    if prior:  # Te flux-function symmetry correction of the axis anchor (run 054)
        anchors = symmetry_corrected(anchors, shot_anchor, prior, index, row, allowed, pr, te)
    fallback = fallback_psi(row, index)
    z_targets = [None] * len(ip)
    lsn = False
    linear = vertical == "config-linear"  # same vertical scheme, EFIT++-style linear basis
    rule = {"config": "p2", "config-linear": "p2", "config-p6": "p6", "config-union": "union"}.get(vertical)
    if rule:
        vertical = "config"
        lsn = shifted_class(currents, ip, rule)
        if not lsn:
            z_targets = [0.0] * len(ip)
    if vertical == "smooth" or (vertical == "config" and lsn):
        # pass 1: force balance; smoothed centroid heights pin pass 2
        heights = np.full(len(ip), np.nan)
        for t in range(len(ip)):
            ok = np.isfinite(te[t])
            try:
                if ip[t] <= 0:
                    continue
                _, cells = solve(ext[t], ip[t], r, z, allowed, PROFILES["force"], None, None,
                                 anchors[t] if anchors[t] is not None else shot_anchor,
                                 (pr[ok], pz[ok], te[t][ok]) if ok.sum() >= 5 else None)
                sz = ((z[1:] + z[:-1]) / 2)[:, None]
                heights[t] = float((cells * sz).sum() / cells.sum())
            except ValueError:
                pass
        good = np.isfinite(heights) & (np.abs(heights) <= 0.4)
        if good.any():
            idx = np.nonzero(good)[0]
            smooth = ndimage.median_filter(heights[idx], size=7, mode="nearest")
            near = idx[np.abs(np.arange(len(ip))[:, None] - idx[None]).argmin(axis=1)]
            z_targets = list(np.interp(near, idx, smooth))
            if vertical == "config":  # LSN: magnitude from force balance, sign downward
                z_targets = [-abs(v) for v in z_targets]
        else:
            z_targets = [0.0] * len(ip)
    edges = [None] * len(ip)
    if edge:  # E3: outboard LCFS at the Te = 30 eV crossing, smoothed over 5 frames
        from research.flux_function_check import te_edge_radius
        raw = np.array([te_edge_radius(pr, te[t]) for t in range(len(ip))])
        good = np.isfinite(raw) & (raw >= 1.2) & (raw <= 1.55)
        if good.sum() >= 3:
            idx = np.nonzero(good)[0]
            sm = ndimage.median_filter(raw[idx], size=5, mode="nearest")
            near = idx[np.abs(np.arange(len(ip))[:, None] - idx[None]).argmin(axis=1)]
            edges = list(np.interp(near, idx, sm))
    psi, kept = [], []
    fvac = np.abs(currents[:, SIGNALS.index("tf_current")]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    beta0s, betan_share = np.full(len(ip), np.nan), np.full(len(ip), np.nan)
    for t in range(len(ip)):
        ok = np.isfinite(te[t])
        info = {}
        try:
            if ip[t] <= 0:
                raise ValueError("Nonpositive plasma current")
            kinetic = vertical != "zero-linear" and not linear and ok.sum() >= 5
            prof = PROFILES["zero-linear" if linear else
                            ("zero" if vertical in ("smooth", "config") else vertical)]
            if te_power != 1.0:
                from dataclasses import replace
                prof = replace(prof, te_power=te_power)
            phi, cells = solve(ext[t], ip[t], r, z, allowed, prof, None, None,
                               anchors[t] if anchors[t] is not None else shot_anchor,
                               (pr[ok], pz[ok], te[t][ok]) if kinetic else None,
                               info, z_targets[t], edges[t], 1.0, 0.0, 0.1, not nobz)
            good = plausible(phi, r, z, allowed)
            if good:  # pressure-share betaN (DIII-D proxy R2 0.928); diagnostics only here
                beta0s[t] = info["beta0"]
                try:
                    betan_share[t] = beta_n_equilibrium(phi, cells, r, z, allowed, info["beta0"],
                                                        info["r0"], fvac[t], ip[t])
                except ValueError:
                    pass
        except ValueError:
            good = False
        kept.append(good)
        psi.append(phi.astype(np.float32) if good else fallback[t])
    psi = np.array(psi)
    meta = {"index": index, "config": config, "frames": len(ip), "kept": int(sum(kept)), "lsn_class": lsn,
            "anchors_valid": len(valid), "kept_mask": [bool(k) for k in kept]}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial.npz")
    np.savez(tmp, psirz=psi, meta=json.dumps(meta), beta0=beta0s, betan_share=betan_share)
    tmp.replace(target)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "mast-025")
    parser.add_argument("--shots", default="pilot")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--vertical", choices=("force", "zero", "zero-linear", "smooth", "config", "config-p6", "config-union",
                                 "config-linear"),
                        default="force")
    parser.add_argument("--inner", choices=("default", "inner"), default="default")
    parser.add_argument("--smooth-anchor", action="store_true")
    parser.add_argument("--edge", action="store_true", help="E3 Thomson outboard-edge constraint")
    parser.add_argument("--te-power", type=float, default=1.0)
    parser.add_argument("--efps", choices=("none", "all", "inner", "outer", "neg-all"),
                        default="none")
    parser.add_argument("--edge-nobz", action="store_true",
                        help="edge correction positions the plasma but is left out of the output")
    parser.add_argument("--vessel", type=json.loads, default=None,
                        help='JSON kwargs for research.passive.wall_filaments, e.g. '
                             '{"r_in":0.19,"r_out":2.02,"z_half":2.2,"thickness":0.02,'
                             '"resistivity":7.2e-7,"spacing":0.1}')
    args = parser.parse_args()
    shots = json.loads((DATA / "data.json").read_text())["shots"]
    worst = [93, 652, 175, 24, 54, 149, 549, 580, 340, 837]
    if args.shots == "pilot":
        indices = sorted(set(worst + list(range(0, len(shots), 50))))
    elif args.shots == "all":
        indices = list(range(len(shots)))
    else:  # path to a JSON list of shot indices
        indices = json.loads(Path(args.shots).read_text())
    started, metas = time.monotonic(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for meta in pool.map(run_shot, [(i, shots[i], args.out, args.vertical, args.vessel,
                                             args.inner, args.smooth_anchor, args.edge,
                                             args.te_power, args.edge_nobz, args.efps)
                                            for i in indices]):
            metas.append(meta)
            if len(metas) % 50 == 0:
                print(f"{len(metas)}/{len(indices)} {time.monotonic() - started:.0f}s", flush=True)
    frames = sum(m["frames"] for m in metas)
    kept = sum(m["kept"] for m in metas)
    summary = {"shots": len(metas), "frames": frames, "kept": kept,
               "kept_fraction": kept / frames, "seconds": time.monotonic() - started}
    (args.out / f"summary-{Path(args.shots).stem}.json").write_text(json.dumps(
        {"summary": summary, "shots": metas}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
