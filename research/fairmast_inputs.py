"""Convert downloaded FAIR-MAST shots (research/fairmast_download.py) to release-format inputs.

The output matches run_data/mast-001/inputs/*.npz, so research.mast_gs_robust runs unchanged
with MAST_DATA_DIR=<out>. Conventions (checked against the release, 18 October):
- PF currents in kA. Release P2 = P2 inner feed + outer feed (level 2 is already x turns);
  the release solenoid is 2x the level-2 solenoid (forum; FAIR-MAST mapping scale 0.5).
- All times in ms. Thomson: level-2 midplane Te(R, t) with a ghost channel 0, as released.
- Coil geometry, grid and units come from one release shot (same machine).
- Frames: Thomson times with |Ip| > 30% of its maximum.
Measured flux loops are stored separately (<out>/magnetics/<shot>.npz) for validation
only; they are never solver inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TEMPLATE = Path("run_data/mast-001")
COPY = ("efit_grid_R", "efit_grid_Z", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height")


def convert(d, template):
    t = d["coil_time"] * 1000
    ch = {str(c): d["coil_current"][i] / 1000 for i, c in enumerate(d["coil_channel"])}
    ip = np.interp(t, d["mag_time"] * 1000, d["ip"]) / 1000
    row = {k: template[k] for k in COPY}
    row.update(source="MAST", magnetics_time=t.astype(np.float32),
               magnetics_plasma_current=ip,
               magnetics_tf_current=np.interp(t, d["tf_time"] * 1000, d["tf_current"]) / 1000,
               magnetics_sol_current=2 * ch["SOL"])
    for s in ("l", "u"):
        S = s.upper()
        row[f"magnetics_p2{s}_current"] = ch[f"P2I{S} FEED"] + ch[f"P2O{S} FEED"]
        row[f"magnetics_p3{s}_current"] = ch[f"P3{S} FEED"]
        row[f"magnetics_p4{s}_current"] = ch[f"P4{S} FEED"]
        row[f"magnetics_p5{s}_current"] = ch[f"P5{S} FEED"]
        row[f"magnetics_p6{s}_current"] = ch[f"P6{S}"]
    row["magnetics_efps_current"] = row["magnetics_p2l_current"] / 20
    ts_t = d["ts_time"].astype(float) * 1000
    te = np.where(d["ts_te"] > 0, d["ts_te"], np.nan).T          # (time, R)
    ne = np.where(d["ts_ne"] > 0, d["ts_ne"], np.nan).T
    row.update(thomson_core_R=d["ts_r"], thomson_core_times=ts_t,
               thomson_core_Te=np.column_stack([np.full(len(ts_t), np.nan), te]),
               thomson_core_ne=np.column_stack([np.full(len(ts_t), np.nan), ne]))
    on = np.abs(np.interp(ts_t, t, ip)) > 0.3 * np.nanmax(np.abs(ip))
    frames = ts_t[on & (np.isfinite(te).sum(axis=1) >= 20)]
    row["efit_times"] = frames.astype(np.float32)
    return row, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("run_data/fairmast/raw"))
    p.add_argument("--out", type=Path, default=Path("run_data/fairmast/data"))
    a = p.parse_args()
    shots = json.loads((TEMPLATE / "data.json").read_text())["shots"]
    template = np.load(TEMPLATE / "inputs" / (Path(shots[0]["path"]).stem + ".npz"))
    (a.out / "inputs").mkdir(parents=True, exist_ok=True)
    (a.out / "magnetics").mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(a.raw.glob("*.npz")):
        d = np.load(f)
        row, frames = convert(d, template)
        if len(frames) < 8:
            continue
        name = f"fairmast_{int(d['shot'])}"
        np.savez_compressed(a.out / "inputs" / f"{name}.npz", **row)
        np.savez_compressed(a.out / "magnetics" / f"{name}.npz", **{k: d[k] for k in (
            "mag_time", "flux_loop_flux", "flux_loop_channel", "flux_loop_geometry_channel",
            "flux_loop_r", "flux_loop_z", "ip")}, frames_ms=frames)
        entries.append({"path": f"{name}.parquet", "shot": int(d["shot"]), "frames": len(frames)})
    (a.out / "data.json").write_text(json.dumps({"source": "FAIR-MAST level 2 (inputs only)",
                                                 "shots": entries}, indent=1))
    print(f"{len(entries)} shots, {sum(e['frames'] for e in entries)} frames")


if __name__ == "__main__":
    main()
