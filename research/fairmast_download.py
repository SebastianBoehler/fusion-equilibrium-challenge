"""Download FAIR-MAST level-2 diagnostics for the magnetic validation harness.

Public, anonymous S3 (s3.echo.stfc.ac.uk, bucket "mast"). Reads ONLY the groups pf_active
(coil currents), magnetics (ip, flux loops), tf (TF current) and thomson_scattering (Te, ne);
never `equilibrium` or any EFIT-derived product. Human approval: 18 October 2026.
Runs in the separate download environment (.venv-fairmast: zarr>=3, s3fs, xarray):
  .venv-fairmast/bin/python -I research/fairmast_download.py --shots 27000-30470 --n 300 \
      --out run_data/fairmast/raw
One .npz per shot; shots with a missing group or no plasma are skipped and logged.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import xarray as xr

STORAGE = dict(anon=True, client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"})
GROUPS = ("pf_active", "magnetics", "tf", "thomson_scattering")


def names(coord):
    return np.array([str(x) for x in coord.values], dtype="U32")


def fetch(shot):
    url = f"s3://mast/level2/shots/{shot}.zarr"
    g = {name: xr.open_zarr(url, group=name, storage_options=STORAGE) for name in GROUPS}
    pf, mag, tf, ts = (g[n] for n in GROUPS)
    out = {
        "shot": shot,
        "coil_time": pf["time"].values, "coil_current": pf["coil_current"].values,
        "coil_channel": names(pf["current_channel"]),
        "mag_time": mag["time"].values, "ip": mag["ip"].values,
        "flux_loop_flux": mag["flux_loop_flux"].values,
        "flux_loop_channel": names(mag["flux_loop_channel"]),
        "flux_loop_geometry_channel": names(mag["flux_loop_geometry_channel"]),
        "flux_loop_r": mag["flux_loop_r"].values, "flux_loop_z": mag["flux_loop_z"].values,
        "tf_time": tf["time"].values, "tf_current": tf["coil_current"].values,
        "ts_time": ts["time"].values, "ts_r": ts["major_radius"].values,
        "ts_te": ts["t_e"].values, "ts_ne": ts["n_e"].values,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", default="27000-30470", help="inclusive range a-b")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", default="0/1", help="k/K: this process takes candidates k::K")
    a = p.parse_args()
    k, nk = map(int, a.worker.split("/"))
    lo, hi = map(int, a.shots.split("-"))
    candidates = random.Random(a.seed).sample(range(lo, hi + 1), min(4 * a.n, hi - lo + 1))[k::nk]
    a.out.mkdir(parents=True, exist_ok=True)
    log = {"kept": [], "skipped": {}}
    for shot in candidates:
        if len(log["kept"]) >= a.n // nk:
            break
        target = a.out / f"{shot}.npz"
        if target.exists():
            log["kept"].append(shot)
            continue
        try:
            d = fetch(shot)
            if np.nanmax(np.abs(d["ip"])) < 1e5 or np.isfinite(d["ts_te"]).sum() < 100:
                raise ValueError("no plasma or no Thomson")
            np.savez_compressed(target, **d)
            log["kept"].append(shot)
        except Exception as e:  # missing group / shot: log and continue
            log["skipped"][shot] = repr(e)[:120]
        if (len(log["kept"]) + len(log["skipped"])) % 20 == 0:
            print(f"kept {len(log['kept'])} skipped {len(log['skipped'])}", flush=True)
            (a.out / f"download-log-{k}.json").write_text(json.dumps(log, indent=1))
    (a.out / f"download-log-{k}.json").write_text(json.dumps(log, indent=1))
    print(f"done: kept {len(log['kept'])} skipped {len(log['skipped'])}")


if __name__ == "__main__":
    main()
