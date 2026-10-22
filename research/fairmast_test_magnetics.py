"""Download public FAIR-MAST magnetics for re-identified test shots (human decision 22 Oct 2026).

Reads ONLY the level-2 `magnetics` group (time, ip, flux loops, CCBV/OBV/OBR pickup coils);
never `equilibrium` or EFIT/CPF products. Writes, per matched release shot NAME:
  OUT/magnetics/NAME.npz  (harness format: mag_time, ip, flux loops, frames_ms = release efit_times)
  OUT/probes/SHOT.npz     (fairmast_probes format)
Runs in .venv-fairmast:
  .venv-fairmast/bin/python -I research/fairmast_test_magnetics.py DATA_DIR MATCH.json OUT [--max-dist 0.005]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import xarray as xr

STORAGE = dict(anon=True, client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"})
ARRAYS = ("ccbv", "obr", "obv")
names = lambda c: np.array([str(x) for x in c.values], dtype="U32")


def fetch(job):
    data, out, e, m = job
    name = Path(e["path"]).stem
    tm, tp = out / "magnetics" / f"{name}.npz", out / "probes" / f"{m['shot']}.npz"
    if tm.exists() and tp.exists():
        return name, "cached"
    try:
        g = xr.open_zarr(f"s3://mast/level2/shots/{m['shot']}.zarr", group="magnetics", storage_options=STORAGE)
        frames = np.load(data / "inputs" / f"{name}.npz")["efit_times"].astype(float)
        np.savez_compressed(tm, mag_time=g["time"].values, ip=g["ip"].values,
                            flux_loop_flux=g["flux_loop_flux"].values,
                            flux_loop_channel=names(g["flux_loop_channel"]),
                            flux_loop_geometry_channel=names(g["flux_loop_geometry_channel"]),
                            flux_loop_r=g["flux_loop_r"].values, flux_loop_z=g["flux_loop_z"].values,
                            frames_ms=frames, shot=m["shot"])
        pr = {"time": g["time"].values}
        for a in ARRAYS:
            b = f"b_field_pol_probe_{a}"
            pr[f"{a}_field"] = g[f"{b}_field"].values
            pr[f"{a}_channel"] = names(g[f"{b}_channel"])
            pr[f"{a}_geometry_channel"] = names(g[f"{b}_geometry_channel"])
            pr[f"{a}_r"], pr[f"{a}_z"] = g[f"{b}_r"].values, g[f"{b}_z"].values
        np.savez_compressed(tp, **pr)
        return name, "ok"
    except Exception as ex:
        return name, f"error {repr(ex)[:100]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path); p.add_argument("match", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--max-dist", type=float, default=0.005); p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    (a.out / "magnetics").mkdir(parents=True, exist_ok=True); (a.out / "probes").mkdir(parents=True, exist_ok=True)
    shots = json.loads((a.data / "data.json").read_text())["shots"]
    matches = [m for m in json.loads(a.match.read_text()) if m["dist"] <= a.max_dist]
    jobs = [(a.data, a.out, shots[m["i"]], m) for m in matches]
    log = {}
    with ThreadPoolExecutor(a.threads) as ex:
        for k, (name, status) in enumerate(ex.map(fetch, jobs)):
            log[name] = status
            if k % 50 == 0:
                print(k, name, status, flush=True)
    (a.out / "download-log.json").write_text(json.dumps(log, indent=0))
    print("done", sum(v in ("ok", "cached") for v in log.values()), "/", len(log))


if __name__ == "__main__":
    main()
