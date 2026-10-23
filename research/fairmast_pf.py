"""Download FAIR-MAST level-2 pf_active (PF/solenoid currents) for matched test shots.

Reads ONLY the `pf_active` group (time, coil_current, current_channel); never equilibrium/EFIT.
Human decision 22 Oct 2026 (re-identification; disclosed). Runs in .venv-fairmast:
  .venv-fairmast/bin/python -I research/fairmast_pf.py MATCH.json OUT_DIR [--n N] [--threads 16]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import xarray as xr

STORAGE = dict(anon=True, client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"})


def fetch(job):
    shot, out = job
    target = out / f"{shot}.npz"
    if target.exists():
        return shot, "cached"
    try:
        pf = xr.open_zarr(f"s3://mast/level2/shots/{shot}.zarr", group="pf_active", storage_options=STORAGE)
        np.savez_compressed(target, coil_time=pf["time"].values, coil_current=pf["coil_current"].values,
                            coil_channel=np.array([str(c) for c in pf["current_channel"].values], dtype="U32"))
        return shot, "ok"
    except Exception as ex:
        return shot, f"error {repr(ex)[:80]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("match", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=0); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    shots = [m["shot"] for m in json.loads(a.match.read_text())][::a.stride]
    shots = shots[:a.n] if a.n else shots
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(fetch, [(s, a.out) for s in shots]))
    print("done", sum(r[1] in ("ok", "cached") for r in res), "/", len(res))


if __name__ == "__main__":
    main()
