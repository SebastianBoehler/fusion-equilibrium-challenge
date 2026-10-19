"""Download FAIR-MAST level-2 pickup-coil data for the harness shots (validation only).

Arrays: magnetics b_field_pol_probe_{ccbv,obr,obv}_{field,r,z} + channel names (T, m),
time base 'time'. Never equilibrium/EFIT. Runs in .venv-fairmast:
  .venv-fairmast/bin/python -I research/fairmast_probes.py --worker 0/6
Writes run_data/fairmast/probes/<shot>.npz for every shot in run_data/fairmast/raw/.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

STORAGE = dict(anon=True, client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"})
ARRAYS = ("ccbv", "obr", "obv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("run_data/fairmast/raw"))
    p.add_argument("--out", type=Path, default=Path("run_data/fairmast/probes"))
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    k, nk = map(int, a.worker.split("/"))
    a.out.mkdir(parents=True, exist_ok=True)
    shots = sorted(int(f.stem) for f in a.raw.glob("*.npz"))[k::nk]
    for shot in shots:
        target = a.out / f"{shot}.npz"
        if target.exists():
            continue
        try:
            m = xr.open_zarr(f"s3://mast/level2/shots/{shot}.zarr", group="magnetics", storage_options=STORAGE)
            out = {"time": m["time"].values}
            for name in ARRAYS:
                base = f"b_field_pol_probe_{name}"
                out[f"{name}_field"] = m[f"{base}_field"].values
                out[f"{name}_channel"] = np.array([str(x) for x in m[f"{base}_channel"].values], dtype="U32")
                out[f"{name}_geometry_channel"] = np.array([str(x) for x in m[f"{base}_geometry_channel"].values], dtype="U32")
                out[f"{name}_r"], out[f"{name}_z"] = m[f"{base}_r"].values, m[f"{base}_z"].values
            np.savez_compressed(target, **out)
        except Exception as e:
            print(shot, "skipped", repr(e)[:100], flush=True)
    print("worker", k, "done", flush=True)


if __name__ == "__main__":
    main()
