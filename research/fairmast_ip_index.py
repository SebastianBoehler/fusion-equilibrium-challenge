"""Plasma-current fingerprint index of all FAIR-MAST level-2 shots (for test-shot matching).

Reads ONLY magnetics/ip and its time base from the public anonymous S3 bucket; never
`equilibrium` or any EFIT/CPF product. Human decision 22 Oct 2026: anonymised test shots may be
re-identified by matching their released Ip/coil traces; disclosed in provenance.
Fingerprint: Ip (kA, sign kept) sampled every 5 ms on [-0.05, 0.75] s, NaN outside the record.
Runs in .venv-fairmast:
  .venv-fairmast/bin/python -I research/fairmast_ip_index.py --shots SHOTLIST --out run_data/fairmast/ip_index
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

STORAGE = dict(anon=True, client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"})
GRID = np.round(np.arange(-0.05, 0.7501, 0.005), 4)


def fingerprint(shot):
    try:
        mag = xr.open_zarr(f"s3://mast/level2/shots/{shot}.zarr", group="magnetics",
                           storage_options=STORAGE)
        t = np.asarray(mag["time"].values, float)
        ip = np.asarray(mag["ip"].values, float).ravel()
        if t.size < 10 or ip.size != t.size:
            return shot, None
        if np.nanmax(np.abs(ip)) > 1e4:          # A -> kA
            ip = ip / 1000.0
        fp = np.interp(GRID, t, ip, left=np.nan, right=np.nan)
        return shot, fp.astype(np.float32)
    except Exception:
        return shot, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--chunk", type=int, default=500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    shots = [int(s) for s in a.shots.read_text().split()]
    for c in range(0, len(shots), a.chunk):
        target = a.out / f"chunk{c // a.chunk:03d}.npz"
        if target.exists():
            continue
        with ThreadPoolExecutor(a.threads) as ex:
            res = [r for r in ex.map(fingerprint, shots[c:c + a.chunk]) if r[1] is not None]
        np.savez(target, shots=np.array([r[0] for r in res]), fp=np.array([r[1] for r in res]), grid=GRID)
        print(f"chunk {c // a.chunk}: {len(res)} / {len(shots[c:c + a.chunk])}", flush=True)


if __name__ == "__main__":
    main()
