"""Read frozen, public DIII-D training shots without downloading the corpus."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq

from vendor.fusion_starter.data_fixes import fix_d3d_ip_times

SIGNALS = ("ECOILA", "bcoil", *(f"F{i}{s}" for i in range(1, 10)
                                  for s in ("A", "B")), "plasma_current")
SCALARS = ("efit_q95", "efit_beta_n")


def read_shot(entry: dict, revision: str, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(entry["path"]).name
    if not path.exists():
        url = ("https://huggingface.co/datasets/Sophelio/"
               f"fusion-equilibrium-challenge/resolve/{revision}/{entry['path']}")
        temporary = path.with_suffix(".part")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {entry['path']}")
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1 or rows[0]["source"] != "DIII-D":
        raise ValueError("Expected one released DIII-D training shot")
    return rows[0]


def features(row: dict) -> np.ndarray:
    """Use only actuator currents and the allowed EFIT timestamp input."""
    target_times = np.asarray(row["efit_times"], dtype=np.float64)
    shared_times = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = fix_d3d_ip_times(row)
    columns = []
    for signal in SIGNALS:
        times = ip_times if signal == "plasma_current" else shared_times
        values = np.asarray(row[f"magnetics_{signal}"], dtype=np.float64)
        valid = np.isfinite(times) & np.isfinite(values)
        if valid.sum() < 2 or np.any(np.diff(times[valid]) <= 0):
            raise ValueError(f"Invalid time series: {signal}")
        columns.append(np.interp(target_times, times[valid], values[valid]))
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model inputs")
    return result


def targets(row: dict) -> np.ndarray:
    psi = np.asarray(row["efit_psirz"], dtype=np.float64)
    if psi.shape != (len(row["efit_times"]), 65, 65):
        raise ValueError(f"Invalid training flux shape: {psi.shape}")
    scalars = [np.asarray(row[name], dtype=np.float64) for name in SCALARS]
    return np.column_stack([psi.reshape(len(psi), -1), *scalars])
