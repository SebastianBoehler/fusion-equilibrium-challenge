"""Read frozen, public DIII-D training shots without downloading the corpus."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from vendor.fusion_starter.data_fixes import fix_d3d_ip_times

SIGNALS = ("ECOILA", "bcoil", *(f"F{i}{s}" for i in range(1, 10)
                                  for s in ("A", "B")), "plasma_current")
SCALARS = ("efit_q95", "efit_beta_n")


def array_value(value: pa.Scalar):
    """Decode nested numeric columns without allocating millions of Python floats."""
    if not value.is_valid:
        return None
    if not (pa.types.is_list(value.type) or pa.types.is_large_list(value.type)):
        return value.as_py()
    values = value.values
    shape = [len(values)]
    while pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        widths = np.diff(values.offsets.to_numpy())
        if len(widths) == 0 or np.any(widths != widths[0]):
            raise ValueError("Expected a rectangular numeric diagnostic array")
        shape.append(int(widths[0]))
        values = values.flatten()
    if pa.types.is_string(values.type):
        return values.to_pylist()
    return values.to_numpy(zero_copy_only=False).reshape(shape)


def read_shot(entry: dict, revision: str, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(entry["path"]).name
    if not path.exists():
        downloaded = hf_hub_download(
            repo_id="Sophelio/fusion-equilibrium-challenge", repo_type="dataset",
            filename=entry["path"], revision=revision, local_dir=cache / "hub")
        Path(downloaded).replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {entry['path']}")
    table = pq.read_table(path)
    if len(table) != 1:
        raise ValueError("Expected one released shot")
    row = {name: array_value(table[name][0]) for name in table.column_names}
    if row["source"] != "DIII-D":
        raise ValueError("Expected a released DIII-D shot")
    return row


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
