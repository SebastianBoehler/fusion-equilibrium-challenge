"""Pinned MAST inputs; equilibrium targets are never decoded or cached."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq

from .data import array_value

SIGNALS = tuple(f"p{i}{s}_current" for i in range(2, 7) for s in ("l", "u")) + (
    "plasma_current", "sol_current", "tf_current", "efps_current")
COLUMNS = ("source", "efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time",
           *(f"magnetics_{s}" for s in SIGNALS),
           *(f"coil_{s}" for s in ("input_column", "R", "Z", "width", "height")),
           *(f"thomson_core_{s}" for s in ("R", "times", "Te", "ne")))


def read_inputs(entry: dict, revision: str, cache: Path) -> dict:
    if not entry["path"].startswith(("data/mast_public_test/", "data/mast_private_test/")):
        raise ValueError("Only the input-only MAST test splits are permitted")
    path = Path(hf_hub_download(
        entry.get("repo_id", "Sophelio/fusion-equilibrium-challenge"), repo_type="dataset",
        filename=entry["path"], revision=revision, local_dir=cache / "hub"))
    with path.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"MAST source hash mismatch: {entry['path']}")
    schema = pq.read_schema(path)
    forbidden = [n for n in schema.names if n.startswith("efit_")
                 and n not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden or "magnetics_dsep" in schema.names:
        raise ValueError("The MAST source unexpectedly contains equilibrium labels")
    table = pq.read_table(path, columns=list(COLUMNS))
    if len(table) != 1:
        raise ValueError("Expected one MAST shot per file")
    row = {name: array_value(table[name][0]) for name in COLUMNS}
    if row["source"] != "MAST":
        raise ValueError("Expected MAST inputs")
    return row


def aligned_currents(row: dict) -> np.ndarray:
    times = np.asarray(row["magnetics_time"], dtype=float)
    centers = np.asarray(row["efit_times"], dtype=float)
    values = []
    for signal in SIGNALS:
        current = np.asarray(row[f"magnetics_{signal}"], dtype=float)
        valid = np.isfinite(times) & np.isfinite(current)
        if valid.sum() < 2 or np.any(np.diff(times[valid]) <= 0):
            raise ValueError(f"Invalid MAST current series: {signal}")
        values.append(np.interp(centers, times[valid], current[valid]))
    return np.column_stack(values)


def prepare(entry: dict, revision: str, cache: Path) -> Path:
    path = cache / "inputs" / (Path(entry["path"]).stem + ".npz")
    if not path.exists():
        row = read_inputs(entry, revision, cache)
        aligned_currents(row)  # Reject malformed current inputs before caching.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **row, source_hash=entry["sha256"], revision=revision)
        temporary.replace(path)
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_hash"]) != entry["sha256"] or str(data["revision"]) != revision:
            raise ValueError("MAST cache does not match the pinned dataset")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1206)  # Final: 1205
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["shots"]
    if manifest["config"] not in ("mast_public_test", "mast_private_test") or len(entries) != args.expected:
        raise ValueError(f"Expected the complete {args.expected}-shot MAST test split")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(
                lambda e: prepare(e, manifest["revision"], args.cache), entries), 1):
            if count % 50 == 0 or count == len(entries):
                print(f"MAST inputs {count}/{len(entries)}; "
                      f"{time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
