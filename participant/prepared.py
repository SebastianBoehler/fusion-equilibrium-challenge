"""Compact training rows so short CPU experiments need little memory."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import features, read_shot, targets
from .kinetic import kinetic_features


def kinetic_inputs(entries: list[dict], revision: str, cache: Path) -> list[np.ndarray]:
    inputs = []
    for entry in entries:
        row = read_shot(entry, revision, cache)
        values = kinetic_features(row)
        redacted = {k: v for k, v in row.items()
                    if not k.startswith("efit_") or k == "efit_times"}
        redacted.pop("magnetics_dsep", None)
        np.testing.assert_array_equal(values, kinetic_features(redacted))
        np.testing.assert_array_equal(features(row), features(redacted))
        inputs.append(values)
    return inputs


def prepare(entry: dict, revision: str, cache: Path) -> dict:
    path = cache / "prepared" / f"{Path(entry['path']).stem}.npz"
    if not path.exists():
        row = read_shot(entry, revision, cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, inputs=features(row), targets=targets(row),
                            times=np.asarray(row["efit_times"]),
                            source_hash=entry["sha256"], revision=revision)
    with np.load(path, allow_pickle=False) as values:
        if (str(values["source_hash"]) != entry["sha256"]
                or str(values["revision"]) != revision):
            raise ValueError("Prepared row does not match frozen source")
        y = values["targets"]
        return {"inputs": values["inputs"], "targets": y,
                "source": "DIII-D", "efit_times": values["times"],
                "efit_psirz": y[:, :-2].reshape(-1, 65, 65),
                "efit_q95": y[:, -2], "efit_beta_n": y[:, -1]}


def load_entries(entries: list[dict], revision: str, cache: Path) -> list[dict]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for row in pool.map(lambda entry: prepare(entry, revision, cache), entries):
            rows.append(row)
            if len(rows) % 8 == 0 or len(rows) == len(entries):
                print(f"Prepared {len(rows)}/{len(entries)} training-release shots", flush=True)
    return rows
