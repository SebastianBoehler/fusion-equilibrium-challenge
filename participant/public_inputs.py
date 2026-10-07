"""Prepare all pinned public input rows, without fitting any model on them."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from .data import features, read_shot


def prepare(entry: dict, revision: str, cache: Path) -> Path:
    path = cache / "inputs" / f"{Path(entry['path']).stem}.npz"
    if not path.exists():
        row = read_shot(entry, revision, cache / "raw")
        inputs = features(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, inputs=inputs, times=row["efit_times"],
                            source_hash=entry["sha256"], revision=revision)
    with np.load(path, allow_pickle=False) as data:
        if (str(data["source_hash"]) != entry["sha256"]
                or str(data["revision"]) != revision):
            raise ValueError("Public inputs do not match pinned source")
        if data["inputs"].shape != (len(data["times"]), 21):
            raise ValueError("Unexpected public actuator input shape")
        if not np.isfinite(data["inputs"]).all():
            raise ValueError("Nonfinite public inputs")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["shots"]
    if len(entries) != 874 or len({e["path"] for e in entries}) != 874:
        raise ValueError("Expected all 874 DIII-D public shots exactly once")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for count, _ in enumerate(pool.map(
                lambda entry: prepare(entry, manifest["revision"], args.cache), entries), 1):
            if count % 25 == 0 or count == len(entries):
                print(f"Public input preparation {count}/{len(entries)}; "
                      f"{time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
