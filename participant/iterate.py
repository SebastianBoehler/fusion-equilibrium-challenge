"""Compare small models using a pinned split of complete plasma shots."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .prepared import load_entries
from .interpolation import InterpolationModel
from .neighbors import NeighborModel
from .ridge import RidgeModel
from .quadratic import QuadraticModel
from .score import composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    train_ids = {entry["path"] for entry in manifest["train"]}
    held_ids = {entry["path"] for entry in manifest["heldout"]}
    if train_ids & held_ids or not train_ids or not held_ids:
        raise ValueError("Training and held-out shots must be separate")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {split: load_entries(manifest[split], manifest["revision"], args.cache)
            for split in ("train", "heldout")}
    x = np.concatenate([row["inputs"] for row in rows["train"]])
    y = np.concatenate([row["targets"] for row in rows["train"]])
    results = {}
    for name, model_type in (("ridge", RidgeModel), ("neighbors", NeighborModel),
                             ("interpolation", InterpolationModel),
                             ("quadratic", QuadraticModel)):
        started = time.monotonic()
        model = model_type().fit_arrays(x, y)
        model.save(args.out / f"{name}.npz")
        reloaded = model_type.load(args.out / f"{name}.npz")
        predictions = [reloaded.predict_inputs(row["inputs"]) for row in rows["heldout"]]
        for row, pred in zip(rows["heldout"], predictions, strict=True):
            clean = model.predict_inputs(row["inputs"])
            for key in pred:
                np.testing.assert_array_equal(pred[key], clean[key])
        score = composite(rows["heldout"], predictions)
        results[name] = {"score": score, "seconds": time.monotonic() - started,
                         "train_shots": len(train_ids), "heldout_shots": len(held_ids),
                         "checkpoint_reload": True,
                         "inference_inputs": "21 actuator currents only"}
        (args.out / "comparison.json").write_text(json.dumps(results, indent=2))
        print(name, json.dumps(results[name]), flush=True)
        del model, reloaded


if __name__ == "__main__":
    main()
