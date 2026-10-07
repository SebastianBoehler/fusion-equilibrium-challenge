"""Fit one frozen kinetic model and score separate validation shot groups."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .kinetic import KineticModel
from .prepared import kinetic_inputs, load_entries
from .score import composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    groups = {"train": manifest["train"], **manifest["validation"]}
    paths = [e["path"] for entries in groups.values() for e in entries]
    if len(paths) != len(set(paths)) or not all(groups.values()):
        raise ValueError("Nonempty disjoint training and validation groups required")
    args.out.mkdir(parents=True, exist_ok=True)
    probe = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
             "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
             "model": "kinetic", "regularization": KineticModel.regularization,
             "shot_counts": {k: len(v) for k, v in groups.items()}}
    (args.out / "runtime.json").write_text(json.dumps(probe, indent=2))
    training = load_entries(groups["train"], manifest["revision"], args.cache)
    x = np.concatenate(kinetic_inputs(groups["train"], manifest["revision"], args.cache))
    y = np.concatenate([row["targets"] for row in training])
    del training
    model = KineticModel().fit_arrays(x, y)
    del x, y
    checkpoint = args.out / "kinetic.npz"
    model.save(checkpoint)
    reloaded = KineticModel.load(checkpoint)
    scores = {}
    for name, entries in manifest["validation"].items():
        rows = load_entries(entries, manifest["revision"], args.cache)
        inputs = kinetic_inputs(entries, manifest["revision"], args.cache)
        predictions = []
        for values in inputs:
            prediction = reloaded.predict_inputs(values)
            for key, value in model.predict_inputs(values).items():
                np.testing.assert_array_equal(value, prediction[key])
            predictions.append({k: v.astype(np.float16 if k == "psirz" else np.float32)
                                for k, v in prediction.items()})
        scores[name] = composite(rows, predictions)
        print(name, json.dumps(scores[name]), flush=True)
    result = {**probe, "status": "completed", "scores": scores,
              "training_frames": model.training_frames, "checkpoint_reload": True,
              "raw_target_redaction": True,
              "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - started,
              "scope": "DIII-D training-release validation; no public-test fitting"}
    (args.out / "scientific-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
