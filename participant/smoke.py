"""Run the preregistered eight-shot/two-shot DIII-D smoke experiment."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .data import read_shot
from .ridge import RidgeModel
from .score import composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    train_ids = {entry["path"] for entry in manifest["train"]}
    held_ids = {entry["path"] for entry in manifest["heldout"]}
    if len(train_ids) != 8 or len(held_ids) != 2 or train_ids & held_ids:
        raise ValueError("Frozen eight/two disjoint shot split required")
    started = time.monotonic()
    probe = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
             "python": platform.python_version(), "numpy": np.__version__,
             "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
             "train_shots": sorted(train_ids), "heldout_shots": sorted(held_ids)}
    (args.out / "runtime.json").write_text(json.dumps(probe, indent=2))
    train = [read_shot(entry, manifest["revision"], args.out / "data")
             for entry in manifest["train"]]
    held = [read_shot(entry, manifest["revision"], args.out / "data")
            for entry in manifest["heldout"]]
    print("Downloaded eight training and two held-out shots", flush=True)
    model = RidgeModel().fit(train)
    model.save(args.out / "model.npz")
    reloaded = RidgeModel.load(args.out / "model.npz")
    predictions = [reloaded.predict(row) for row in held]
    for row, pred in zip(held, predictions, strict=True):
        redacted = {key: value for key, value in row.items()
                    if not key.startswith("efit_") or key == "efit_times"}
        redacted.pop("magnetics_dsep", None)
        clean = reloaded.predict(redacted)
        for key in pred:
            np.testing.assert_array_equal(pred[key], clean[key])
    np.savez_compressed(args.out / "heldout_predictions.npz",
                        **{f"shot_{i:04d}_{key}": value.astype(np.float32)
                           for i, pred in enumerate(predictions) for key, value in pred.items()})
    print("Training, checkpoint reload, and target-redaction checks passed", flush=True)
    # Use real targets only to verify the scoring harness, never as model inputs.
    perfect = [{"psirz": np.asarray(held[0]["efit_psirz"]),
                "q95": np.asarray(held[0]["efit_q95"]),
                "betaN": np.asarray(held[0]["efit_beta_n"])}]
    self_check = composite(held[:1], perfect)
    if abs(self_check["S"] - 1.0) > 1e-4:
        raise ValueError(f"Organizer scoring adapter self-check failed: {self_check}")
    score = composite(held, predictions)
    result = {**probe, "status": "completed", "training_frames": model.training_frames,
              "heldout_frames": sum(len(row["efit_times"]) for row in held),
              "perfect_self_check": self_check["S"], "score": score,
              "elapsed_seconds": time.monotonic() - started,
              "scope": "DIII-D held-out smoke; no official score or MAST transfer result"}
    (args.out / "scientific-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
