"""A bounded kinetic-feature comparison on the frozen 64/8 shot split."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .data import features, read_shot
from .kinetic import KineticModel, kinetic_features
from .prepared import load_entries
from .quadratic import QuadraticModel
from .score import composite
from .scalar_trees import ScalarTreeModel

MODELS = {"quadratic": QuadraticModel, "kinetic": KineticModel,
          "scalar-trees": ScalarTreeModel}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if {e["path"] for e in manifest["train"]} & {e["path"] for e in manifest["heldout"]}:
        raise ValueError("Training and validation shots overlap")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {s: load_entries(manifest[s], manifest["revision"], args.cache)
            for s in ("train", "heldout")}
    kinetic = {}
    for split in rows:
        kinetic[split] = []
        for entry in manifest[split]:
            row = read_shot(entry, manifest["revision"], args.cache)
            inputs = kinetic_features(row)
            redacted = {k: v for k, v in row.items()
                        if not k.startswith("efit_") or k == "efit_times"}
            redacted.pop("magnetics_dsep", None)
            np.testing.assert_array_equal(inputs, kinetic_features(redacted))
            np.testing.assert_array_equal(features(row), features(redacted))
            kinetic[split].append(inputs)
    y = np.concatenate([row["targets"] for row in rows["train"]])
    results = {}
    for name in args.models:
        model_type = MODELS[name]
        inputs = (kinetic if name != "quadratic" else
                  {s: [row["inputs"] for row in rows[s]] for s in rows})
        started = time.monotonic()
        model = model_type().fit_arrays(np.concatenate(inputs["train"]), y)
        suffix = "joblib" if name == "scalar-trees" else "npz"
        model.save(args.out / f"{name}.{suffix}")
        reloaded = model_type.load(args.out / f"{name}.{suffix}")
        predictions = [reloaded.predict_inputs(x) for x in inputs["heldout"]]
        for x, prediction in zip(inputs["heldout"], predictions, strict=True):
            for key, value in model.predict_inputs(x).items():
                np.testing.assert_array_equal(value, prediction[key])
        quantized = [{k: v.astype(np.float16 if k == "psirz" else np.float32)
                      for k, v in p.items()} for p in predictions]
        results[name] = {"score": composite(rows["heldout"], predictions),
                         "upload_precision_score": composite(rows["heldout"], quantized),
                         "seconds": time.monotonic() - started,
                         "train_shots": len(rows["train"]), "heldout_shots": len(rows["heldout"]),
                         "checkpoint_reload": True, "raw_target_redaction": True}
        (args.out / "comparison.json").write_text(json.dumps(results, indent=2))
        print(name, json.dumps(results[name]), flush=True)


if __name__ == "__main__":
    main()
