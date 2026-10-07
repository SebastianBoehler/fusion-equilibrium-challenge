"""Compare a selected candidate with the frozen first model on fresh shots."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .data import read_shot
from .kinetic import KineticModel, kinetic_features
from .physical import PhysicalModel, PhysicalRegularizedModel
from .prepared import load_entries
from .score import composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-model", choices=("physical", "physical-strong"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    training_path = Path("protocols/iteration-002-data.json")
    if sha256(training_path.read_bytes()).hexdigest() != manifest["training_manifest_sha256"]:
        raise ValueError("Confirmation references a different training protocol")
    old = json.loads(training_path.read_text())
    used = {e["path"] for s in ("train", "heldout") for e in old[s]}
    fresh = {e["path"] for e in manifest["heldout"]}
    if len(fresh) != 8 or used & fresh:
        raise ValueError("Confirmation shots are not fresh")
    rows = load_entries(manifest["heldout"], manifest["revision"], args.cache)
    inputs = [kinetic_features(read_shot(e, manifest["revision"], args.cache))
              for e in manifest["heldout"]]
    cls = PhysicalModel if args.candidate_model == "physical" else PhysicalRegularizedModel
    models = {"baseline": KineticModel.load(args.baseline), "candidate": cls.load(args.candidate)}
    result = {"confirmation_manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
              "heldout_shots": 8, "model_fitting": False, "candidate_model": args.candidate_model}
    for name, model in models.items():
        predictions = [model.predict_inputs(x) for x in inputs]
        quantized = [{k: v.astype(np.float16 if k == "psirz" else np.float32)
                      for k, v in p.items()} for p in predictions]
        result[name] = composite(rows, quantized)
    result["candidate_improved"] = result["candidate"]["S"] > result["baseline"]["S"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
