"""Build a complete DIII-D submission from a frozen checkpoint and input manifest."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .interpolation import InterpolationModel
from .kinetic import KineticModel, kinetic_features
from .neighbors import NeighborModel
from .quadratic import QuadraticModel
from .data import features, read_shot
from .scalar_trees import ScalarTreeModel

MODELS = {"neighbors": NeighborModel, "interpolation": InterpolationModel,
          "quadratic": QuadraticModel, "kinetic": KineticModel,
          "scalar-trees": ScalarTreeModel}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    order = json.loads(Path("run_data/iteration-002/stream-order-verified.json").read_text())
    entries = manifest["shots"]
    if (len(entries) != 874 or order["dataset_revision"] != manifest["revision"]
            or order["ordered_paths"] != [e["path"] for e in entries]):
        raise ValueError("The complete official test stream order was not verified")
    args.out.mkdir(parents=True, exist_ok=True)
    model = MODELS[args.model].load(args.checkpoint)
    started = time.monotonic()
    target = args.out / "diii_d_public_test.npz"
    frames = []
    deadline = started + 30 * 60
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, entry in enumerate(entries):
            path = args.cache / "raw" / Path(entry["path"]).name
            while not path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Public-data preparation did not finish: {entry['path']}")
                time.sleep(1)
            row = read_shot(entry, manifest["revision"], args.cache / "raw")
            count = len(row["efit_times"])
            x = (kinetic_features(row) if args.model in ("kinetic", "scalar-trees")
                 else features(row))
            prediction = model.predict_inputs(x)
            frames.append(count)
            for key, expected in (("psirz", (count, 65, 65)),
                                  ("q95", (count,)), ("betaN", (count,))):
                values = np.asarray(prediction[key], dtype=(
                    np.float16 if key == "psirz" else np.float32))
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f"Invalid prediction: shot {index}, {key}")
                with archive.open(f"shot_{index:04d}_{key}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
            if (index + 1) % 100 == 0 or index + 1 == len(entries):
                print(f"Built {index + 1}/{len(entries)} shots", flush=True)
    with np.load(target, allow_pickle=False) as upload:
        expected_keys = {f"shot_{i:04d}_{k}" for i in range(874)
                         for k in ("psirz", "q95", "betaN")}
        if set(upload.files) != expected_keys:
            raise ValueError("Wrong submission keys")
        for i, count in enumerate(frames):
            for key, shape in (("psirz", (count, 65, 65)), ("q95", (count,)), ("betaN", (count,))):
                values = upload[f"shot_{i:04d}_{key}"]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError("Saved submission validation failed")
    zipped = args.out / f"fusion-{args.model}-001.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_STORED) as archive:
        archive.write(target, target.name)
    with zipped.open("rb") as handle:
        zipped_hash = file_digest(handle, "sha256").hexdigest()
    receipt = {"model": args.model, "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
               "dataset_revision": manifest["revision"], "public_shots": len(entries),
               "public_frames": sum(frames), "public_test_used_only_for_inference": True,
               "zip_sha256": zipped_hash,
               "zip_bytes": zipped.stat().st_size, "validation": "all keys, shapes, finite values",
               "elapsed_seconds": time.monotonic() - started}
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
