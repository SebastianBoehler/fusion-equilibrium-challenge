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
from .scalar_heads import ScalarHeads

MODELS = {"neighbors": NeighborModel, "interpolation": InterpolationModel,
          "quadratic": QuadraticModel, "kinetic": KineticModel,
          "scalar-trees": ScalarTreeModel, "scalar-heads": ScalarHeads}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order-manifest", type=Path, required=True)
    parser.add_argument("--expected-shots", type=int, required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--flux-dtype", choices=("float16", "float32"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    order = json.loads(args.order_manifest.read_text())
    entries = manifest["shots"]
    config = manifest['config']
    if (not config.startswith('diii_d_') or Path(config).name != config
            or Path(args.archive_name).name != args.archive_name or not args.archive_name.endswith('.zip')):
        raise ValueError('DIII-D configuration and a basename ending in .zip required')
    if any(Path(e['path']).parent.as_posix() != f'data/{config}' for e in entries):
        raise ValueError('Test inputs do not match the declared dataset configuration')
    if (args.expected_shots <= 0 or len(entries) != args.expected_shots
            or order['shot_count'] != args.expected_shots
            or order["dataset_revision"] != manifest["revision"]
            or order["ordered_paths"] != [e["path"] for e in entries]):
        raise ValueError("The complete official test stream order was not verified")
    args.out.mkdir(parents=True, exist_ok=True)
    model = MODELS[args.model].load(args.checkpoint)
    started = time.monotonic()
    target = args.out / f'{config}.npz'
    zipped = args.out / args.archive_name
    if target.exists() or zipped.exists():
        raise FileExistsError('Use a fresh output directory for immutable submission artifacts')
    frames = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, entry in enumerate(entries):
            path = args.cache / "raw" / Path(entry["path"]).name
            if not path.exists():
                raise FileNotFoundError(f"Frozen test input not prepared: {entry['path']}")
            row = read_shot(entry, manifest["revision"], args.cache / "raw")
            count = len(row["efit_times"])
            x = (kinetic_features(row) if args.model in ("kinetic", "scalar-trees", "scalar-heads")
                 else features(row))
            prediction = model.predict_inputs(x)
            frames.append(count)
            for key, expected in (("psirz", (count, 65, 65)),
                                  ("q95", (count,)), ("betaN", (count,))):
                values = np.asarray(prediction[key], dtype=(
                    np.dtype(args.flux_dtype) if key == "psirz" else np.float32))
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f"Invalid prediction: shot {index}, {key}")
                with archive.open(f"shot_{index:04d}_{key}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
            if (index + 1) % 100 == 0 or index + 1 == len(entries):
                print(f"Built {index + 1}/{len(entries)} shots", flush=True)
    with np.load(target, allow_pickle=False) as upload:
        expected_keys = {f"shot_{i:04d}_{k}" for i in range(args.expected_shots)
                         for k in ("psirz", "q95", "betaN")}
        if set(upload.files) != expected_keys:
            raise ValueError("Wrong submission keys")
        for i, count in enumerate(frames):
            for key, shape in (("psirz", (count, 65, 65)), ("q95", (count,)), ("betaN", (count,))):
                values = upload[f"shot_{i:04d}_{key}"]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError("Saved submission validation failed")
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_STORED) as archive:
        archive.write(target, target.name)
    with zipped.open("rb") as handle:
        zipped_hash = file_digest(handle, "sha256").hexdigest()
    receipt = {"model": args.model, "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
               "dataset_revision": manifest["revision"], "dataset_config": config, "shots": len(entries),
               "frames": sum(frames), "test_data_used_only_for_inference": True,
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "order_manifest_sha256": sha256(args.order_manifest.read_bytes()).hexdigest(),
               "expected_shots": args.expected_shots, "flux_dtype": args.flux_dtype,
               "zip_sha256": zipped_hash,
               "zip_bytes": zipped.stat().st_size, "validation": "all keys, shapes, finite values",
               "elapsed_seconds": time.monotonic() - started}
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
