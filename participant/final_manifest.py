"""Pinned shot manifest + stream order for any dataset config (public or Final).

Lists data/<config>/*.parquet at a full commit SHA, records each file's LFS sha256 and size,
and orders shots by sorted path (the order both public submissions were scored with).
Optionally downloads and hash-verifies every file into a cache directory, so that the
existing readers (participant.data.read_shot uses cache/<name> when present) need no change.

Validation: regenerating the public manifests must reproduce run_data/iteration-002 and
run_data/mast-001 exactly (python -m participant.final_manifest --check-public).
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download

PUBLIC = "Sophelio/fusion-equilibrium-challenge"
FINAL = "Sophelio/fusion-equilibrium-challenge-final"


def list_shots(repo_id: str, config: str, revision: str) -> list[dict]:
    if len(revision) != 40:
        raise ValueError("Pin a full 40-character commit SHA")
    tree = HfApi().list_repo_tree(repo_id, path_in_repo=f"data/{config}", repo_type="dataset",
                                  revision=revision, expand=True)
    shots = [{"path": f.path, "sha256": f.lfs.sha256, "size": f.size}
             for f in tree if f.path.endswith(".parquet")]
    if any(s["sha256"] is None for s in shots):
        raise ValueError("A shot file has no LFS sha256")
    return sorted(shots, key=lambda s: s["path"])


def download(repo_id: str, revision: str, shots: list[dict], cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for k, entry in enumerate(shots, 1):
        path = cache / Path(entry["path"]).name
        if not path.exists():
            got = hf_hub_download(repo_id, filename=entry["path"], repo_type="dataset",
                                  revision=revision, local_dir=cache / "hub")
            shutil.move(got, path)
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {entry['path']}")
        if k % 100 == 0 or k == len(shots):
            print(f"{k}/{len(shots)} verified", flush=True)


def check_public() -> None:
    """The public manifests used by scored submissions must be reproduced exactly."""
    d3d = json.loads(Path("run_data/iteration-002/public-data.json").read_text())
    order = json.loads(Path("run_data/iteration-002/stream-order-verified.json").read_text())
    mast = json.loads(Path("run_data/mast-001/data.json").read_text())
    got = list_shots(PUBLIC, "diii_d_public_test", d3d["revision"])
    assert [(s["path"], s["sha256"], s["size"]) for s in got] == [
        (s["path"], s["sha256"], s["size"]) for s in d3d["shots"]], "DIII-D manifest differs"
    assert [s["path"] for s in got] == order["ordered_paths"], "DIII-D order differs"
    got = list_shots(PUBLIC, "mast_public_test", mast["revision"])
    assert [(s["path"], s["sha256"], s["size"]) for s in got] == [
        (s["path"], s["sha256"], s["bytes"]) for s in mast["shots"]], "MAST manifest differs"
    print("public manifests reproduced exactly (874 DIII-D, 1206 MAST)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-public", action="store_true")
    p.add_argument("--repo-id", default=FINAL)
    p.add_argument("--config", choices=("diii_d_private_test", "mast_private_test",
                                        "diii_d_public_test", "mast_public_test"))
    p.add_argument("--revision")
    p.add_argument("--expected", type=int)
    p.add_argument("--out", type=Path, help="directory for manifest.json and stream-order.json")
    p.add_argument("--download-to", type=Path)
    a = p.parse_args()
    if a.check_public:
        return check_public()
    shots = list_shots(a.repo_id, a.config, a.revision)
    if a.expected is not None and len(shots) != a.expected:
        raise ValueError(f"Expected {a.expected} shots, found {len(shots)}")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"repo_id": a.repo_id, "revision": a.revision, "config": a.config,
                "split": a.config.split("_", 2)[-1], "labels_used": False,
                "shots": [s | {"bytes": s["size"], "repo_id": a.repo_id} for s in shots]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (a.out / "stream-order.json").write_text(json.dumps(
        {"dataset_revision": a.revision, "shot_count": len(shots),
         "ordered_paths": [s["path"] for s in shots]}, indent=1))
    print(f"{a.config}: {len(shots)} shots at {a.revision}")
    if a.download_to is not None:
        download(a.repo_id, a.revision, shots, a.download_to)


if __name__ == "__main__":
    main()
