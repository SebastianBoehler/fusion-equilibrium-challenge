# Final-phase runbook (opens 19 October 2026, 02:00 CEST)

Dataset: `Sophelio/fusion-equilibrium-challenge-final`. Configs: `diii_d_private_test`
(879 shots) and `mast_private_test` (1,205 shots). At most **three** Final submissions.
Each needs a reproduction-package reference in the `Code URL` field at submission time.
Submissions need the human: a fresh quota check, and confirmation of their upload workflow.

## 0. Before it opens

- Read the Final guide and rebuild tolerance when they are published.
- Check that the Final parquet has every input column the readers use, including
  `source`, the coil columns and Thomson.
- Pin the dataset at its full 40-character commit SHA:
  `HfApi().list_repo_refs(..., repo_type="dataset")`.

## 1. Manifests and inputs (validated: reproduces both public manifests exactly)

```bash
REV=<40-char Final commit>
.venv/bin/python -B -m participant.final_manifest --config diii_d_private_test --revision $REV \
  --expected 879 --out run_data/final/d3d --download-to run_data/final/d3d-cache
.venv/bin/python -B -m participant.final_manifest --config mast_private_test --revision $REV \
  --expected 1205 --out run_data/final/mast
cp run_data/final/mast/manifest.json run_data/final/mast/data.json
.venv/bin/python -B -m participant.mast_inputs --manifest run_data/final/mast/manifest.json \
  --cache run_data/final/mast --expected 1205
```

**Shot order:** sorted file path. Both public submissions scored with this order. Confirm
it against the Final guide.

## 2. DIII-D: the incumbent 11-member ensemble, unchanged code (CPU, about 4 min)

```bash
CK=($(.venv/bin/python -c "import json;print(' '.join(json.load(open('run_data/submission-010/build-receipt.json'))['checkpoints']))"))
.venv/bin/python -B -m participant.nn_submit --checkpoints "${CK[@]}" \
  --manifest run_data/final/d3d/manifest.json --order-manifest run_data/final/d3d/stream-order.json \
  --cache run_data/final/d3d-cache --expected-shots 879 \
  --archive-name d3d-private.zip --out run_data/final/d3d-pred
```

- `read_shot` uses the hash-verified files in `d3d-cache`, so the pinned `data.py` stays
  unchanged.
- **Public rebuild check (17 Oct, today's `.venv`):** all 2,622 arrays are present; 770 are
  byte-identical and 1,852 differ by at most 1.9e-6 absolute (float32 rounding from CPU
  math-library order). The file SHA therefore differs (`7a338bb9…` vs `6f243219…`).
  Declare float32-level rebuild tolerance in the package.

## 3. MAST: one solver run (CPU; about 60–100 min on 12–14 local workers)

The defaults reproduce the 030 recipe (`config-union`, inner wall, smoothed anchor). Add
`--passive ALLg` and/or `--anchor-prior <previous run dir>` only if the 18 October uploads
(035–037) promote them. Choose `--betan` (`flux` or `flux-kin`) from the 034 result.

```bash
export MAST_DATA_DIR=run_data/final/mast OMP_NUM_THREADS=1 PYTHONPATH=.
.venv/bin/python -B -m research.mast_gs_robust --out run_data/final/mast-solve --workers 12
.venv/bin/python -B -m research.mast_final --solve-dir run_data/final/mast-solve \
  --config mast_private_test --expected 1205 --betan flux-kin --out run_data/final/mast-pred
```

The cluster alternative is a 4-way array with the same flags; `ops/mast-gs-array.sbatch`
needs `MAST_DATA_DIR` exported.

**Regression evidence on public data:**
- `config-union` reproduces the scored 030 flux bit for bit (3 shots, one per class).
- The same holds with `MAST_DATA_DIR` set and the fallback recomputed instead of read from
  the 004 archive.

## 4. Archive

```bash
.venv/bin/python -B -m participant.final_archive \
  --member run_data/final/d3d-pred/diii_d_private_test.npz 879 \
  --member run_data/final/mast-pred/mast_private_test.npz 1205 \
  --out run_data/final/fusion-final-1.zip
```

`final_archive` checks shot counts, shapes, float32 and finiteness for every member, and
writes a receipt with the SHA256 values.

## 5. Reproduction package (blocker for the Code URL)

| Item | Status |
|---|---|
| Source, OSI licence, pinned env | MIT `LICENSE` added (18 Oct); repo still private; torch pinned in `uv.lock` |
| Exact weights | 11 × `model.pt` (1.34 GB), each over 100 MB. Needs a release/HF/Zenodo upload with SHA256 (human decision) |
| Training entrypoint | `participant/nn_prepare.py` + `nn_train.py`; the nn-008 sbatch launchers must be added to `ops/` |
| Inference entrypoint | The commands above |
| Provenance | DIII-D TRAIN only; MAST is label-free physics; no pretrained models |
| Hardware and wall time | DIII-D training ≈ 2.8 A100-h; MAST CPU-only (~100 min on 12 cores) |

Make the repo public only with the human's explicit confirmation, at final submission.
