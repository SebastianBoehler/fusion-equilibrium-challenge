"""Pack exact fixed DIII-D training contours and bit-packed interior masks."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from . import score  # Initialize unchanged vendored absolute imports.
from .data import array_value
from .contour_packing import pack_targets
from .inductance_loss import geometry, inductance
from vendor.fusion_starter.fusion_scoring.derive import _points_in_polygon

COLUMNS = ('efit_psirz', 'efit_grid_R', 'efit_grid_Z', 'efit_lcfs_n', 'efit_lcfs_r', 'efit_lcfs_z')


def extract(job):
    path, expected_sha = job
    torch.set_num_threads(1)
    with path.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != expected_sha:
            raise ValueError(f'Training file hash mismatch: {path.name}')
    table = pq.read_table(path, columns=list(COLUMNS))
    row = {name: array_value(table[name][0]) for name in COLUMNS}
    fields = np.asarray(row['efit_psirz'], np.float32)
    r, z = (np.asarray(row[name], float) for name in ('efit_grid_R', 'efit_grid_Z'))
    rr, zz = np.meshgrid(r, z)
    contours, masks = [], []
    for i, count in enumerate(np.asarray(row['efit_lcfs_n'], int)):
        contour = np.column_stack((row['efit_lcfs_r'][i][:count], row['efit_lcfs_z'][i][:count]))
        contours.append(torch.tensor(contour, dtype=torch.float64))
        masks.append(torch.tensor(_points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(65, 65)))
    rt, zt = (torch.tensor(v, dtype=torch.float64) for v in (r, z))
    points, weights, interior = (v.float() for v in geometry(contours, masks, rt, zt))
    target = inductance(torch.from_numpy(fields), rt.float(), zt.float(), points, weights, interior)
    packed_masks = np.packbits(torch.stack(masks).numpy().reshape(len(fields), -1), axis=1)
    return points.numpy(), weights.numpy(), packed_masks, target.numpy(), r.astype(np.float32), z.astype(np.float32)


def extract_valid(job):
    result = extract(job)
    return (*result, np.ones(len(result[0]), dtype=bool))


def pack(raw, packed, split_path, out, workers):
    index = json.loads((packed / 'index.json').read_text())
    names = index['names']
    paths = [raw / f'{name}.parquet' for name in names]
    maximum = 0
    for i, path in enumerate(paths):
        table = pq.read_table(path, columns=['efit_lcfs_n'])
        counts = np.asarray(array_value(table['efit_lcfs_n'][0]), int)
        if len(counts) != index['offsets'][i + 1] - index['offsets'][i]:
            raise ValueError(f'Training frame order mismatch: {path.name}')
        maximum = max(maximum, int(counts.max()))
    pack_targets(raw, packed, split_path, out, workers, maximum, extract_valid,
                 'same functional of true flux on fixed true EFIT contour; not raw EFIT LI')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    pack(args.raw, args.packed, args.split, args.out, args.workers)


if __name__ == '__main__':
    main()
