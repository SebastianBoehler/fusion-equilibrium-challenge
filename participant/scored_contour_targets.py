"""Training-only contours and inductance targets from the unchanged local scorer."""

import argparse
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from . import fast_score
from .contour_packing import pack_targets
from .data import array_value
from .inductance_loss import geometry, inductance
from vendor.fusion_starter.fusion_scoring.derive import _points_in_polygon

DESCRIPTION = 'scorer-extracted true training contour; functional li; fixed local normal approximation'


def extract(job):
    path, expected = job
    torch.set_num_threads(1)
    with path.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != expected:
            raise ValueError(f'Training file hash mismatch: {path.name}')
    columns = ['source', 'efit_psirz', 'efit_grid_R', 'efit_grid_Z']
    table = pq.read_table(path, columns=columns)
    row = {k: array_value(table[k][0]) for k in columns}
    if row['source'] != 'DIII-D':
        raise ValueError('Only DIII-D training targets are allowed')
    fields = np.asarray(row['efit_psirz'], np.float32)
    r, z, coarse, mask_f = fast_score._geometry()
    for actual, expected_grid in ((row['efit_grid_R'], r), (row['efit_grid_Z'], z)):
        if not np.allclose(actual, expected_grid, rtol=0, atol=5e-7):
            raise ValueError('Raw training grid differs from scorer geometry')
    contours, derived, _, _ = fast_score.official.build_reference(fields.astype(float), r, z, coarse, mask_f)
    valid = np.array([c is not None for c in contours]) & np.isfinite(derived[:, fast_score.official.LI_IDX])
    n, width = len(fields), fast_score.official.N_POINTS
    points = np.full((n, width, 2), np.nan, np.float32)
    weights = np.full((n, width), np.nan, np.float32)
    target = np.full(n, np.nan, np.float32)
    masks = np.zeros((n, 529), np.uint8)
    if valid.any():
        rr, zz = np.meshgrid(r, z)
        indices = np.flatnonzero(valid)
        polygons = [contours[i] for i in indices]
        inside = [torch.tensor(_points_in_polygon(rr.ravel(), zz.ravel(), c).reshape(65, 65)) for c in polygons]
        rt, zt = torch.tensor(r), torch.tensor(z)
        p, w, interior = (v.float() for v in geometry([torch.tensor(c) for c in polygons], inside, rt, zt))
        truth = torch.from_numpy(fields[indices])
        dz, dr = torch.gradient(truth, spacing=(zt.float(), rt.float()), dim=(1, 2))
        sampled = torch.nn.functional.grid_sample(torch.stack((dr, dz), dim=1), p[:, :, None],
                                                   align_corners=True)[:, :, :, 0]
        good = (torch.isfinite(sampled).all((1, 2)) & (sampled.square().sum(1) > 0).all(1)).numpy()
        valid[indices[~good]] = False
        picks = indices[good]
        measured = inductance(truth[good], rt.float(), zt.float(), p[good], w[good], interior[good]).numpy() if len(picks) else np.empty(0)
        if len(picks) and np.max(abs(measured - derived[picks, fast_score.official.LI_IDX])) > 2e-5:
            raise ValueError('Packed scorer-contour quadrature mismatch')
        if len(picks):
            points[picks], weights[picks], target[picks] = p.numpy()[good], w.numpy()[good], measured
            masks[picks] = np.packbits(torch.stack(inside).numpy()[good].reshape(len(picks), 4225), axis=1)
    return points, weights, masks, target, r.astype(np.float32), z.astype(np.float32), valid


def main():
    parser = argparse.ArgumentParser()
    for key in ('raw', 'packed', 'split', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    a = parser.parse_args()
    pack_targets(a.raw, a.packed, a.split, a.out, a.workers, 512, extract, DESCRIPTION)


if __name__ == '__main__':
    main()
