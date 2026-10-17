"""Training-only LCFS and axis samples for differentiable flux-shape losses."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import map_coordinates

from .data import array_value

BOUNDARY_POINTS = 64
COLUMNS = ('efit_psirz', 'efit_grid_R', 'efit_grid_Z', 'efit_lcfs_n',
           'efit_lcfs_r', 'efit_lcfs_z', 'efit_r_axis', 'efit_z_axis')


def sample_flux(psi, points):
    """Bilinear samples; normalized coordinates are (R, Z), images are (Z, R)."""
    frames = np.broadcast_to(np.arange(len(psi))[:, None], points.shape[:2])
    indices = (points + 1) * 32
    coordinates = np.stack((frames, indices[..., 1], indices[..., 0]))
    return map_coordinates(psi, coordinates, order=1, mode='nearest', prefilter=False)


def targets(row):
    psi = np.asarray(row['efit_psirz'], dtype=np.float32)
    r, z = (np.asarray(row[k], dtype=float) for k in ('efit_grid_R', 'efit_grid_Z'))
    if psi.shape[1:] != (65, 65) or r.shape != (65,) or z.shape != (65,):
        raise ValueError('Expected a 65 by 65 DIII-D flux grid')
    if not np.allclose(np.diff(r), np.diff(r).mean()) or not np.allclose(np.diff(z), np.diff(z).mean()):
        raise ValueError('Geometry sampling requires a uniform grid')
    points = np.zeros((len(psi), BOUNDARY_POINTS + 5, 2), dtype=np.float32)
    valid = np.zeros(len(psi), dtype=bool)
    for i, count in enumerate(np.asarray(row['efit_lcfs_n'], dtype=int)):
        contour = np.column_stack((row['efit_lcfs_r'][i][:count], row['efit_lcfs_z'][i][:count]))
        axis = np.array([row['efit_r_axis'][i], row['efit_z_axis'][i]])
        if count < 4 or not np.isfinite(contour).all() or not np.isfinite(axis).all():
            continue
        contour = np.concatenate((contour, contour[:1]))
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
        if distance[-1] <= 0:
            continue
        spacing = np.arange(BOUNDARY_POINTS) * distance[-1] / BOUNDARY_POINTS
        boundary = np.column_stack([np.interp(spacing, distance, contour[:, j]) for j in (0, 1)])
        neighbors = axis + np.array([[0, 0], [r[1] - r[0], 0], [r[0] - r[1], 0],
                                     [0, z[1] - z[0]], [0, z[0] - z[1]]])
        physical = np.concatenate((boundary, neighbors))
        normalized = 2 * (physical - [r[0], z[0]]) / [r[-1] - r[0], z[-1] - z[0]] - 1
        if np.any(np.abs(normalized) > 1):
            continue
        points[i] = normalized
        valid[i] = True
    values = sample_flux(psi, points).astype(np.float32)
    span = np.abs(np.median(values[:, :BOUNDARY_POINTS], axis=1) - values[:, BOUNDARY_POINTS])
    valid &= np.isfinite(span) & (span > 1e-5)
    return points, values, span.astype(np.float32), valid


def extract(path):
    table = pq.read_table(path, columns=list(COLUMNS))
    return targets({k: array_value(table[k][0]) for k in COLUMNS})


def pack(raw, packed, out, workers):
    index = json.loads((packed / 'index.json').read_text())
    if packed.name != 'train':
        raise ValueError('Geometry targets may only be prepared from the training split')
    out.mkdir(parents=True, exist_ok=False)
    n = index['frames']
    arrays = {k: np.lib.format.open_memmap(out / f'{k}.npy', mode='w+', dtype=dtype, shape=shape)
              for k, dtype, shape in [('points', np.float32, (n, BOUNDARY_POINTS + 5, 2)),
                                      ('values', np.float32, (n, BOUNDARY_POINTS + 5)),
                                      ('span', np.float32, (n,)), ('valid', bool, (n,))]}
    paths = [raw / f'{name}.parquet' for name in index['names']]
    with ProcessPoolExecutor(workers) as pool:
        for i, result in enumerate(pool.map(extract, paths, chunksize=4)):
            a, b = index['offsets'][i:i + 2]
            if len(result[0]) != b - a:
                raise ValueError(f'Geometry frame count mismatch: {paths[i].name}')
            for key, values in zip(arrays, result, strict=True):
                arrays[key][a:b] = values
            if i % 250 == 0:
                print(f'geometry {i}/{len(paths)} shots', flush=True)
    for array in arrays.values():
        array.flush()
    valid_count = int(arrays['valid'].sum())
    if not valid_count:
        raise ValueError('No usable training geometry')
    (out / 'index.json').write_text(json.dumps(index | {'valid_geometry_frames': valid_count,
                                                      'boundary_points': BOUNDARY_POINTS}))
    print(f'geometry packed: {valid_count}/{n} usable frames', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    pack(args.raw, args.packed, args.out, args.workers)


if __name__ == '__main__':
    main()
