"""Ordered process-parallel measurements through the unchanged local extractor."""

import numpy as np

from . import fast_score


def derive_block(job):
    start, fields = job
    geometry = fast_score._geometry()
    r, z, coarse, mask = geometry
    official = fast_score.official
    values = []
    for offset, field in enumerate(np.asarray(fields, dtype=np.float64)):
        contour, _ = official.extract_lcfs_with_sign(field, r, z, 'DIII-D', coarse, mask,
            n_iter=official.N_ITER, axis_sign=official.AXIS_SIGN['DIII-D'])
        if contour is None or not np.isfinite(contour).all():
            raise FloatingPointError(f'Invalid actual contour at TRAIN frame{start + offset}')
        derived = official.derive_frame(field, r, z, 'DIII-D', coarse, mask,
            contour=contour, axis_sign=official.AXIS_SIGN['DIII-D'])
        row = [derived[key] for key in official.CONS_SCALARS]
        if not np.isfinite(row).all():
            raise FloatingPointError(f'Invalid actual derived value at TRAIN frame{start + offset}')
        values.append(row)
    return start, np.asarray(values)


def measure_derived(fields, executor=None):
    arrays = np.asarray(fields, dtype=np.float32).reshape(-1, 65, 65)
    if not len(arrays) or not np.isfinite(arrays).all():
        raise ValueError('Actual measurements require nonempty finite float32 fields')
    jobs = [(start, arrays[start:start + 32]) for start in range(0, len(arrays), 32)]
    results = map(derive_block, jobs) if executor is None else executor.map(derive_block, jobs)
    parts = []
    for (start, block), (returned_start, values) in zip(jobs, results, strict=True):
        if returned_start != start or values.shape != (len(block), 7):
            raise ValueError('Actual measurement order or component count changed')
        parts.append(values)
    return np.concatenate(parts)
