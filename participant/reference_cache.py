"""Cache exact organizer contours and pooled reference statistics per held-out shot."""

import json
from pathlib import Path

import numpy as np

from .artifacts import code_digest, digest, identity, write_json
from .cache import arrays
from .score import official


def geometry():
    with np.load(official.HERE / 'fusion_scoring/masks/d3d_envelope.npz') as mask:
        r, z = mask['grid_R'], mask['grid_Z']
        coarse = mask['mask_coarse'].astype(bool)
    return r, z, coarse, coarse.astype(float)


def scorer_identity() -> str:
    root = Path('vendor/fusion_starter')
    files = [root / 'local_score.py', *sorted((root / 'fusion_scoring').glob('*.py')),
             root / 'fusion_scoring/masks/d3d_envelope.npz', Path(__file__)]
    return code_digest(files)


def shot(record: dict) -> dict:
    _, y = arrays(record)
    return {'psi': y[:, :-2].reshape(-1, 65, 65), 'q95': y[:, -2], 'betaN': y[:, -1]}


def statistics(gt: dict, cons: np.ndarray) -> dict:
    psi = np.asarray(gt['psi'], dtype=np.float64)
    scal = np.column_stack([gt['q95'], gt['betaN']]).astype(np.float64)
    out = {}
    for prefix, values, axis in [('psi', psi, None), ('scal', scal, 0), ('cons', cons, 0)]:
        out[prefix + '_sum'] = np.nansum(values, axis=axis)
        out[prefix + '_sumsq'] = np.nansum(values ** 2, axis=axis)
        out[prefix + '_n'] = np.isfinite(values).sum(axis=axis)
    return out


def reference(record: dict, cache: Path) -> tuple[tuple, dict]:
    provenance = {'source': record['identity'], 'targets': record['files']['targets'],
                  'scorer_sha256': scorer_identity(), 'scoring_version': official.SCORING_VERSION}
    key = identity(provenance)
    path = cache / key
    receipt = path / 'receipt.json'
    if not receipt.exists():
        gt = shot(record)
        contours, cons, valid, rgeo = official.build_reference(gt['psi'], *geometry())
        lengths = [len(c) if c is not None else 0 for c in contours]
        offsets = np.r_[0, np.cumsum(lengths)]
        present = [c for c in contours if c is not None]
        points = np.concatenate(present) if present else np.empty((0, 2))
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / 'reference.partial.npz'
        np.savez_compressed(temporary, points=points, offsets=offsets, cons=cons,
                            valid=valid, rgeo=rgeo, **statistics(gt, cons))
        temporary.replace(path / 'reference.npz')
        write_json(receipt, {'identity': key, 'provenance': provenance,
                            'file_sha256': digest(path / 'reference.npz')})
    value = json.loads(receipt.read_text())
    if value['identity'] != key or value['provenance'] != provenance:
        raise ValueError('Reference cache identity mismatch')
    if digest(path / 'reference.npz') != value['file_sha256']:
        raise ValueError('Corrupt reference cache')
    with np.load(path / 'reference.npz', allow_pickle=False) as data:
        offsets, points = data['offsets'], data['points']
        contours = [points[a:b] if b > a else None for a, b in zip(offsets[:-1], offsets[1:])]
        ref = contours, data['cons'], data['valid'], float(data['rgeo'])
        stats = {k: data[k] for k in data.files if k.endswith(('_sum', '_sumsq', '_n'))}
    return ref, stats
