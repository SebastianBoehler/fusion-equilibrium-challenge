"""One-shot, memory-mapped kinetic features and lossless float32 bulk targets."""

from pathlib import Path
import json

import numpy as np

from .artifacts import code_digest, digest, identity, write_json
from .data import SCALARS, SIGNALS, read_shot, targets
from .kinetic import kinetic_features

PREPARE_COLUMNS = ('source', 'efit_times', 'efit_psirz', *SCALARS,
                   'magnetics_time', 'magnetics_plasma_current_times',
                   *(f'magnetics_{signal}' for signal in SIGNALS),
                   *(f'thomson_{system}_{value}' for system in ('core', 'edge')
                     for value in ('times', 'Te', 'ne')))

FEATURE_FILES = [Path('participant') / name for name in
                 ('data.py', 'kinetic.py', 'cache.py', 'artifacts.py')]
FEATURE_FILES.append(Path('vendor/fusion_starter/data_fixes.py'))


def feature_identity() -> str:
    return code_digest(FEATURE_FILES)


def prepare_shot(entry: dict, revision: str, raw: Path, cache: Path) -> dict:
    provenance = {'revision': revision, 'source_sha256': entry['sha256'],
                  'path': entry['path'], 'feature_sha256': feature_identity(),
                  'configuration': {'features': 'kinetic-33', 'inputs_dtype': 'float64',
                                    'targets_dtype': 'float32', 'format': 1}}
    key = identity(provenance)
    directory = cache / key
    receipt = directory / 'receipt.json'
    if not receipt.exists():
        row = read_shot(entry, revision, raw, columns=PREPARE_COLUMNS)
        x, y = kinetic_features(row), targets(row)
        redacted = {k: v for k, v in row.items()
                    if not k.startswith('efit_') or k == 'efit_times'}
        redacted.pop('magnetics_dsep', None)
        np.testing.assert_array_equal(x, kinetic_features(redacted))
        compact = y.astype(np.float32)
        # The released labels are float32 values decoded into float64 by Arrow.
        # Refuse lossy reference storage rather than quietly changing the score.
        np.testing.assert_array_equal(y, compact.astype(np.float64))
        if not np.isfinite(x).all() or not np.isfinite(y[:, :-2]).all():
            raise ValueError(f'Invalid features or flux targets: {entry["path"]}')
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in [('inputs', x), ('targets', compact),
                             ('times', np.asarray(row['efit_times'], dtype=np.float64))]:
            temporary = directory / f'{name}.partial.npy'
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(directory / f'{name}.npy')
        write_json(receipt, {'identity': key, 'provenance': provenance,
                            'frames': len(x), 'raw_target_redaction': True,
                            'lossless_target_storage': True,
                            'files': {name: digest(directory / f'{name}.npy')
                                      for name in ('inputs', 'targets', 'times')}})
    value = json.loads(receipt.read_text())
    if value['identity'] != key or value['provenance'] != provenance:
        raise ValueError('Feature cache identity mismatch')
    for name, expected in value['files'].items():
        if digest(directory / f'{name}.npy') != expected:
            raise ValueError(f'Corrupt feature cache: {name}')
    return {**value, 'directory': directory}


def arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.load(record['directory'] / f'{name}.npy', mmap_mode='r',
                         allow_pickle=False) for name in ('inputs', 'targets'))


def blocks(records: list[dict], size: int = 2048):
    for record in records:
        x, y = arrays(record)
        for start in range(0, len(x), size):
            yield np.asarray(x[start:start + size], dtype=np.float64), \
                np.asarray(y[start:start + size], dtype=np.float64)


def input_blocks(records: list[dict], size: int = 2048):
    for record in records:
        x = np.load(record['directory'] / 'inputs.npy', mmap_mode='r', allow_pickle=False)
        for start in range(0, len(x), size):
            yield np.asarray(x[start:start + size], dtype=np.float64)
