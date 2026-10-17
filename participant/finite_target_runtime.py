"""Pinned, portable provenance for actual TRAIN loss targets; no inference use."""

from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import sys

from .artifacts import digest, identity, write_json
from .finite_target_batch import FiniteTargetBatch


MODULES = ('artifacts', 'data', 'score', 'fast_score', 'finite_metrics',
           'reference_cache', 'finite_target_batch', 'finite_target_packing',
           'finite_target_runtime', 'cache', 'kinetic', 'quadratic', 'ridge')
SCORER_MODULES = ('common', 'contour', 'o_point', 'lcfs', 'derive', 'metrics')


def runtime_identity(root):
    root = Path(root).resolve()
    scoring = root / 'vendor/fusion_starter/fusion_scoring'
    files = [root / f'participant/{name}.py' for name in MODULES]
    files += [root / 'participant/__init__.py',
              root / 'vendor/fusion_starter/data_fixes.py',
              root / 'vendor/fusion_starter/local_score.py',
              scoring / 'masks/d3d_envelope.npz', *sorted(scoring.glob('*.py'))]
    if not tuple(scoring.glob('*.py')):
        raise ValueError('Missing actual TRAIN scoring source')
    body = {'schema': 1,
            'source_sha256': {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)},
            'dependencies': {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
                             'numpy': version('numpy'), 'pyarrow': version('pyarrow'),
                             'huggingface-hub': version('huggingface-hub')}}
    return body | {'identity_sha256': identity(body)}


def require_runtime(root, pin):
    root = Path(root).resolve()
    if runtime_identity(root) != pin:
        raise ValueError('Actual TRAIN target runtime pin changed')
    if Path(import_module('participant').__file__).resolve() != root / 'participant/__init__.py':
        raise ValueError('TRAIN target package imported from another checkout')
    for name in MODULES:
        module = import_module(f'participant.{name}')
        if Path(module.__file__).resolve() != root / f'participant/{name}.py':
            raise ValueError('TRAIN target module imported from another checkout')
    helper = import_module('vendor.fusion_starter.data_fixes')
    if Path(helper.__file__).resolve() != root / 'vendor/fusion_starter/data_fixes.py':
        raise ValueError('TRAIN data helper imported from another checkout')
    official = import_module('vendor.fusion_starter.local_score')
    if Path(official.__file__).resolve() != root / 'vendor/fusion_starter/local_score.py':
        raise ValueError('TRAIN scorer imported from another checkout')
    for name in SCORER_MODULES:
        if Path(sys.modules[name].__file__).resolve() != root / f'vendor/fusion_starter/fusion_scoring/{name}.py':
            raise ValueError('TRAIN scoring component imported from another checkout')
    return pin


def _write_runtime_validation(path, pin):
    validation = json.loads((path / 'validation.json').read_text())
    index = json.loads((path / 'index.json').read_text())
    if not validation['complete'] or not validation['training_only']:
        raise ValueError('Complete TRAIN target validation required for provenance')
    value = {'complete': True, 'training_only': True, 'schema': 1,
             'runtime': pin, 'manifest_sha256': validation['manifest_sha256'],
             'index_sha256': digest(path / 'index.json'),
             'validation_sha256': digest(path / 'validation.json')}
    if index['manifest_sha256'] != value['manifest_sha256']:
        raise ValueError('TRAIN target provenance manifest changed')
    out = path / 'runtime-validation.json'
    if out.exists():
        raise FileExistsError('TRAIN runtime validation already exists')
    write_json(out, value)
    return value


class RuntimePinnedTargetBatch(FiniteTargetBatch):
    def __init__(self, path, train_index, manifest_sha256, provenance_sha256, source_pin):
        root = Path(__file__).resolve().parents[1]
        require_runtime(root, source_pin)
        provenance_path = path / 'runtime-validation.json'
        if digest(provenance_path) != provenance_sha256:
            raise ValueError('TRAIN target runtime provenance hash changed')
        value = json.loads(provenance_path.read_text())
        if (value.get('schema') != 1 or not value.get('complete') or not value.get('training_only')
                or value['runtime'] != source_pin or value['manifest_sha256'] != manifest_sha256
                or value['index_sha256'] != digest(path / 'index.json')
                or value['validation_sha256'] != digest(path / 'validation.json')):
            raise ValueError('TRAIN target runtime provenance differs')
        super().__init__(path, train_index, manifest_sha256)


def pack_with_runtime(raw, packed, contours, split_path, reference_validation, out, workers, source_pin):
    from .finite_target_packing import pack

    root = Path(__file__).resolve().parents[1]
    require_runtime(root, source_pin)
    result = pack(raw, packed, contours, split_path, reference_validation, out, workers)
    require_runtime(root, source_pin)
    _write_runtime_validation(out, source_pin)
    return result
