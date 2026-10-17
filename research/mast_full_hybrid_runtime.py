"""Pinned MAST016 identity, input, geometry and bounded runtime checks."""
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_inputs import COLUMNS, SIGNALS
from research.mast_unit_current import unit_group

ROOT = Path('run_data/mast-016')
GEOMETRY = ('coil_input_column', 'coil_R', 'coil_Z', 'coil_width', 'coil_height',
            'efit_grid_R', 'efit_grid_Z')


def require(value, message):
    if not value: raise ValueError(message)


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def identities(plan, freeze, deadline):
    require(digest(ROOT/'predeclared-plan.json') == freeze['plan_sha256'], '016plan changed')
    pins = plan['source_sha256'] | freeze['source_sha256'] | plan['input_cache_sha256']
    pins.update({plan[k+'_path']: plan[k+'_sha256'] for k in ('predictions', 'manifest', 'order')})
    for name, expected in pins.items():
        deadline(); require(digest(Path(name)) == expected, '016identity changed: '+name)
    require(sys.version == freeze['Python'] and sys.executable == freeze['executable'], '016Python changed')
    for name, version in freeze['dependency_versions'].items():
        require(importlib.import_module(name).__version__ == version, '016dependency changed: '+name)
    for name, origin in freeze['module_origins'].items():
        require(str(Path(importlib.import_module(name).__file__).resolve()) == origin, '016origin changed: '+name)


def inputs(entry, plan):
    path = Path('run_data/mast-001/inputs')/(Path(entry['path']).stem+'.npz')
    require(digest(path) == plan['input_cache_sha256'][str(path)], '016current cache identity changed')
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(COLUMNS)|{'revision', 'source_hash'}
            and str(data['revision']) == plan['dataset_revision']
            and str(data['source_hash']) == entry['sha256'], '016input allowlist/provenance changed')
        row = {key: data[key] for key in data.files}
    require(str(row['source']) == 'MAST', '016source machine changed')
    for signal in SIGNALS:
        axis = row['magnetics_time']; values = row['magnetics_'+signal]
        axis = axis[np.isfinite(axis)&np.isfinite(values)]
        require(len(axis) >= 2 and np.all(np.diff(axis)>0)
            and np.all(row['efit_times'] >= axis[0]) and np.all(row['efit_times'] <= axis[-1]),
            '016current bracketing failed: '+Path(entry['path']).stem+'/'+signal)
    require(np.isfinite(row['efit_times']).all() and np.all(np.diff(row['efit_times'])>0), '016frame times invalid')
    return row


def geometry_hash(row):
    identity = hashlib.sha256()
    for key in GEOMETRY:
        a = row[key]; identity.update(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes())
    return identity.hexdigest()


def unit_evidence(row, identity, deadline):
    previous = json.loads(Path('run_data/mast-013/unit-current-check.json').read_text())
    for shot in previous['shots']:
        if shot['exact_geometry_sha256'] == identity:
            return {'origin': 'immutable013 exact geometry', 'groups': shot['groups']}
    names = np.asarray(row['coil_input_column']).astype(str); groups = {}
    for name in np.unique(names):
        selected = names == name; normal = 1 if name == 'magnetics_sol_current' else selected.sum()
        value = unit_group(*(row[k][selected] for k in ('coil_R', 'coil_Z', 'coil_width', 'coil_height')),
            row['efit_grid_R'], row['efit_grid_Z'], normal, {}, deadline)
        require(value['direct_exact_passed'] and value['paired_passed'], '016new geometry unit-current gate failed: '+name)
        groups[name] = value
    return {'origin': 'new exact geometry original013 oracle', 'groups': groups}


class Budget:
    def __init__(self, plan, result):
        self.start = time.monotonic(); self.plan = plan; self.result = result

    def check(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss*1024
        self.result['maximum_process_RSS_bytes'] = rss
        self.result['actual_wall_seconds'] = time.monotonic()-self.start
        require(self.result['actual_wall_seconds'] <= self.plan['resources']['hard_local_seconds']
            and rss <= self.plan['resources']['max_process_RSS_bytes'], '016original resource cap exceeded')
