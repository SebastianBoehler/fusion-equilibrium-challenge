"""Fixed047 disk persistence check; no labels, optimizer or GPU."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant.artifacts import digest, write_json
from participant.data import read_shot
from participant.finite_disk_cache import cache_parent_disk
from participant.finite_parent_cache import CHUNK_FRAMES, cache_parent
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection


ROOT = Path('run_data/analysis-047')


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    (ROOT / 'running.lock').mkdir()
    result = {'complete': False, 'optimizer_steps': 0, 'GPU_count': 0,
              'labels_or_heldout_read': False, 'diagnostic_parent_cache_saved': True,
              'submission_or_learned_artifacts_saved': False,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': digest(ROOT / 'predeclared-plan.json'),
              'limitations': plan['limitations']}
    output = ROOT / 'disk-cache-check.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        torch.set_num_threads(plan['CPU_threads'])
        frozen = json.loads(Path('run_data/analysis-038/predeclared-plan.json').read_text())['frozen037_source_sha256']
        frozen |= json.loads(Path('run_data/analysis-045/staging-receipt.json').read_text())['source_files']
        frozen |= json.loads(Path('run_data/analysis-046/cache-check.json').read_text())['source_sha256']
        for name, expected in frozen.items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen source changed before047')
        for key, name in plan['paths'].items():
            if digest(Path(name)) != plan['input_sha256'][key]:
                raise ValueError('Pinned047 input changed')
        for name, expected in plan['raw_input_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Pinned TRAIN raw source changed')
        manifest = json.loads(Path(plan['paths']['TRAIN_manifest']).read_text())
        index = json.loads(Path(plan['paths']['TRAIN_index']).read_text())
        if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
            raise ValueError('Pinned TRAIN order changed')
        rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                          INPUT_COLUMNS) for e in manifest['train']]
        raw = np.concatenate([rich_features(row) for row in rows])
        valid = np.load(plan['paths']['eligibility'], allow_pickle=False)
        if len(raw) != plan['frames']:
            raise ValueError('Fixed047 support changed')
        parent = torch.load(plan['paths']['parent'], map_location='cpu', weights_only=False)
        model = SmoothCorrection(parent).eval()
        destination = ROOT / 'diagnostic-cache'
        cache = cache_parent_disk(model, raw, valid, destination, plan['input_sha256'])
        indices = np.flatnonzero(valid)
        np.testing.assert_array_equal(cache.indices, indices)
        for start in range(0, len(indices), CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, len(indices))
            values = raw[indices[start:end]]
            reference = cache_parent(model, values, np.ones(len(values), dtype=np.bool_))
            for key in ('features', 'fields', 'scalars', 'parent_std'):
                if not torch.equal(getattr(cache, key)[start:end], getattr(reference, key)):
                    raise ValueError('Disk cache differs from unchanged046 chunk reference')
        stored = np.load(destination / 'parent-fields.npy', mmap_mode='r', allow_pickle=False)
        np.testing.assert_array_equal(stored, cache.fields.numpy())
        receipt = json.loads((destination / 'receipt.json').read_text())
        if (receipt['fields_sha256'] != digest(destination / 'parent-fields.npy')
                or receipt['chunks'] != [256, 252]
                or receipt['provenance'] != plan['input_sha256']):
            raise ValueError('Disk cache cold storage/provenance mismatch')
        np.testing.assert_array_equal(np.load(destination / 'indices.npy'), indices)
        bad = raw.copy()
        bad[0, 0] = np.nan
        controls = [(raw.astype(np.float64), valid), (raw[:, :-1], valid),
                    (raw, valid.astype(np.uint8)), (raw, valid[:-1]),
                    (raw, np.zeros_like(valid)), (bad, valid)]
        for i, (values, mask) in enumerate(controls):
            try:
                cache_parent_disk(model, values, mask, ROOT / f'rejected-{i}', {})
            except ValueError:
                pass
            else:
                raise ValueError('Malformed disk-cache input accepted')
        try:
            cache_parent_disk(model, raw, valid, destination, {})
        except FileExistsError:
            pass
        else:
            raise ValueError('Existing disk cache overwritten')
        for name, value in parent['state'].items():
            if not torch.equal(model.parent.state_dict()[name], value):
                raise ValueError('Frozen parent parameter changed')
        for name in ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis', 's_mean', 's_std'):
            if not torch.equal(getattr(model, name), parent[name]):
                raise ValueError('Frozen parent statistic changed')
        for name, expected in frozen.items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen source changed during047')
        elapsed = time.monotonic() - started
        if elapsed > plan['hard_local_wall_seconds']:
            raise TimeoutError('Fixed047 local CPU cap exceeded')
        result.update(complete=True, checks_passed=True, actual_body_seconds=elapsed,
                      frames=len(indices), disk_receipt=receipt, rejected_controls=7,
                      frozen_sources_unchanged=True, source_sha256={str(p): digest(p) for p in
                          [Path('participant/finite_disk_cache.py'), Path(__file__).relative_to(Path.cwd())]})
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      actual_body_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output, result)


if __name__ == '__main__':
    main()
