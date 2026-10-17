"""Fixed046 new preallocated-cache check on existing permitted TRAIN inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant.artifacts import digest, write_json
from participant.data import read_shot
from participant.finite_parent_cache import CHUNK_FRAMES, cache_parent
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection


ROOT = Path('run_data/analysis-046')


def rejection(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Invalid TRAIN cache request accepted')


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'cache-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': digest(plan_path), 'optimizer_steps': 0, 'GPU_count': 0,
              'true_targets_or_locked_development_MAST_labels_read': False,
              'prediction_or_learned_artifacts_saved': False,
              'source_sha256': {str(p): digest(p) for p in
                  (Path(__file__), Path('participant/finite_parent_cache.py'))},
              'limitations': plan['limitations']}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        torch.set_num_threads(plan['CPU_threads'])
        for key, name in plan['paths'].items():
            if digest(Path(name)) != plan['input_sha256'][key]:
                raise ValueError(f'Pinned046 input changed: {key}')
        for name, expected in plan['raw_input_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Existing permitted TRAIN raw source changed')
        frozen037 = json.loads(Path('run_data/analysis-038/predeclared-plan.json').read_text())['frozen037_source_sha256']
        frozen045 = json.loads(Path('run_data/analysis-045/staging-receipt.json').read_text())['source_files']
        for name, expected in (frozen037 | frozen045).items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen037/045 source changed')
        manifest = json.loads(Path(plan['paths']['TRAIN_manifest']).read_text())
        index = json.loads(Path(plan['paths']['TRAIN_index']).read_text())
        if index['names'] != [Path(entry['path']).stem for entry in manifest['train']]:
            raise ValueError('Pinned TRAIN shot order differs')
        rows = [read_shot(entry, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                         INPUT_COLUMNS) for entry in manifest['train']]
        raw = np.concatenate([rich_features(row) for row in rows])
        valid = np.load(plan['paths']['eligibility'], allow_pickle=False)
        if len(raw) != plan['frames'] or CHUNK_FRAMES != plan['chunk_frames']:
            raise ValueError('Fixed046 frame count/chunk changed')
        parent = torch.load(plan['paths']['parent'], map_location='cpu', weights_only=False)
        model = SmoothCorrection(parent).eval()
        cache = cache_parent(model, raw, valid)
        if not np.array_equal(cache.indices, np.flatnonzero(valid)):
            raise ValueError('Eligible frame cache order changed')
        params = parameters(parent, 'cpu')
        chunks = []
        with torch.no_grad():
            for start in range(0, len(cache.indices), CHUNK_FRAMES):
                end = min(start + CHUNK_FRAMES, len(cache.indices))
                inputs = prepare_inputs(raw[cache.indices[start:end]], params, model.config, 'cpu')
                fields, scalars = model(inputs)
                features = model.parent.head[0](model.parent.blocks(model.parent.stem(inputs)))
                if (not torch.equal(cache.features[start:end], features)
                        or not torch.equal(cache.fields[start:end], fields)
                        or not torch.equal(cache.scalars[start:end], scalars)
                        or not torch.equal(cache.parent_std[start:end], fields.std(1, unbiased=False))):
                    raise ValueError('New cache differs from actual per-chunk permitted-input inference')
                chunks.append(end - start)
        if chunks != [256, 252] or cache.fields.device.type != 'cpu':
            raise ValueError('Fixed cache coverage/storage changed')
        for name, value in parent['state'].items():
            if not torch.equal(model.parent.state_dict()[name], value):
                raise ValueError('Frozen parent parameter changed')
        for name in ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis', 's_mean', 's_std'):
            if not torch.equal(getattr(model, name), parent[name]):
                raise ValueError('Frozen parent statistic changed')
        bad = raw.copy()
        bad[0, 0] = np.nan
        controls = [lambda: cache_parent(model, raw.astype(np.float64), valid),
                    lambda: cache_parent(model, raw[:, :-1], valid),
                    lambda: cache_parent(model, raw, valid.astype(np.uint8)),
                    lambda: cache_parent(model, raw, valid[:-1]),
                    lambda: cache_parent(model, raw, np.zeros_like(valid)),
                    lambda: cache_parent(model, bad, valid)]
        for action in controls:
            rejection(action)
        for name, expected in (frozen037 | frozen045).items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen037/045 source changed during check')
        elapsed = time.monotonic() - started
        if elapsed > plan['hard_local_wall_seconds']:
            raise TimeoutError('Fixed cache check exceeded its cap')
        result.update(complete=True, all_predeclared_checks_passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_wall_seconds=elapsed,
            frames=len(cache.indices), chunks=chunks, permitted_INPUT_COLUMNS=list(INPUT_COLUMNS),
            cache_and_actual_per_chunk_native_fields_scalars_features_std_bit_equal=True,
            cache_payload_bytes=cache.payload_bytes(), frozen_parent_and_statistics_exact=True,
            invalid_request_controls_rejected=len(controls), frozen037_and045_sources_unchanged=True,
            full_corpus_cache_or_support_validated=False, CUDA_route_validated=False,
            peak_memory_measured=False, A100_budget_validated=False)
        write_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      failed_at=datetime.now(timezone.utc).isoformat(),
                      actual_CPU_wall_seconds=time.monotonic() - started)
        write_json(output, result)
        raise


if __name__ == '__main__':
    main()
