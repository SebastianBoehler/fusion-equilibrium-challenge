"""One predeclared038 TRAIN target-cache route check; zero optimizer steps."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.data import read_shot
from participant.fast_score import _geometry, official
from participant.finite_target_batch import FiniteTargetBatch, usable_indices
from participant.finite_target_packing import pack

ROOT = Path('run_data/analysis-038')


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['frozen037_source_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen037 source changed')
    for key in ('input_manifest', 'reference_validation', 'original_full_mask'):
        if digest(Path(plan[key])) != plan[key + '_sha256']:
            raise ValueError(f'Pinned038 input changed: {key}')
    (ROOT / 'running.lock').mkdir()
    result_path = ROOT / 'target-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(plan_path), 'training_only': True, 'optimizer_steps': 0, 'GPU_count': 0,
        'true_targets_saved_for_TRAIN_loss_only': True, 'prediction_or_learned_artifacts_saved': False,
        'locked_development_or_MAST_labels_read': False, 'limitations': plan['limitations'],
        'source_sha256': {str(p): digest(p) for p in (Path(__file__),
            Path('participant/finite_target_batch.py'), Path('participant/finite_target_packing.py'))}}
    with result_path.open('x') as handle:
        json.dump(result, handle)
    start = time.monotonic()
    try:
        manifest = json.loads(Path(plan['input_manifest']).read_text())
        index = json.loads((Path(plan['packed']) / 'index.json').read_text())
        if index['frames'] != plan['frames'] or len(index['names']) != plan['shots']:
            raise ValueError('Fixed038 TRAIN sample changed')
        out = ROOT / 'train-targets'
        validation = pack(Path(plan['raw']), Path(plan['packed']), Path(plan['reference_contours']),
            Path(plan['input_manifest']), Path(plan['reference_validation']), out, plan['CPU_workers'])
        rows = [read_shot(entry, manifest['revision'], Path(plan['raw']),
            ('source', 'efit_psirz')) for entry in manifest['train']]
        truth = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
        expected = official.build_reference(truth.astype(np.float64), *_geometry())[1]
        batch = FiniteTargetBatch(out, index, plan['input_manifest_sha256'])
        actual = batch.get(np.arange(plan['frames']))
        if not np.array_equal(actual, expected):
            raise ValueError('Actual TRAIN target cache differs from independent scorer reference')
        if not np.allclose(batch.variances, expected.var(0), rtol=1e-12, atol=1e-14):
            raise ValueError('TRAIN target population variances differ')
        original = np.load(plan['original_full_mask'], mmap_mode='r', allow_pickle=False)
        if (original.shape != (plan['original_full_mask_frames'],) or original.dtype != np.bool_ or
                int(original.sum()) != plan['original_full_mask_usable'] or
                int((~original).sum()) != plan['original_full_mask_excluded']):
            raise ValueError('Original full TRAIN eligibility mask changed')
        excluded = np.flatnonzero(~original)
        for index_value in excluded:
            try:
                usable_indices(original, np.array([index_value]))
            except ValueError as exc:
                if str(exc) != 'Excluded finite TRAIN target requested':
                    raise
            else:
                raise ValueError('Original excluded TRAIN target accepted')
        for invalid in (np.array([], dtype=int), np.array([-1]), np.array([len(original)])):
            try:
                usable_indices(original, invalid)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid target index request accepted')
        for path, sha in plan['frozen037_source_sha256'].items():
            if digest(Path(path)) != sha:
                raise ValueError('Frozen037 source changed during target check')
        elapsed = time.monotonic() - start
        if elapsed > plan['hard_local_wall_seconds']:
            raise TimeoutError('Target check exceeded predeclared local CPU cap')
        result.update(complete=True, all_predeclared_target_checks_passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_seconds=elapsed,
            targets=validation, all508_cache_reference_values_bit_exact=True,
            population_variances_match=True, original174_excluded_requests_rejected=len(excluded),
            invalid_index_guards_passed=True, frozen037_sources_unchanged=True,
            full_corpus_target_validity_validated=False, Galvani_CPU_cost_established=False,
            A100_budget_validated=False, GPU_dispatched=False)
        write_json(result_path, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
            failed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_seconds=time.monotonic() - start)
        write_json(result_path, result)
        raise


if __name__ == '__main__':
    main()
