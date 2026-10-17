"""One guarded full TRAIN target extraction; no predictor or optimizer work."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.finite_target_runtime import (
    RuntimePinnedTargetBatch, pack_with_runtime, require_runtime,
)


def require_order(plan, manifest, index):
    names = [Path(entry['path']).stem for entry in manifest['train']]
    offsets = np.asarray(index['offsets'])
    if (len(names) != plan['TRAIN_shots'] or names != index['names']
            or index['frames'] != plan['frames'] or len(offsets) != len(names) + 1
            or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or offsets[-1] != plan['frames'] or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Full pinned TRAIN manifest/frame order differs')


def allocation(root, plan):
    job = os.environ['SLURM_JOB_ID']
    command = ['scontrol', 'show', 'job', job, '-o']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        write_json(root / 'allocation-receipt.json', {'job_id': job, 'returncode': None,
            'timed_out': True, 'stdout': decoded(exc.stdout), 'stderr': decoded(exc.stderr)})
        raise
    receipt = {'job_id': job, 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr}
    write_json(root / 'allocation-receipt.json', receipt)
    result.check_returncode()
    values = dict(token.split('=', 1) for token in result.stdout.split() if '=' in token)
    if (values.get('JobId') != job or values.get('JobState') != 'RUNNING'
            or values.get('Account') != '<account>' or values.get('Partition') != 'cpu-galvani'
            or values.get('NumCPUs') != str(plan['cpus']) or values.get('NumNodes') != '1'
            or values.get('MinMemoryNode') != plan['memory']
            or values.get('TimeLimit') != '01:00:00' or values.get('Requeue') != '0'
            or values.get('Restarts') != '0' or 'gpu' in values.get('AllocTRES', '')
            or 'gpu' in values.get('ReqTRES', '')):
        raise ValueError('Full TRAIN CPU allocation differs from declared resources')
    return receipt


def run(root):
    plan_path = root / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    (root / 'running.lock').mkdir()
    output = root / 'full-targets.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': digest(plan_path), 'training_only': True,
              'optimizer_steps': 0, 'GPU_count': 0, 'predictor_inference': False,
              'locked_development_or_MAST_labels_read': False,
              'prediction_or_learned_artifacts_saved': False,
              'limitations': plan['limitations']}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        allocation(root, plan)
        checkout = Path(__file__).resolve().parents[1]
        pin = plan['source_pin']
        require_runtime(checkout, pin)
        manifest_path = root / 'full-split.json'
        if digest(manifest_path) != plan['manifest_sha256']:
            raise ValueError('Full TRAIN manifest hash changed')
        paths = {key: Path(value) for key, value in plan['input_paths'].items()}
        reference = paths['reference_validation']
        if digest(reference) != plan['reference_validation_sha256']:
            raise ValueError('Original full TRAIN reference validation changed')
        for name, expected in plan['reference_files_sha256'].items():
            if digest(paths['contours'] / name) != expected:
                raise ValueError(f'Original full TRAIN reference changed: {name}')
        index = json.loads((paths['packed'] / 'index.json').read_text())
        manifest = json.loads(manifest_path.read_text())
        require_order(plan, manifest, index)
        destination = root / 'train-targets'
        validation = pack_with_runtime(paths['raw'], paths['packed'], paths['contours'],
            manifest_path, reference, destination, plan['CPU_workers'], pin)
        provenance_sha = digest(destination / 'runtime-validation.json')
        batch = RuntimePinnedTargetBatch(destination, index, plan['manifest_sha256'],
                                         provenance_sha, pin)
        actual = batch.get(batch.valid_indices)
        if (len(batch.valid_indices) != plan['usable_frames']
                or int((~batch.valid).sum()) != plan['original_exclusions']
                or validation['additional_exclusions'] != 0
                or not np.isnan(batch.values[~batch.valid]).all()
                or not np.allclose(actual.var(0), batch.variances, rtol=1e-12, atol=1e-14)):
            raise ValueError('Full TRAIN support, sentinels or population variances differ')
        excluded = np.flatnonzero(~batch.valid)
        for pick in excluded:
            try:
                batch.get(np.array([pick]))
            except ValueError as exc:
                if str(exc) != 'Excluded finite TRAIN target requested':
                    raise
            else:
                raise ValueError('Original excluded TRAIN target accepted')
        require_runtime(checkout, pin)
        for name, expected in plan['reference_files_sha256'].items():
            if digest(paths['contours'] / name) != expected:
                raise ValueError('Original full TRAIN reference changed during extraction')
        elapsed = time.monotonic() - started
        if elapsed > plan['hard_allocation_seconds']:
            raise TimeoutError('Full TRAIN target preparation exceeded its declared cap')
        result.update(complete=True, all_predeclared_target_checks_passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_wall_seconds=elapsed,
            targets=validation, provenance_sha256=provenance_sha,
            independent_population_variances_match=True,
            excluded_requests_rejected=len(excluded), original_reference_unchanged=True,
            full_corpus_seven_targets_validated=True,
            global_predictor_constraint_support_validated=False,
            A100_budget_validated=False, GPU_dispatched=False)
        write_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      failed_at=datetime.now(timezone.utc).isoformat(),
                      actual_CPU_wall_seconds=time.monotonic() - started)
        write_json(output, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
