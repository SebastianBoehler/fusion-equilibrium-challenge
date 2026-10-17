"""Fixed043 analyzed closure/relocation revision; no new target extraction."""

from datetime import datetime, timezone
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.finite_target_runtime import (RuntimePinnedTargetBatch,
    _write_runtime_validation, require_runtime, runtime_identity)

ROOT = Path('run_data/analysis-043')


def rejected(action):
    try:
        action()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid runtime or target provenance accepted')


def relocated_process(copied, cache, provenance_sha, receipt_name):
    code = '''import json,sys
from pathlib import Path
from participant.finite_target_runtime import require_runtime,RuntimePinnedTargetBatch
pin=json.loads(Path(sys.argv[1]).read_text())
require_runtime(Path.cwd(),pin)
cache=Path(sys.argv[2]);index=json.loads((cache/'index.json').read_text())
batch=RuntimePinnedTargetBatch(cache,index,index['manifest_sha256'],sys.argv[3],pin)
assert len(batch.valid_indices)==508
print(json.dumps({'relocated_import_and_guard_passed':True,'frames':508}))
'''
    command = [sys.executable, '-c', code, str((ROOT / 'source-pin.json').resolve()),
               str(cache.resolve()), provenance_sha]
    try:
        child = subprocess.run(command, cwd=copied, capture_output=True,
                               text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        write_json(ROOT / receipt_name, {'timed_out': True,
            'stdout': str(exc.stdout), 'stderr': str(exc.stderr), 'timeout_seconds': 30})
        raise
    write_json(ROOT / receipt_name, {'returncode': child.returncode,
        'stdout': child.stdout, 'stderr': child.stderr, 'timeout_seconds': 30})
    return child


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    (ROOT / 'running.lock').mkdir()
    result_path = ROOT / 'runtime-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(plan_path), 'optimizer_steps': 0, 'GPU_count': 0,
        'predictor_inference': False, 'new_target_extractions': 0,
        'source_sha256': {str(p): digest(p) for p in (Path(__file__),
                            Path('participant/finite_target_runtime.py'))},
        'limitations': plan['limitations']}
    with result_path.open('x') as handle:
        json.dump(result, handle)
    start = time.monotonic()
    try:
        for name, expected_sha in plan['prior_frozen_source_sha256'].items():
            if digest(Path('run_data/analysis-042/frozen-source') / name) != expected_sha:
                raise ValueError('Failed042 frozen source snapshot changed')
        for name, expected in plan['frozen037_source_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen037 source changed')
        original = Path(plan['existing_TRAIN_fixture'])
        for name, expected in plan['existing_fixture_sha256'].items():
            if digest(original / name) != expected:
                raise ValueError('Immutable038 TRAIN fixture changed')
        pin = runtime_identity(Path.cwd())
        require_runtime(Path.cwd(), pin)
        write_json(ROOT / 'source-pin.json', pin)
        copied = ROOT / 'relocated-source'
        copied.mkdir()
        files = set(pin['source_sha256'])
        for name in sorted(files):
            destination = copied / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, destination)
        if runtime_identity(copied) != pin:
            raise AssertionError('Actual source relocation changed portable identity')
        cache = ROOT / 'copied-TRAIN-fixture'
        shutil.copytree(original, cache)
        _write_runtime_validation(cache, pin)
        provenance_sha = digest(cache / 'runtime-validation.json')
        index = json.loads((cache / 'index.json').read_text())
        manifest_sha = index['manifest_sha256']
        args = (cache, index, manifest_sha, provenance_sha, pin)
        batch = RuntimePinnedTargetBatch(*args)
        expected = np.load(original / 'values.npy', allow_pickle=False)
        if (len(batch.values) != plan['frames_in_existing_TRAIN_fixture'] or
                not np.array_equal(batch.get(batch.valid_indices), expected[batch.valid]) or
                not np.array_equal(batch.valid, np.load(original / 'valid.npy', allow_pickle=False)) or
                not np.array_equal(batch.variances,
                    json.loads((original / 'validation.json').read_text())['statistics']['variance'])):
            raise AssertionError('Guarded copied TRAIN fixture changed')
        child = relocated_process(copied, cache, provenance_sha, 'relocated-subprocess.json')
        child.check_returncode()
        child_result = json.loads(child.stdout)
        controls = []
        for label, relative, mode in (
                ('scoring_source_drift', 'vendor/fusion_starter/fusion_scoring/common.py', 'append'),
                ('D3D_geometry_drift', 'vendor/fusion_starter/fusion_scoring/masks/d3d_envelope.npz', 'flip'),
                ('missing_scoring_file', 'vendor/fusion_starter/fusion_scoring/common.py', 'remove'),
                ('data_fixes_content_drift', 'vendor/fusion_starter/data_fixes.py', 'append'),
                ('cache_content_drift', 'participant/cache.py', 'append'),
                ('kinetic_content_drift', 'participant/kinetic.py', 'append'),
                ('missing_data_fixes_helper', 'vendor/fusion_starter/data_fixes.py', 'remove')):
            p = copied / relative
            saved = p.read_bytes()
            if mode == 'remove':
                p.unlink()
            else:
                p.write_bytes(saved + b'\n#043 control\n' if mode == 'append'
                              else bytes([saved[0] ^ 1]) + saved[1:])
            rejected(lambda: require_runtime(copied, pin))
            if label == 'missing_data_fixes_helper':
                invalid = relocated_process(copied, cache, provenance_sha, 'rejected-subprocess.json')
                if invalid.returncode == 0 or not invalid.stderr or 'data_fixes.py' not in invalid.stderr:
                    raise AssertionError('Missing helper subprocess rejection/diagnostic differs')
                controls.append('fresh_subprocess_missing_helper_diagnostics')
            p.write_bytes(saved)
            controls.append(label)
        extra = copied / 'vendor/fusion_starter/fusion_scoring/unexpected043.py'
        extra.write_text('# actual source-set rejection control\n')
        rejected(lambda: require_runtime(copied, pin))
        extra.unlink()
        controls.append('added_scoring_file')
        malformed = dict(pin, identity_sha256='0' * 64)
        rejected(lambda: require_runtime(Path.cwd(), malformed))
        controls.append('malformed_source_pin')
        changed = json.loads(json.dumps(pin))
        changed['dependencies']['numpy'] = 'different043'
        rejected(lambda: require_runtime(Path.cwd(), changed))
        controls.append('different_dependency_identity')
        rejected(lambda: require_runtime(copied, pin))
        controls.append('wrong_imported_checkout_root')
        helper_name = 'vendor.fusion_starter.data_fixes'
        original_helper = sys.modules[helper_name]
        spec = spec_from_file_location(helper_name, copied / 'vendor/fusion_starter/data_fixes.py')
        copied_helper = module_from_spec(spec)
        spec.loader.exec_module(copied_helper)
        try:
            sys.modules[helper_name] = copied_helper
            rejected(lambda: require_runtime(Path.cwd(), pin))
        finally:
            sys.modules[helper_name] = original_helper
        controls.append('actual_data_fixes_imported_from_copied_root')
        for name in ('runtime-validation.json', 'validation.json', 'index.json'):
            p = cache / name
            saved = p.read_bytes()
            p.write_bytes(saved + b' ')
            rejected(lambda: RuntimePinnedTargetBatch(*args))
            p.write_bytes(saved)
            controls.append('changed_' + name)
        if len(controls) != plan['fixed_negative_control_count'] or runtime_identity(copied) != pin:
            raise AssertionError('Fixed runtime controls or restored copied source differ')
        for name, expected_sha in plan['existing_fixture_sha256'].items():
            if digest(original / name) != expected_sha or digest(cache / name) != expected_sha:
                raise AssertionError('Actual038 target fixture hashes changed')
        for name, expected_sha in plan['frozen037_source_sha256'].items():
            if digest(Path(name)) != expected_sha:
                raise AssertionError('Frozen037 source changed during runtime check')
        for name, expected_sha in plan['prior_frozen_source_sha256'].items():
            if digest(Path('run_data/analysis-042/frozen-source') / name) != expected_sha:
                raise AssertionError('Failed042 frozen source snapshot changed during check')
        elapsed = time.monotonic() - start
        if elapsed > plan['hard_local_wall_seconds']:
            raise TimeoutError('Fixed local provenance check exceeded its cap')
        result.update(complete=True, all_predeclared_checks_passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_seconds=elapsed,
            runtime_identity=pin, actual_source_relocation_identity_exact=True,
            fresh_relocated_subprocess=child_result, rejected_controls=controls,
            actual508_copied_target_values_and_original_hashes_exact=True,
            frozen037_sources_unchanged=True, fixture_provenance_sha256=provenance_sha,
            failed042_source_snapshots_unchanged=True,
            fixture_only=True, production_pack_wrapper_executed=False,
            full_corpus_targets_or_runtime_validated=False, GPU_dispatched=False)
        write_json(result_path, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
            failed_at=datetime.now(timezone.utc).isoformat(), actual_CPU_seconds=time.monotonic() - start)
        write_json(result_path, result)
        raise


if __name__ == '__main__':
    main()
