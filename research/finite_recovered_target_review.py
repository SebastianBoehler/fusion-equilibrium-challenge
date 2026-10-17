"""Original045 evidence gates, followed by an independent full-target reload."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from participant.artifacts import digest, write_json
from participant.finite_target_runtime import RuntimePinnedTargetBatch, require_runtime
from research.finite_recovery_evidence import require_terminal_accounting, sha

ROOT = Path('run_data/analysis-055')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt_gates(path, plan):
    recovery = json.loads(path.read_text())
    require(recovery['all_bytes_recovered'] and not recovery['errors'],
            'Complete byte recovery required before numerical read')
    prefix = Path(str(path).removesuffix('-receipt.json'))
    capture = json.loads(Path(str(prefix) + '.json').read_text())
    raw = Path(str(prefix) + '.jsonl').read_bytes()
    require(capture['returncode'] == 0 and not capture['timeout']
            and len(raw) == capture['stdout_bytes'] and sha(raw) == capture['stdout_sha256'],
            'Complete original recovery capture identity required')
    del raw
    files = recovery['files']
    for name, value in files.items():
        file = Path(value['path'])
        require(file.stat().st_size == value['bytes'] and digest(file) == value['sha256'],
                f'Recovered bytes changed: {name}')
    records = {r['kind']: r['value'] for r in recovery['records_without_file_payloads']}
    own = require_terminal_accounting(records['accounting'])
    parent = next(r for r in own if r[0] == '2955114')
    require(parent[7] == '01:00:00' and parent[8] == '16' and parent[9] == '32G'
            and 'gpu' not in parent[10] + parent[11], 'Final original allocation differs')
    hours, minutes, seconds = map(int, parent[6].split(':'))
    require(hours * 3600 + minutes * 60 + seconds <= 3600, 'Original final hour cap exceeded')
    for kind, count in (('source_hashes', 25), ('input_hashes', 5)):
        value = records[kind]
        lines = value['stdout'].splitlines()
        require(value['returncode'] == 0 and not value['timed_out'] and len(lines) == count
                and all(line.endswith(': OK') for line in lines), f'Original {kind} failed')
    directory = Path(files['full-targets.json']['path']).parent
    require((directory / 'step-exit.txt').read_text().strip() == '0', 'Original step exit failed')
    allocation = json.loads((directory / 'allocation-receipt.json').read_text())
    require(allocation['returncode'] == 0, 'Original allocation query failed')
    fields = dict(t.split('=', 1) for t in allocation['stdout'].split() if '=' in t)
    expected = {'JobId': '2955114', 'JobState': 'RUNNING', 'Account': '<account>',
                'Partition': 'cpu-galvani', 'NumCPUs': '16', 'NumNodes': '1',
                'MinMemoryNode': '32G', 'TimeLimit': '01:00:00', 'Requeue': '0', 'Restarts': '0'}
    require(all(fields.get(k) == v for k, v in expected.items())
            and 'gpu' not in fields.get('ReqTRES', '') + fields.get('AllocTRES', ''),
            'Original running allocation gates failed')
    result = json.loads((directory / 'full-targets.json').read_text())
    original = json.loads(Path('run_data/analysis-045/predeclared-plan.json').read_text())
    validation = json.loads((directory / 'train-targets/validation.json').read_text())
    flags = ('complete', 'all_predeclared_target_checks_passed',
             'independent_population_variances_match', 'original_reference_unchanged',
             'full_corpus_seven_targets_validated', 'training_only')
    require(all(result.get(k) is True for k in flags)
            and result['plan_sha256'] == digest(Path('run_data/analysis-045/predeclared-plan.json'))
            and result['actual_CPU_wall_seconds'] <= 3600 and result['targets'] == validation
            and result['excluded_requests_rejected'] == 174, 'Original target completion gates failed')
    require(result['optimizer_steps'] == result['GPU_count'] == 0
            and result['predictor_inference'] is False
            and result['locked_development_or_MAST_labels_read'] is False
            and result['prediction_or_learned_artifacts_saved'] is False,
            'Original target-only isolation failed')
    require(validation['frames'] == 1469502 and validation['usable_frames'] == 1469328
            and validation['original_reference_excluded_frames'] == 174
            and validation['additional_exclusions'] == 0
            and validation['maximum_actual_li_quadrature_difference'] <= 2e-5
            and validation['file_sha256']['valid.npy'] == original['reference_files_sha256']['valid.npy'],
            'Original support/exclusion/li gates failed')
    return directory, original, result, validation, parent


def main(path):
    started = time.monotonic()
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT / 'source-freeze.json').read_text())
    require(digest(ROOT / 'predeclared-plan.json') == freeze['plan_sha256'], 'Review plan changed')
    pins = plan['source_sha256'] | freeze['source_sha256']
    for name, expected in pins.items():
        require(digest(Path(name)) == expected, f'Frozen review source changed: {name}')
    (ROOT / 'running.lock').mkdir()
    review = {'complete': False, 'recovery_receipt': str(path),
              'recovery_receipt_sha256': digest(path), 'numerical_targets_read': False,
              'optimizer_steps': 0, 'inference': False, 'GPU_count': 0, 'SSH_attempts': 0}
    output = ROOT / 'completion-gate-review.json'
    try:
        directory, original, result, validation, parent = receipt_gates(path, plan)
        require_runtime(Path.cwd(), original['source_pin'])
        review['original_receipt_resource_source_isolation_gates_passed'] = True
        index = json.loads((directory / 'train-targets/index.json').read_text())
        manifest = json.loads(Path('run_data/analysis-045/full-split.json').read_text())
        names = [Path(e['path']).stem for e in manifest['train']]
        require(index['names'] == names and len(names) == 6641
                and index['manifest_sha256'] == original['manifest_sha256'], 'Original manifest order changed')
        offsets = np.asarray(index['offsets'])
        require(len(offsets) == 6642 and offsets[0] == 0 and offsets[-1] == 1469502
                and offsets.dtype.kind in 'iu' and np.all(np.diff(offsets) > 0), 'Frame offsets changed')
        batch = RuntimePinnedTargetBatch(directory / 'train-targets', index,
            original['manifest_sha256'], result['provenance_sha256'], original['source_pin'])
        review['numerical_targets_read'] = True
        actual = batch.get(batch.valid_indices)
        require(np.isfinite(actual).all() and np.isnan(batch.values[~batch.valid]).all()
                and len(actual) == 1469328 and int((~batch.valid).sum()) == 174,
                'Recovered targets/exclusions differ')
        variances = actual.var(0)
        require(np.isfinite(variances).all() and np.all(variances > 0)
                and np.allclose(variances, batch.variances, rtol=1e-12, atol=1e-14),
                'Independent full population variance failed')
        rejected = 0
        for pick in np.flatnonzero(~batch.valid):
            try:
                batch.get(np.array([pick]))
            except ValueError as exc:
                require(str(exc) == 'Excluded finite TRAIN target requested', 'Wrong exclusion rejection')
                rejected += 1
            else:
                raise ValueError('Original excluded target accepted')
        require_runtime(Path.cwd(), original['source_pin'])
        for name, expected in pins.items():
            require(digest(Path(name)) == expected, f'Review source changed during read: {name}')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
        require(time.monotonic() - started <= 120 and rss_bytes <= 2147483648,
                'Independent local review resource gate failed')
        review.update(complete=True, original045_gates_passed=True,
            all174excluded_requests_rejected=rejected == 174,
            independent_population_variances=variances.tolist(), original_li_gate=validation['maximum_actual_li_quadrature_difference'],
            parent_accounting=parent, maximum_process_RSS_bytes=rss_bytes,
            targets_ready_for_source_pinned_reference=True, production_ready=False,
            aggregate_cluster_memory_validated=False)
    except Exception as exc:
        review.update(failed=True, failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        review['actual_wall_seconds'] = time.monotonic() - started
        write_json(output, review)
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recovery_receipt', type=Path)
    main(parser.parse_args().recovery_receipt)
