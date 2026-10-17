"""Original045 scientific gates after four disjoint byte-recovery phases."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from participant.artifacts import digest, write_json
from participant.finite_target_runtime import RuntimePinnedTargetBatch, require_runtime
from research.finite_recovered_target_review import require, receipt_gates
from research.finite_recovery_evidence import sha
ROOT = Path('run_data/analysis-057')


def transport_gates(path):
    prefix = Path(str(path).removesuffix('-receipt.json'))
    capture = json.loads(Path(str(prefix)+'.json').read_text())
    require(capture['kind'] == 'four_complete_disjoint_phase_captures', 'Wrong transport identity')
    require(len(capture['phase_receipts']) == 4, 'Missing disjoint recovery phase')
    for phase, name in enumerate(capture['phase_receipts']):
        saved = json.loads(Path(name).read_text())
        review = json.loads(Path(f'run_data/analysis-056/phase-{phase}-review.json').read_text())
        raw = Path(f'run_data/analysis-056/phase-{phase}.jsonl').read_bytes()
        require(saved['returncode'] == 0 and not saved['timeout'] and review['complete']
                and len(raw) == saved['stdout_bytes'] and sha(raw) == saved['stdout_sha256'],
                'Incomplete or changed phase evidence')
    recovery = json.loads(path.read_text())
    require(len(recovery['files']) == 18, 'Expected all18 original target/receipt files')


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
        transport_gates(path)
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
