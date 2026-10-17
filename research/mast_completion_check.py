"""Single MAST017 disjoint suffix execution with full immutable prefix binding."""
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_green import coil_kernel
from research.mast_full_hybrid_runtime import (require, inputs, geometry_hash,
    unit_evidence, Budget, array_hash)
from research.mast_full_hybrid_shot import audit
from research.mast_completion_prefix import prefix_shots

ROOT = Path('run_data/mast-017')


def identities(plan, freeze, deadline):
    require(digest(ROOT/'predeclared-plan.json') == freeze['plan_sha256'], '017plan changed')
    pins = plan['source_sha256'] | freeze['source_sha256'] | plan['input_cache_sha256']
    pins.update({plan[k+'_path']: plan[k+'_sha256'] for k in ('predictions', 'manifest', 'order')})
    pins.update({plan[k+'_path']: plan[k+'_sha256'] for k in ('prefix_status', 'interruption_analysis')})
    pins.update({v['path']:v['sha256'] for v in plan['prefix_record_files']})
    for name, expected in pins.items():
        deadline(); require(digest(Path(name)) == expected, '017identity changed: '+name)
    require(sys.version == freeze['Python'] and sys.executable == freeze['executable'], '017Python changed')
    for name, version in freeze['dependency_versions'].items():
        require(importlib.import_module(name).__version__ == version, '017dependency changed')
    for name, origin in freeze['module_origins'].items():
        require(str(Path(importlib.import_module(name).__file__).resolve()) == origin, '017module origin changed')


def save_q(index, q):
    path = ROOT/'q95'/f'shot_{index:04d}.npy'
    with path.open('xb') as stream: np.save(stream, q, allow_pickle=False)
    saved = np.load(path, allow_pickle=False)
    require(saved.dtype == np.float32 and np.isfinite(saved).all() and np.all(saved>0)
            and np.array_equal(saved, q), '017per-shot diagnostic export/reload failed')
    return {'index': index, 'frames': len(q), 'q_sha256': array_hash(q), 'npy_sha256': digest(path)}


def main():
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    (ROOT/'running.lock').mkdir(); (ROOT/'q95').mkdir(); (ROOT/'geometry-evidence').mkdir()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(), 'process_PID': os.getpid(),
              'shots': [], 'prefix_validated_frames': 0, 'suffix_attempted_frames': 0, 'failed_suffix_frames': 0,
              'MAST_labels_demos_envelopes_read': False, 'optimizer_steps': 0, 'GPU_count': 0,
              'cluster_allocations': 0, 'old016_rerun_or_resume': False, 'competition_candidate_or_upload': False}
    budget = Budget(plan, result)
    try:
        identities(plan, freeze, budget.check)
        manifest = json.loads(Path(plan['manifest_path']).read_text()); order = json.loads(Path(plan['order_path']).read_text())
        require(manifest['shots'] == plan['entries'] and len(plan['entries']) == plan['expected_shots']
            and order['ordered_paths'] == [v['path'] for v in plan['entries']], '017full input order changed')
        for index, q, summary in prefix_shots(plan, budget.check):
            result['prefix_validated_frames'] += len(q)
            if index < plan['suffix_start_shot']:
                result['shots'].append(save_q(index, q) | summary | {'origin': 'immutable016prefix'})
            else: partial_q, partial_summary = q, summary
        require(result['prefix_validated_frames'] == plan['prefix_frames'], '017prefix coverage incomplete')
        write_json(ROOT/'full-audit.json', result)
        print(f'017 immutable prefix verified: {result["prefix_validated_frames"]} frames', flush=True)
        last_geometry = None
        with np.load(plan['predictions_path'], allow_pickle=False) as stored:
            for index in range(plan['suffix_start_shot'], plan['expected_shots']):
                budget.check(); row = inputs(plan['entries'][index], plan); offset = plan['suffix_start_frame'] if index == plan['suffix_start_shot'] else 0
                identity = geometry_hash(row)
                if identity != last_geometry:
                    kernels = coil_kernel(row); units = unit_evidence(row, identity, budget.check); last_geometry = identity
                    write_json(ROOT/'geometry-evidence'/f'{identity}.json', units)
                prefix = f'shot_{index:04d}_'; psi, oldq, beta = (stored[prefix+k] for k in ('psirz', 'q95', 'betaN'))
                full_hashes = [array_hash(a) for a in (psi, beta)]; n = len(row['efit_times'])
                require(0 <= offset < n, '017suffix offset outside original shot')
                if offset:
                    require(np.array_equal(row['efit_times'][:offset], partial_summary['prefix_times_ms']), '017partial frame time binding changed')
                row['efit_times'] = row['efit_times'][offset:]
                with (ROOT/f'frames-{index:04d}.jsonl').open('x') as stream:
                    def emit(value):
                        value = value | {'index': value['index']+offset}
                        stream.write(json.dumps(value)+'\n'); stream.flush(); result['suffix_attempted_frames'] += 1
                        write_json(ROOT/'progress.json', {'shot_index': index, 'frame_index': value['index'],
                            'suffix_attempted_frames': result['suffix_attempted_frames'], 'prefix_validated_frames': result['prefix_validated_frames'],
                            'process_PID': os.getpid(), 'actual_wall_seconds': result['actual_wall_seconds'],
                            'observed_at': datetime.now(timezone.utc).isoformat()})
                    q, detail = audit(row, psi[offset:], oldq[offset:], beta[offset:], kernels, units, plan, budget.check, emit)
                require(full_hashes == [array_hash(a) for a in (psi, beta)], '017full frozen field/beta mutated')
                result['failed_suffix_frames'] += detail['failed_frames']
                if offset: q = np.concatenate([partial_q, q]); del partial_q, partial_summary
                require(len(q) == n, '017combined per-shot frame coverage changed')
                shot = save_q(index, q) | {'frozen_psi_betaN_sha256': full_hashes, 'origin': '016prefix+017suffix' if offset else '017suffix',
                    'suffix_records_sha256': digest(ROOT/f'frames-{index:04d}.jsonl')}
                result['shots'].append(shot); write_json(ROOT/'full-audit.json', result)
                print(f'MAST017 {index+1}/1206 shots; {result["suffix_attempted_frames"]}/22357 new frames', flush=True)
                del row, psi, oldq, beta, q
        require(result['suffix_attempted_frames'] == plan['suffix_frames'] and len(result['shots']) == plan['expected_shots']
            and sum(s['frames'] for s in result['shots']) == plan['expected_frames'], '017full unique coverage incomplete')
        identities(plan, freeze, budget.check)
        with zipfile.ZipFile(ROOT/'q95-diagnostic.npz', 'x', zipfile.ZIP_DEFLATED) as archive:
            for index, shot in enumerate(result['shots']):
                budget.check(); require(shot['index'] == index, '017export shot order changed')
                archive.write(ROOT/'q95'/f'shot_{index:04d}.npy', f'shot_{index:04d}_q95.npy')
        with np.load(ROOT/'q95-diagnostic.npz', allow_pickle=False) as export:
            require(export.files == [f'shot_{i:04d}_q95' for i in range(plan['expected_shots'])], '017final export order changed')
            for index, shot in enumerate(result['shots']):
                budget.check(); q = export[f'shot_{index:04d}_q95']
                require(q.dtype == np.float32 and q.shape == (shot['frames'],) and np.isfinite(q).all() and np.all(q>0)
                    and array_hash(q) == shot['q_sha256'], '017final export reload failed')
        result.update(complete=True, original_gates_passed=result['failed_suffix_frames']==0,
            full_frames=plan['expected_frames'], full_export_reload_exact=True,
            diagnostic_export_sha256=digest(ROOT/'q95-diagnostic.npz'), accuracy_or_promotion_established=False)
        budget.check()
    except Exception as exc:
        result.update(failed=True, failure=type(exc).__name__+': '+str(exc)); raise
    finally:
        result['actual_wall_seconds'] = time.monotonic()-budget.start
        result['suffix_unattempted_frames'] = plan['suffix_frames']-result['suffix_attempted_frames']
        write_json(ROOT/'full-audit.json', result)


if __name__ == '__main__': main()
