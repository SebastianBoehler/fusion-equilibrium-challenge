"""Exclusive full MAST016 audit/export. No competition candidate build."""
from datetime import datetime, timezone
import json
import os
import zipfile
import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_green import coil_kernel
from research.mast_hybrid_q95_controls import controls
from research.mast_full_hybrid_runtime import (ROOT, require, identities, inputs,
    geometry_hash, unit_evidence, Budget, array_hash)
from research.mast_full_hybrid_shot import audit


def main():
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    (ROOT/'running.lock').mkdir()
    (ROOT/'geometry-evidence').mkdir()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'process_PID': os.getpid(), 'shots': [], 'attempted_frames': 0,
              'failed_frames': 0, 'MAST_labels_demos_envelopes_read': False,
              'optimizer_steps': 0, 'GPU_count': 0, 'cluster_allocations': 0,
              'new_total_flux_betaN_or_competition_archive': False}
    budget = Budget(plan, result); output = ROOT/'full-audit.json'
    try:
        identities(plan, freeze, budget.check)
        manifest = json.loads(open(plan['manifest_path']).read())
        order = json.loads(open(plan['order_path']).read())
        require(manifest['shots'] == plan['entries'] and len(plan['entries']) == plan['expected_shots']
            and manifest['revision'] == plan['dataset_revision']
            and order['ordered_paths'] == [e['path'] for e in plan['entries']], '016full order changed')
        result['manufactured_controls'] = controls(plan)
        last_geometry = None; kernels = None; units = None
        with np.load(plan['predictions_path'], allow_pickle=False) as stored, \
                zipfile.ZipFile(ROOT/'q95-diagnostic.npz', 'x', zipfile.ZIP_DEFLATED) as export:
            for index, entry in enumerate(plan['entries']):
                budget.check(); row = inputs(entry, plan); identity = geometry_hash(row)
                if identity != last_geometry:
                    kernels = coil_kernel(row); units = unit_evidence(row, identity, budget.check)
                    last_geometry = identity
                    write_json(ROOT/'geometry-evidence'/f'{identity}.json', units)
                prefix = f'shot_{index:04d}_'
                psi, oldq, beta = (stored[prefix+k] for k in ('psirz', 'q95', 'betaN'))
                shot = {'index': index, 'shot': entry['path'], 'geometry_sha256': identity,
                        'completed': False, 'records_path': f'frames-{index:04d}.jsonl'}
                result['shots'].append(shot)
                with (ROOT/shot['records_path']).open('x') as records:
                    def emit(value):
                        records.write(json.dumps(value)+'\n'); records.flush()
                        result['attempted_frames'] += 1
                        write_json(ROOT/'progress.json', {'shot_index': index, 'frame_index': value['index'],
                            'attempted_frames': result['attempted_frames'], 'actual_wall_seconds': result['actual_wall_seconds'],
                            'process_PID': os.getpid(), 'observed_at': datetime.now(timezone.utc).isoformat()})
                    q, details = audit(row, psi, oldq, beta, kernels, units, plan, budget.check, emit)
                with export.open(prefix+'q95.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, q, allow_pickle=False)
                shot.update(details, completed=True, q_export_sha256=array_hash(q),
                    records_sha256=digest(ROOT/shot['records_path']))
                result['failed_frames'] += details['failed_frames']
                write_json(output, result)
                print(f'MAST016 {index+1}/1206 shots; {result["attempted_frames"]}/69075 frames', flush=True)
                # Release the complete shot before reading the next one.
                del row, psi, oldq, beta, q
        budget.check(); identities(plan, freeze, budget.check)
        require(result['attempted_frames'] == plan['expected_frames'], '016full frame count changed')
        with np.load(ROOT/'q95-diagnostic.npz', allow_pickle=False) as saved:
            require(saved.files == [f'shot_{i:04d}_q95' for i in range(plan['expected_shots'])], '016export order changed')
            for i, shot in enumerate(result['shots']):
                budget.check(); q = saved[f'shot_{i:04d}_q95']
                require(q.dtype == np.float32 and q.shape == (shot['frames'],)
                    and np.isfinite(q).all() and np.all(q>0)
                    and array_hash(q) == shot['q_export_sha256'], '016export reload changed')
                del q
        budget.check(); result.update(complete=True, original_gates_passed=result['failed_frames']==0,
            sources_inputs_predictions_unchanged=True, export_reload_exact=True,
            diagnostic_export_sha256=digest(ROOT/'q95-diagnostic.npz'),
            official_accuracy_or_candidate_validated=False)
        budget.check()
    except Exception as exc:
        result.update(failed=True, failure=type(exc).__name__+': '+str(exc)); raise
    finally:
        result['actual_wall_seconds'] = __import__('time').monotonic()-budget.start
        result['unattempted_frames'] = plan['expected_frames']-result['attempted_frames']
        write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='shots'}, indent=2), flush=True)


if __name__ == '__main__': main()
