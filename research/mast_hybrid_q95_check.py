"""Fixed first20/all1122 MAST015 diagnostic; original candidate stays frozen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

from participant.artifacts import digest, write_json
from participant.mast_green import MU0, TF_TURNS, coil_kernel, pressure_profiles
from participant.mast_inputs import COLUMNS, SIGNALS, aligned_currents
from participant.mast_surface import flux_surface, safety_factor
from research.mast_hybrid_q95 import gradient, integrate, external_grid, external_gradient
from research.mast_hybrid_q95_controls import controls

ROOT = Path('run_data/mast-015')


def require(value, message):
    if not value: raise ValueError(message)


def identities(plan, freeze):
    require(digest(ROOT/'predeclared-plan.json') == freeze['plan_sha256'], '015plan changed')
    pins = plan['source_sha256'] | freeze['source_sha256'] | plan['input_cache_sha256']
    pins.update({plan[key+'_path']: plan[key+'_sha256'] for key in ('predictions', 'manifest', 'order')})
    for name, expected in pins.items():
        require(digest(Path(name)) == expected, '015source/input changed: '+name)


def main():
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    (ROOT/'running.lock').mkdir(); start = time.monotonic()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'shots': [], 'attempted_frames': 0, 'MAST_labels_demos_envelopes_read': False,
              'new_total_flux_or_betaN': False, 'optimizer_steps': 0, 'GPU_count': 0,
              'candidate_archive_upload': False}

    def deadline():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss*1024
        result['maximum_process_RSS_bytes'] = rss
        require(time.monotonic()-start <= plan['resources']['hard_local_seconds']
                and rss <= plan['resources']['max_process_RSS_bytes'], '015resource cap exceeded')

    try:
        identities(plan, freeze); deadline(); result['manufactured'] = controls(plan)
        manifest = json.loads(Path(plan['manifest_path']).read_text())
        order = json.loads(Path(plan['order_path']).read_text())
        require(manifest['shots'][:20] == plan['entries']
            and order['ordered_paths'] == [e['path'] for e in manifest['shots']], '015order changed')
        previous = json.loads(Path('run_data/mast-014/replacement-check.json').read_text())
        require(previous['complete'] and previous['fixed_linear_closure_passed'], '014closure incomplete')
        kernels_cache = {}
        with np.load(plan['predictions_path'], allow_pickle=False) as stored:
            for index, entry in enumerate(plan['entries']):
                name = Path(entry['path']).stem
                with np.load(Path('run_data/mast-001/inputs')/(name+'.npz'), allow_pickle=False) as data:
                    require(set(data.files) == set(COLUMNS)|{'revision', 'source_hash'}
                        and str(data['revision']) == plan['dataset_revision']
                        and str(data['source_hash']) == entry['sha256'], '015allowlist/provenance changed')
                    row = {key: data[key] for key in data.files}
                for signal in SIGNALS:
                    axis = row['magnetics_time']; value = row['magnetics_'+signal]
                    axis = axis[np.isfinite(axis)&np.isfinite(value)]
                    require(len(axis) >= 2 and np.all(np.diff(axis)>0)
                        and np.all(row['efit_times'] >= axis[0])
                        and np.all(row['efit_times'] <= axis[-1]), '015current bracketing failed')
                currents = aligned_currents(row)
                radius, pressure = pressure_profiles(row)
                ordering = np.argsort(radius); radius, pressure = radius[ordering], pressure[:, ordering]
                weighted = np.where(np.isfinite(pressure), pressure, 0)*np.gradient(radius)
                center = (weighted*radius).sum(1)/weighted.sum(1)
                r, z = row['efit_grid_R'], row['efit_grid_Z']
                conductors = np.column_stack([np.interp(row['coil_Z'], z, np.arange(len(z))),
                                              np.interp(row['coil_R'], r, np.arange(len(r)))])
                prefix = f'shot_{index:04d}_'; psi = stored[prefix+'psirz']
                oldq = stored[prefix+'q95']; beta = stored[prefix+'betaN']
                n = len(row['efit_times'])
                require(psi.shape == (n, 65, 65) and psi.dtype == np.float32
                    and oldq.shape == beta.shape == (n,) and oldq.dtype == beta.dtype == np.float32
                    and all(np.isfinite(a).all() for a in (psi, oldq, beta))
                    and previous['shots'][index]['shot'] == name, '015stored schema/binding changed')
                geometry = hashlib.sha256()
                for key in ('coil_input_column', 'coil_R', 'coil_Z', 'coil_width', 'coil_height', 'efit_grid_R', 'efit_grid_Z'):
                    a = row[key]; geometry.update(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes())
                key = geometry.hexdigest()
                if key not in kernels_cache: kernels_cache[key] = coil_kernel(row)
                fvac = abs(currents[:, SIGNALS.index('tf_current')])*1000*MU0*TF_TURNS/(2*np.pi)
                shot = {'shot': name, 'frames': n, 'records': [], 'psi_sha256': hashlib.sha256(psi.tobytes()).hexdigest(),
                        'betaN_sha256': hashlib.sha256(beta.tobytes()).hexdigest(), 'geometry_sha256': key}
                result['shots'].append(shot)
                for frame in range(n):
                    deadline(); contour = flux_surface(psi[frame], r, z, center[frame], conductors)
                    mid = (contour[1:]+contour[:-1])/2
                    nativeq = safety_factor(psi[frame], r, z, contour, fvac[frame])
                    external = external_grid(row, currents[frame], kernels_cache[key])
                    plasma = gradient(psi[frame].astype(float)-external, r, z, mid)
                    native = gradient(psi[frame], r, z, mid)
                    reconstructed = plasma+gradient(external, r, z, mid)
                    residual = float(np.max(abs(reconstructed-native)))
                    scale = float(np.max(abs(native)))
                    direct = external_gradient(row, currents[frame], mid, plan['radial_h_m'], deadline)
                    qs = np.array([integrate(contour, plasma+v, fvac[frame]) for v in direct])
                    paired = float(abs(qs[0]/qs[1]-1))
                    native_exact = bool(np.float32(nativeq).tobytes() == oldq[frame].tobytes())
                    reconstruction_passed = residual <= 1e-10+1e-10*scale
                    shot['records'].append({'index': frame, 'time_ms': float(row['efit_times'][frame]),
                        'contour_sha256': hashlib.sha256(contour.tobytes()).hexdigest(), 'segments': len(mid),
                        'stored_q95': float(oldq[frame]), 'native_q95': nativeq, 'native_bit_equal': native_exact,
                        'spline_reconstruction_max_error': residual, 'reconstruction_passed': reconstruction_passed,
                        'hybrid_h_h2_q95': qs.tolist(), 'paired_relative': paired,
                        'paired_passed': paired <= plan['gates']['direct_h_h2_q_relative']})
                    result['attempted_frames'] += 1
                    require(native_exact and reconstruction_passed, '015native/reconstruction control failed')
                write_json(ROOT/'q95-check.json', result)
                print(f'MAST015 {index+1}/20 shots; {result["attempted_frames"]}/1122 frames; {time.monotonic()-start:.1f}s', flush=True)
        identities(plan, freeze); deadline()
        records = [v for s in result['shots'] for v in s['records']]
        require(len(records) == plan['expected_frames'], '015original frame count changed')
        failed = [dict(shot=s['shot'], index=v['index']) for s in result['shots'] for v in s['records'] if not v['paired_passed']]
        ratios = [v['hybrid_h_h2_q95'][1]/v['stored_q95'] for v in records]
        result.update(complete=True, original_gates_passed=not failed, paired_failed_frames=failed,
            maximum_paired_relative=max(v['paired_relative'] for v in records),
            hybrid_to_stored_ratio_percentiles=np.percentile(ratios, [0, 25, 50, 75, 100]).tolist(),
            sources_inputs_predictions_unchanged=True, full_corpus_or_accuracy_validated=False)
    except Exception as exc:
        result.update(failed=True, failure=type(exc).__name__+': '+str(exc)); raise
    finally:
        result['actual_wall_seconds'] = time.monotonic()-start
        result['unattempted_frames'] = plan['expected_frames']-result['attempted_frames']
        write_json(ROOT/'q95-check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'shots'}, indent=2), flush=True)


if __name__ == '__main__': main()
