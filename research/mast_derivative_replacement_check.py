"""Fixed linear source-derivative replacement, diagnostic currents only."""
from datetime import datetime,timezone
import json,time,resource,sys,hashlib
from pathlib import Path
import numpy as np
from participant.artifacts import digest,write_json
from participant.mast_inputs import COLUMNS,SIGNALS,aligned_currents
ROOT=Path('run_data/mast-014')
GEOMETRY=['coil_input_column','coil_R','coil_Z','coil_width','coil_height','efit_grid_R','efit_grid_Z']


def identities(plan,freeze):
    if digest(ROOT/'predeclared-plan.json')!=freeze['plan_sha256']:raise ValueError('014plan changed')
    for name,expected in (plan['source_sha256']|freeze['source_sha256']|plan['input_cache_sha256']).items():
        if digest(Path(name))!=expected:raise ValueError('014source/cache changed:'+name)


def main():
    p=json.loads((ROOT/'predeclared-plan.json').read_text());f=json.loads((ROOT/'source-freeze.json').read_text())
    identities(p,f);(ROOT/'running.lock').mkdir();start=time.monotonic()
    old=json.loads(Path('run_data/mast-012/closure-check.json').read_text())
    unit=json.loads(Path('run_data/mast-013/unit-current-check.json').read_text())
    if not old['complete'] or not unit['direct_exact_and_paired_gates_passed']:raise ValueError('Actual causal evidence incomplete')
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'shots':[],
            'MAST_labels_envelopes_or_pressure_read':False,'new_predictions_parameters_or_archive':False,
            'optimizer_steps':0,'GPU_count':0}
    try:
        for index,entry in enumerate(p['entries']):
            name=Path(entry['path']).stem;path=Path('run_data/mast-001/inputs')/(name+'.npz')
            with np.load(path,allow_pickle=False) as d:
                if set(d.files)!=set(COLUMNS)|{'source_hash','revision'} or str(d['source_hash'])!=entry['sha256'] or str(d['revision'])!=p['dataset_revision']:
                    raise ValueError('014allowlist/provenance changed')
                row={k:d[k] for k in ['efit_times','magnetics_time',*GEOMETRY,*('magnetics_'+s for s in SIGNALS)]}
            identity=hashlib.sha256()
            for key in GEOMETRY:
                a=row[key];identity.update(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes())
            if identity.hexdigest()!=unit['shots'][index]['exact_geometry_sha256'] or name!=old['shots'][index]['shot']:
                raise ValueError('014input/geometry/order binding changed')
            for signal in SIGNALS:
                times=row['magnetics_time'];values=row['magnetics_'+signal]
                axis=times[np.isfinite(times)&np.isfinite(values)]
                if len(axis)<2 or np.any(np.diff(axis)<=0) or np.any(row['efit_times']<axis[0]) or np.any(row['efit_times']>axis[-1]):
                    raise ValueError('Original current signal bracketing support failed')
            currents=aligned_currents(row);correction=np.zeros(len(currents));terms={}
            for group,g in unit['shots'][index]['groups'].items():
                coefficient=g['direct_orders_h_h2_A'][-1][-1]-g['grid_unit_A']
                terms[group]=currents[:,SIGNALS.index(group.removeprefix('magnetics_'))]*1000*coefficient
                correction+=terms[group]
            previous=old['shots'][index];actual=np.asarray(previous['actual_orders_A'])[:,-1]
            expected=np.asarray(previous['expected_A']);scale=np.asarray(previous['scale_A'])
            corrected=actual+correction;residual=(corrected-expected)/scale
            result['shots'].append({'shot':name,'frames':len(actual),'original_normalized_residual':previous['normalized_current_residual'],
                'fixed_coil_derivative_correction_A':correction.tolist(),'corrected_current_A':corrected.tolist(),
                'corrected_normalized_residual':residual.tolist(),'failed_indices':np.flatnonzero(abs(residual)>.01).tolist()})
            write_json(ROOT/'replacement-check.json',result)
        identities(p,f)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rss=rss if sys.platform=='darwin' else rss*1024
        if time.monotonic()-start>120 or rss>2147483648:raise ValueError('Original014resource cap exceeded')
        all_res=np.concatenate([s['corrected_normalized_residual'] for s in result['shots']])
        count=sum(s['frames'] for s in result['shots']);failed=sum(len(s['failed_indices']) for s in result['shots'])
        if count!=1122:raise ValueError('Original fixedframe count changed')
        result.update(complete=True,frames=count,failed_frames=failed,fixed_linear_closure_passed=failed==0,
            median_absolute_corrected_residual=float(np.median(abs(all_res))),maximum_absolute_corrected_residual=float(np.max(abs(all_res))),
            maximum_process_RSS_bytes=rss,sources_inputs_unchanged=True,model_replacement_ready=False)
    except Exception as e:
        result.update(failed=True,failure=type(e).__name__+': '+str(e));raise
    finally:
        result['actual_wall_seconds']=time.monotonic()-start;write_json(ROOT/'replacement-check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='shots'},indent=2))


if __name__=='__main__':main()
