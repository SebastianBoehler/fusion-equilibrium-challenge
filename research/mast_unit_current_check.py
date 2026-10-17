"""Geometry-only causal013 check, no measured currents or pressure."""
from datetime import datetime,timezone
import hashlib,json,time,resource,sys
from pathlib import Path
import numpy as np
from participant.artifacts import digest,write_json
from participant.mast_inputs import COLUMNS
from research.mast_unit_current import unit_group,direct

ROOT=Path('run_data/mast-013')
KEYS=['coil_input_column','coil_R','coil_Z','coil_width','coil_height','efit_grid_R','efit_grid_Z']


def identities(plan,freeze):
    if digest(ROOT/'predeclared-plan.json')!=freeze['plan_sha256']:raise ValueError('013plan changed')
    for name,expected in (plan['source_sha256']|freeze['source_sha256']|plan['input_cache_sha256']).items():
        if digest(Path(name))!=expected:raise ValueError('013 source/cache changed: '+name)


def main():
    p=json.loads((ROOT/'predeclared-plan.json').read_text());f=json.loads((ROOT/'source-freeze.json').read_text())
    identities(p,f);(ROOT/'running.lock').mkdir();started=time.monotonic()
    def deadline():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rss=rss if sys.platform=='darwin' else rss*1024
        if time.monotonic()-started>300 or rss>2147483648:raise ValueError('Original013 resource cap exceeded')
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'shots':[],
        'measured_currents_pressure_or_predictions_read':False,'MAST_labels_envelopes_read':False,
        'optimizer_steps':0,'GPU_count':0,'candidate_or_upload':False}
    try:
        r=np.linspace(.5,1.5,65);z=np.linspace(-.5,.5,65)
        manufactured={}
        for center,expected in [(1.,1.),(1.8,0.)]:
            response=direct(np.array([center]),np.array([0.]),np.zeros(1),np.zeros(1),r,z,32,1e-5,deadline)
            if np.max(abs(response-expected))>1e-4:raise ValueError('Direct unit-ring independent Ampere oracle failed')
            manufactured[str(center)]=response.tolist()
        result['manufactured_ring_oracles']=manufactured;cache={}
        for entry in p['entries']:
            path=Path('run_data/mast-001/inputs')/(Path(entry['path']).stem+'.npz')
            with np.load(path,allow_pickle=False) as d:
                if set(d.files)!=set(COLUMNS)|{'revision','source_hash'} or str(d['revision'])!=p['dataset_revision'] or str(d['source_hash'])!=entry['sha256']:
                    raise ValueError('013allowedcacheprovenance changed')
                row={key:d[key] for key in KEYS}
            identity=hashlib.sha256()
            for key in KEYS:
                a=row[key];identity.update(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes())
            key=identity.hexdigest()
            if key not in cache:
                names=row['coil_input_column'].astype(str);groups={}
                for name in np.unique(names):
                    mask=names==name
                    args=[row[k][mask] for k in ['coil_R','coil_Z','coil_width','coil_height']]
                    normalization=1 if name=='magnetics_sol_current' else int(mask.sum())
                    groups[name]=unit_group(*args,row['efit_grid_R'],row['efit_grid_Z'],normalization,p,deadline)
                cache[key]=groups
            result['shots'].append({'shot':Path(entry['path']).stem,'exact_geometry_sha256':key,'groups':cache[key]})
            deadline();write_json(ROOT/'unit-current-check.json',result)
        identities(p,f);groups=[g for v in cache.values() for g in v.values()]
        result.update(complete=True,shots_checked=20,unique_geometries=len(cache),unique_coil_groups=len(groups),
            direct_exact_and_paired_gates_passed=all(g['direct_exact_passed'] and g['paired_passed'] for g in groups),
            maximum_direct_absolute_current_error_A=max(g['direct_max_error_A'] for g in groups),
            maximum_grid_absolute_unit_current_error_A=max(abs(g['grid_error_A']) for g in groups),
            sources_inputs_unchanged=True,model_replacement_ready=False)
    except Exception as e:
        result.update(failed=True,failure=type(e).__name__+': '+str(e));raise
    finally:
        result['actual_wall_seconds']=time.monotonic()-started;write_json(ROOT/'unit-current-check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='shots'},indent=2))


if __name__=='__main__':main()
