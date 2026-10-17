"""Fixed first20 stored MAST flux/input diagnostics, no new candidate."""
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from participant.artifacts import digest,write_json
from participant.mast_green import MU0
from participant.mast_inputs import COLUMNS
from research.mast_vertical_force import baseline_current
from research.mast_ampere_closure import rectangle_current,expected_currents

ROOT=Path('run_data/mast-012')


def require(value,message):
    if not value:raise ValueError(message)


def identities(plan,freeze):
    require(digest(ROOT/'predeclared-plan.json')==freeze['plan_sha256'],'012plan changed')
    pins=plan['source_sha256']|freeze['source_sha256']|plan['input_cache_sha256']
    pins.update({plan['predictions_path']:plan['predictions_sha256'],
                 plan['manifest_path']:plan['manifest_sha256'],plan['order_path']:plan['order_sha256']})
    for name,expected in pins.items():require(digest(Path(name))==expected,f'012source/input changed: {name}')


def controls(plan):
    m=plan['manufactured'];r=np.linspace(*m['R'],m['grid_points']);z=np.linspace(*m['Z'],m['grid_points'])
    rr,zz=np.meshgrid(r,z);c=m['c'];r0=m['R0'];psi=-c*((rr-r0)**2+zz**2)
    exact=4*c*z[-1]/MU0*(np.log(r[-1]/r[0])+r0*(1/r[0]-1/r[-1]))
    result={}
    for order in plan['quadrature_orders']:
        actual=rectangle_current(psi,r,z,order)
        require(abs(actual-exact)/exact<=1e-6,'Manufactured exact current failed')
        require(abs(rectangle_current(rr**2,r,z,order))<=1e-4,'Vacuum current control failed')
        require(np.isclose(rectangle_current(-psi,r,z,order),-actual,rtol=1e-12,atol=1e-6)
            and np.isclose(rectangle_current(3*psi,r,z,order),3*actual,rtol=1e-12,atol=1e-6),'Orientation/linearity controls failed')
        result[str(order)]={'exact_A':exact,'actual_A':actual,'relative_error':abs(actual-exact)/exact}
    bad=psi.copy();bad[0,0]=np.nan
    for args in [(psi[:,:-1],r,z,4),(bad,r,z,4),(psi,r[::-1],z,4),(psi,r*0,z,4),(psi,r,z,3)]:
        try:rectangle_current(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid Ampere control accepted')
    return result


def main():
    plan=json.loads((ROOT/'predeclared-plan.json').read_text());freeze=json.loads((ROOT/'source-freeze.json').read_text())
    started=time.monotonic();identities(plan,freeze);(ROOT/'running.lock').mkdir()
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'shots':[],
        'new_field_inference':False,'MAST_labels_demos_envelopes_read':False,'optimizer_steps':0,
        'GPU_count':0,'new_archive_or_upload':False,'original_pressure_endpoint_holding_unchanged':True}
    output=ROOT/'closure-check.json'
    try:
        result['manufactured_controls']=controls(plan)
        manifest=json.loads(Path(plan['manifest_path']).read_text());order=json.loads(Path(plan['order_path']).read_text())
        require(manifest['shots'][:20]==plan['entries'] and order['ordered_paths']==[e['path'] for e in manifest['shots']],
                'Original selected prediction/input order changed')
        count=0;maxrss=0
        with np.load(plan['predictions_path'],allow_pickle=False) as stored:
            for index,entry in enumerate(plan['entries']):
                path=Path('run_data/mast-001/inputs')/(Path(entry['path']).stem+'.npz')
                with np.load(path,allow_pickle=False) as data:
                    require(set(data.files)==set(COLUMNS)|{'revision','source_hash'}
                        and str(data['revision'])==plan['dataset_revision'] and str(data['source_hash'])==entry['sha256'],
                        'MAST cache provenance/allowlist changed')
                    row={key:data[key] for key in data.files}
                currents,weights,sr,sz,_,_=baseline_current(row)
                psi=stored[f'shot_{index:04d}_psirz'];n=len(row['efit_times']);count+=n
                require(psi.shape==(n,65,65) and psi.dtype==np.float32 and np.isfinite(psi).all(),
                        'Original stored flux schema changed')
                expected,scale,components=expected_currents(row,currents,weights)
                actual=np.array([[rectangle_current(field,row['efit_grid_R'],row['efit_grid_Z'],k)
                                   for k in plan['quadrature_orders']] for field in psi])
                paired=np.abs(actual[:,1]-actual[:,2]);converged=(paired/scale<=1e-4)|(paired<=.001)
                residual=(actual[:,2]-expected)/scale
                result['shots'].append({'shot':Path(entry['path']).stem,'frames':n,'expected_A':expected.tolist(),
                    'scale_A':scale.tolist(),'components_A':{key:value.tolist() for key,value in components.items()},
                    'actual_orders_A':actual.tolist(),'paired4vs8_A':paired.tolist(),
                    'convergence_passed_per_frame':converged.tolist(),'normalized_current_residual':residual.tolist(),
                    'closure_failed_indices':np.flatnonzero(np.abs(residual)>.01).tolist()})
                write_json(output,result)
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;maxrss=max(maxrss,rss if sys.platform=='darwin' else rss*1024)
                require(time.monotonic()-started<=120 and maxrss<=2147483648,'012original resource limit exceeded')
        identities(plan,freeze);require(count==1122,'Original20shot frame count changed')
        convergence=all(all(s['convergence_passed_per_frame']) for s in result['shots'])
        failed=sum(len(s['closure_failed_indices']) for s in result['shots'])
        result.update(complete=True,frames=count,numerical_convergence_passed=convergence,
            closure_failed_frames=failed,current_closure_passed=convergence and failed==0,
            maximum_process_RSS_bytes=maxrss,sources_inputs_unchanged=True,model_replacement_ready=False)
    except Exception as exc:
        result.update(failed=True,failure=type(exc).__name__+': '+str(exc));raise
    finally:
        result['actual_wall_seconds']=time.monotonic()-started;write_json(output,result)
    print(json.dumps({key:value for key,value in result.items() if key!='shots'},indent=2))


if __name__=='__main__':main()
