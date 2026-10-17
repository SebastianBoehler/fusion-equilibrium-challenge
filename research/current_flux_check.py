"""MAST018 manufactured-only elliptic correctness and falsification gates."""
import json,resource,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from participant.artifacts import digest,write_json
from research.current_flux_operator import MU0,delta_star,solve_current

ROOT=Path('run_data/mast-018')


def main():
    p=json.loads((ROOT/'predeclared-plan.json').read_text());f=json.loads((ROOT/'source-freeze.json').read_text())
    (ROOT/'running.lock').mkdir();start=time.monotonic()
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'cases':[],
        'actual_data_or_labels_envelopes_read':False,'optimizer_steps':0,'GPU_count':0,'new_candidate':False}

    def require(ok,msg):
        if not ok:raise ValueError(msg)

    def deadline():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rss=rss if sys.platform=='darwin' else rss*1024
        result['maximum_process_RSS_bytes']=rss
        require(time.monotonic()-start<=120 and rss<=2147483648,'018resource cap exceeded')

    def identities():
        require(digest(ROOT/'predeclared-plan.json')==f['plan_sha256'],'018plan changed')
        for name,h in (p['source_sha256']|f['source_sha256']).items():require(digest(Path(name))==h,'018source changed:'+name)

    try:
        identities();errors=[]
        for size in p['grid_sizes']:
            r=np.linspace(*p['R_m'],size);z=np.linspace(*p['Z_m'],size);rr,zz=np.meshgrid(r,z)
            cases=[('vacuum',.1*rr**2,np.zeros((size-2,size-2))),
                ('uniform',-MU0*1e6/3*rr**3,np.full((size-2,size-2),1e6)),
                ('cross',.1*rr**2*zz**2,np.broadcast_to(-.2*r[None,1:-1]/MU0,(size-2,size-2)).copy())]
            for name,exact,jtor in cases:
                solved,residual=solve_current(jtor,r,z,exact,deadline)
                error=float(np.max(abs(solved-exact))/np.max(abs(exact)))
                result['cases'].append({'name':name,'size':size,'relative_flux_error':error,'relative_system_residual':residual})
                write_json(ROOT/'operator-check.json',result)
                require(residual<=p['gates']['relative_linear_system_residual'],'018elliptic residual failed')
                if name=='uniform':
                    errors.append(error)
                    if size>=65:require(error<=1e-3,'018uniform current flux error failed')
                else:require(error<=1e-10,'018exact vacuum/cross control failed')
            # Reuse the last manufactured problem for explicit sign and scale invariance.
            opposite,_=solve_current(-jtor,r,z,-exact,deadline)
            scaled,_=solve_current(3*jtor,r,z,3*exact,deadline)
            require(np.max(abs(opposite+solved))/np.max(abs(solved))<=1e-10
                and np.max(abs(scaled-3*solved))/np.max(abs(3*solved))<=1e-10,'018sign/scale failed')
        require(all(b/a<=.35 for a,b in zip(errors[:-1],errors[1:])),'018grid refinement failed')
        bad=exact.copy();bad[0,0]=np.nan
        invalid=[(jtor[:,:-1],r,z,exact),(jtor,r[::-1],z,exact),(jtor,r*0,z,exact),
            (jtor,r,z,bad),(jtor.astype(np.float32),r,z,exact),(jtor,r,z,exact.astype(complex))]
        for args in invalid:
            try:solve_current(*args,deadline)
            except ValueError:pass
            else:raise AssertionError('018invalid control accepted')
        identities();deadline()
        result.update(complete=True,original_gates_passed=True,uniform_errors=errors,
            refinement_ratios=[b/a for a,b in zip(errors[:-1],errors[1:])],invalid_rejected=len(invalid),
            real_input_or_pressure_closure_or_accuracy_validated=False)
    except Exception as exc:
        result.update(failed=True,failure=type(exc).__name__+': '+str(exc));raise
    finally:
        result['actual_wall_seconds']=time.monotonic()-start;write_json(ROOT/'operator-check.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
