"""All original usable TRAIN inputs: two global scans and native RMS replay."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from participant.artifacts import digest,write_json
from participant import correction_constraint as constraint
from participant.critical_artifact import restore_correction
from participant.data import read_shot
from participant.finite_parent_cache import cache_parent
from participant.finite_target_runtime import require_runtime
from participant.rich import INPUT_COLUMNS,rich_features
from research.full_constraint_resources import Memory,allocation


def require(value,message):
    if not value:raise ValueError(message)


def source_gates(root,plan):
    freeze=json.loads((root/'source-freeze.json').read_text())
    checkout=Path(__file__).resolve().parents[1]
    require(digest(root/'predeclared-plan.json')==freeze['plan_sha256'],'059 plan changed')
    for name,expected in freeze['source_sha256'].items():
        require(digest(checkout/name)==expected,f'Frozen059 source changed: {name}')
    require_runtime(checkout,plan['source_pin'])
    require(digest(root/'full-split.json')==plan['manifest_sha256'],'Full manifest changed')
    for name in ('parent','fixture'):
        require(digest(Path(plan['paths'][name]))==plan[name+'_sha256'],f'Original{name} changed')
    target=Path(plan['paths']['targets'])
    for name,bounds in plan['target_file_sha256'].items():
        require(digest(target/Path(name).name)==bounds['sha256'],'Original completed target bytes changed')
    require(digest(target/'index.json')==plan['target_index_sha256'],'Full target index changed')


def rows(plan,manifest,index,valid):
    for i,entry in enumerate(manifest['train']):
        row=read_shot(entry,manifest['revision'],Path(plan['paths']['raw']),INPUT_COLUMNS)
        raw=rich_features(row)
        a,b=index['offsets'][i:i+2]
        require(len(raw)==b-a,'Full input frame count/order differs')
        yield raw,valid[a:b]


def run(root):
    plan=json.loads((root/'predeclared-plan.json').read_text())
    (root/'running.lock').mkdir()
    started=time.monotonic();deadline=started+plan['resources']['hard_seconds']
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),
        'optimizer_steps':0,'GPU_count':0,'label_columns_read':False,'learned_or_prediction_files_saved':False,
        'plan_sha256':digest(root/'predeclared-plan.json'),'global_scans':[]}
    output=root/'full-support.json';memory=None
    try:
        job=allocation(root,plan);memory=Memory(job);memory.sample('before_inputs',deadline)
        source_gates(root,plan);torch.set_num_threads(2)
        manifest=json.loads((root/'full-split.json').read_text())
        target=Path(plan['paths']['targets']);index=json.loads((target/'index.json').read_text())
        valid=np.load(target/'valid.npy',allow_pickle=False)
        require(len(manifest['train'])==6641 and index['names']==[Path(e['path']).stem for e in manifest['train']]
            and index['frames']==1469502 and index['offsets'][0]==0 and index['offsets'][-1]==1469502
            and len(index['offsets'])==6642 and np.all(np.diff(index['offsets'])>0)
            and valid.shape==(1469502,) and valid.dtype==np.bool_ and int(valid.sum())==1469328,
            'Original full support/order/exclusions differ')
        parent=torch.load(plan['paths']['parent'],map_location='cpu',weights_only=False)
        fixture=torch.load(plan['paths']['fixture'],map_location='cpu',weights_only=False)
        model=restore_correction(parent,fixture)
        initial_parent={k:v.clone() for k,v in model.parent.state_dict().items()}
        initial_statistics={k:getattr(model,k).clone() for k in
            ('x_mean','x_std','psi_mean','coef_scale','basis','s_mean','s_std','correction_scale')}
        features=torch.empty((1469328,model.correction.in_features),dtype=torch.float32)
        std=torch.empty(1469328,dtype=torch.float32)
        cursor=0;tick=time.monotonic()
        for shot,(raw,mask) in enumerate(rows(plan,manifest,index,valid)):
            if mask.any():
                cache=cache_parent(model,raw,mask);n=len(cache.indices)
                features[cursor:cursor+n].copy_(cache.features);std[cursor:cursor+n].copy_(cache.parent_std)
                cursor+=n
                del cache
            memory.sample(f'cache_shot{shot}',deadline)
            if shot%100==0:result.update(cached_frames=cursor);write_json(output,result)
        require(cursor==1469328,'Full eligible input cache incomplete')
        result['full_cache_seconds']=time.monotonic()-tick
        gram=constraint.interpolation_gram(model)
        original_scan=constraint.maximum_relative_rms
        def recorded_scan(*args):
            memory.sample('before_full_scan',deadline);tick=time.monotonic()
            maximum=original_scan(*args)
            result['global_scans'].append({'frames':len(args[1]),'maximum_relative_RMS':maximum,
                'actual_seconds':time.monotonic()-tick})
            memory.sample('after_full_scan',deadline);write_json(output,result)
            return maximum
        constraint.maximum_relative_rms=recorded_scan
        try:
            projection=constraint.project_correction(model,features,std,gram,.001)
        finally:
            constraint.maximum_relative_rms=original_scan
        require(len(result['global_scans'])==2 and all(s['frames']==1469328 for s in result['global_scans']),
            'Both original full support scans required')
        cursor=0;maximum_native=0.;maximum_discrepancy=0.;tick=time.monotonic()
        with torch.no_grad():
            for shot,(raw,mask) in enumerate(rows(plan,manifest,index,valid)):
                if mask.any():
                    cache=cache_parent(model,raw,mask);n=len(cache.indices)
                    require(torch.equal(cache.features,features[cursor:cursor+n])
                        and torch.equal(cache.parent_std,std[cursor:cursor+n]),'Permitted input replay changed')
                    for start in range(0,n,256):
                        latent=model.correction(cache.features[start:start+256])
                        delta=F.interpolate(latent.reshape(-1,1,17,17),size=(65,65),
                            mode='bilinear',align_corners=True).flatten(1)*model.correction_scale
                        base=cache.fields[start:start+256];scale=cache.parent_std[start:start+256]
                        native=((base+delta)-base).double().square().mean(1).sqrt()/scale
                        analytic=constraint.correction_rms(latent,gram)/scale
                        maximum_native=max(maximum_native,float(native.max()))
                        maximum_discrepancy=max(maximum_discrepancy,float((native-analytic).abs().max()))
                    cursor+=n;del cache
                memory.sample(f'replay_shot{shot}',deadline)
                if shot%100==0:result.update(replayed_frames=cursor);write_json(output,result)
        require(cursor==1469328 and maximum_native<=.001 and maximum_discrepancy<=1e-6,
            'Original full actual-native RMS gates failed')
        require(all(torch.equal(model.parent.state_dict()[key],value) for key,value in initial_parent.items())
            and all(torch.equal(getattr(model,key),value) for key,value in initial_statistics.items())
            and all(p.grad is None for p in model.parameters()),'Frozen parent/statistics or zero-optimizer gate failed')
        source_gates(root,plan);memory.sample('completed',deadline)
        result.update(complete=True,all_predeclared_checks_passed=True,usable_frames=1469328,
            original_excluded_frames=174,projection=projection,maximum_actual_native_relative_RMS=maximum_native,
            maximum_Gram_native_discrepancy=maximum_discrepancy,native_replay_seconds=time.monotonic()-tick,
            full_RMS_constraint_support_verified=True,all_frame_contour_support_verified=False,
            CUDA_verified=False,production_ready=False)
    except Exception as exc:
        result.update(failed=True,failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(actual_body_seconds=time.monotonic()-started)
        if memory is not None:write_json(root/'aggregate-memory.json',{'samples':memory.samples,'scope':'Enclosing actual Slurm job including pagecache; zero workers'})
        write_json(output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
