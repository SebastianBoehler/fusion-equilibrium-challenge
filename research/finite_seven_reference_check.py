"""Fixed053 eight-TRAIN-frame actual seven-loss reference; zero optimizer."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import sys
from importlib import import_module
from importlib.metadata import version

import numpy as np
import torch

from participant.artifacts import digest,write_json
from participant.critical_artifact import restore_correction
from participant.data import read_shot
from participant.fast_score import official
from participant.finite_parent_cache import cache_parent
from participant.finite_target_runtime import RuntimePinnedTargetBatch,require_runtime
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.rich import INPUT_COLUMNS,rich_features
from participant.smooth_correction import SmoothCorrection
from research.finite_seven_reference import check_model,require
from research.finite_seven_rejections import reject,helper_rejections

ROOT = Path('run_data/analysis-053')


def identities(plan,freeze):
    require(digest(ROOT/'execution-plan.json')==freeze['plan_sha256'],'Execution plan changed')
    pins = plan['source_sha256']|freeze['source_sha256']|plan['target_files_sha256']
    for key,name in plan['existing_fixture_paths'].items():
        pins[name]=plan['historical_fixture_sha256'][key]
    for name,expected in pins.items():
        require(digest(Path(name))==expected,f'Pinned source/input changed: {name}')
    require_runtime(Path.cwd(),plan['original_source_pin'])
    actual_runtime={'python':sys.version.split()[0],**{key:version(key) for key in
        ['numpy','torch','scipy','pyarrow','huggingface-hub']}}
    require(actual_runtime==freeze['runtime'],'Exact reference runtime changed')
    for name in ('finite_seven_reference','finite_seven_rejections','finite_constraint_adjoint',
                 'finite_device_bridge','finite_metric_loss'):
        module=import_module('research.'+name)
        require(Path(module.__file__).resolve()==Path('research',name+'.py').resolve(),
                'Reference module imported from another checkout')


def target_batch(plan,index,components,variance,provenance):
    directory = Path(plan['target_directory'])
    actual_index = json.loads((directory/'index.json').read_text())
    expected_variance = np.asarray(json.loads((directory/'validation.json').read_text())['statistics']['variance'])
    require(components==list(official.CONS_SCALARS),'Component order binding changed')
    require(np.array_equal(variance,expected_variance),'Full population variance binding changed')
    require(index==actual_index,'Full target frame/count/order binding changed')
    return RuntimePinnedTargetBatch(directory,index,index['manifest_sha256'],provenance,plan['original_source_pin'])


def bind_selection(batch,index,selected,expected):
    require(np.array_equal(selected,expected),'Fixed selected frame order changed')
    return batch.get(selected)


def main():
    plan = json.loads((ROOT/'execution-plan.json').read_text())
    freeze = json.loads((ROOT/'execution-freeze.json').read_text())
    identities(plan,freeze)
    (ROOT/'execution.lock').mkdir()
    started=time.monotonic();deadline=started+plan['resources']['hard_local_wall_seconds']
    result={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':freeze['plan_sha256'],'optimizer_steps':0,'GPU_count':0,'job_submissions':0,
        'SSH_attempts':0,'MAST_development_locked_labels_read':False,
        'TRAIN_raw_label_columns_read':False,'learned_files_or_predictions_saved':False,'controls':[]}
    output=ROOT/'seven-reference-check.json'
    try:
        torch.set_num_threads(2)
        directory=Path(plan['target_directory'])
        index=json.loads((directory/'index.json').read_text())
        validation=json.loads((directory/'validation.json').read_text())
        variance=np.asarray(validation['statistics']['variance'],dtype=np.float64)
        provenance=digest(directory/'runtime-validation.json')
        batch=target_batch(plan,index,list(official.CONS_SCALARS),variance,provenance)
        manifest=json.loads(Path(plan['existing_fixture_paths']['manifest']).read_text())
        sample_index=json.loads(Path(plan['existing_fixture_paths']['index']).read_text())
        require(sample_index['names']==[Path(e['path']).stem for e in manifest['train']],
                'Original permitted-input sample order changed')
        entry=manifest['train'][0]
        raw_path=Path(plan['existing_fixture_paths']['raw'])
        require(Path(entry['path']).name==raw_path.name and raw_path.stem.endswith(plan['selection']['shot']),
                'Predeclared TRAIN shot changed')
        position=index['names'].index(raw_path.stem)
        local=np.asarray(plan['selection']['indices'])
        selected=index['offsets'][position]+local
        require(np.array_equal(local,np.arange(8)) and np.load(plan['existing_fixture_paths']['eligibility'],
                allow_pickle=False)[local].all(),'Original eight-frame eligibility differs')
        targets=bind_selection(batch,index,selected,selected)
        invalid_indices=[dict(index,names=list(reversed(index['names']))),
            dict(index,offsets=[1]+index['offsets'][1:]),dict(index,frames=index['frames']-1)]
        rejected=0
        for wrong in invalid_indices:
            reject(lambda wrong=wrong:target_batch(plan,wrong,list(official.CONS_SCALARS),variance,provenance));rejected+=1
        reject(lambda:target_batch(plan,index,list(reversed(official.CONS_SCALARS)),variance,provenance));rejected+=1
        reject(lambda:target_batch(plan,index,list(official.CONS_SCALARS),variance*.5,provenance));rejected+=1
        reject(lambda:target_batch(plan,index,list(official.CONS_SCALARS),variance,'0'*64));rejected+=1
        reject(lambda:bind_selection(batch,index,selected[::-1],selected));rejected+=1
        result['binding_controls']={'rejected':rejected,'oracle_calls':0,
                                    'global_indices':selected.tolist(),'full_population_variances':variance.tolist()}
        row=read_shot(entry,manifest['revision'],raw_path.parent,INPUT_COLUMNS)
        raw=rich_features(row)[local]
        parent=torch.load(plan['existing_fixture_paths']['parent'],map_location='cpu',weights_only=False)
        fixture=torch.load(plan['existing_fixture_paths']['fixture'],map_location='cpu',weights_only=False)
        zero=SmoothCorrection(parent).eval()
        cache=cache_parent(zero,raw,np.ones(8,dtype=np.bool_))
        inputs=prepare_inputs(raw,parameters(parent,'cpu'),parent['config'],'cpu')
        for label,model in [('zero_parent',zero),('rejected031_numerical_fixture',restore_correction(parent,fixture))]:
            if time.monotonic()>=deadline:raise TimeoutError('Original053 deadline reached')
            item=check_model(model,parent,cache,inputs,targets,variance,plan['random_seed'],deadline)
            item.update(helper_rejections(cache.fields,cache.parent_std,model.correction_scale,
                targets,variance,plan['random_seed'],deadline),fixture=label)
            result['controls'].append(item);write_json(output,result)
        identities(plan,freeze)
        require(time.monotonic()<=deadline,'Original053 deadline reached')
        result.update(complete=True,all_predeclared_checks_passed=True,frames=8,
            actual_target_frames_used_only_in_loss=8,full_variance_usable_frames=1469328,
            frozen_inputs_sources_exact=True,CPU_threads=2,process_workers=0,
            CUDA_verified=False,full_predictor_constraint_support_verified=False,production_ready=False)
    except Exception as exc:
        result.update(failed=True,failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['actual_body_seconds']=time.monotonic()-started
        write_json(output,result)
    print(json.dumps({key:value for key,value in result.items() if key!='controls'},indent=2))


if __name__=='__main__':
    main()
