"""Actual own allocation/step/job/ancestor memory evidence before validation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
from research.full_constraint_resources_native import Memory as OriginalMemory


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(root):
    (root/'running.lock').mkdir()
    job=os.environ['SLURM_JOB_ID']
    result={'complete':False,'job_id':job,'started_at':datetime.now(timezone.utc).isoformat(),
            'TRAIN_MAST_arrays_read':False,'optimizer_steps':0,'GPU_count':0,'groups':[]}
    out=root/'memory-diagnostic.json'
    command=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=10)
    result['allocation']={'returncode':command.returncode,'stdout':command.stdout,'stderr':command.stderr}
    lines=Path('/proc/self/cgroup').read_text().splitlines();result['proc_self_cgroup']=lines
    result['memory_mounts']=[s for s in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup' in s and 'memory' in s]
    relative=next((s.split(':',2)[2] for s in lines if 'memory' in s.split(':',2)[1].split(',')),None)
    write(out,result)
    if relative is None:raise ValueError('Native memory controller absent in actual allocation')
    current=Path('/sys/fs/cgroup/memory')/relative.lstrip('/')
    mount=Path('/sys/fs/cgroup/memory')
    for group in (current,*current.parents):
        if mount not in (group,*group.parents):break
        values={'path':str(group),'role':'job' if group.name==f'job_{job}' else 'step_or_ancestor'}
        for name in ['memory.limit_in_bytes','memory.soft_limit_in_bytes','memory.usage_in_bytes',
                     'memory.max_usage_in_bytes','memory.use_hierarchy','memory.memsw.limit_in_bytes',
                     'memory.failcnt','memory.oom_control','memory.stat']:
            p=group/name
            if p.is_file():values[name]=p.read_text()
        result['groups'].append(values)
    write(out,result)
    try:
        old=OriginalMemory(job)
        result['original060_guard']={'passed':True,'path':str(old.path)}
    except Exception as exc:
        result['original060_guard']={'passed':False,'failure':type(exc).__name__+': '+str(exc)}
    write(out,result)
    command.check_returncode()
    f=dict(t.split('=',1) for t in command.stdout.split() if '=' in t)
    expected={'JobId':job,'JobState':'RUNNING','NumCPUs':'16','NumNodes':'1','MinMemoryNode':'32G',
              'TimeLimit':'00:05:00','Account':'<account>','Partition':'cpu-galvani','Requeue':'0','Restarts':'0'}
    if any(f.get(k)!=v for k,v in expected.items()) or 'gpu' in f.get('ReqTRES','')+f.get('AllocTRES',''):
        raise ValueError('061 actual diagnostic allocation differs')
    if not any(g['role']=='job' for g in result['groups']):raise ValueError('Own job hierarchy not identified')
    result.update(complete=True,actual_memory_values_saved_before_old_guard=True,
                  full_constraint_support_verified=False,production_ready=False)
    write(out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);run(parser.parse_args().root)
