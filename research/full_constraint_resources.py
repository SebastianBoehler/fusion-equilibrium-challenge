"""Actual allocation and enclosing Slurm-job cgroup memory, no estimates."""
import os
from pathlib import Path
import subprocess
import time
from participant.artifacts import write_json


def allocation(root,plan):
    job=os.environ['SLURM_JOB_ID']
    p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=10)
    write_json(root/'allocation-receipt.json',{'job':job,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    p.check_returncode()
    fields=dict(token.split('=',1) for token in p.stdout.split() if '=' in token)
    expected={'JobId':job,'JobState':'RUNNING','Account':'<account>','Partition':'cpu-galvani',
        'NumCPUs':'16','NumNodes':'1','MinMemoryNode':'32G','TimeLimit':'02:00:00','Requeue':'0','Restarts':'0'}
    if any(fields.get(key)!=value for key,value in expected.items()) or 'gpu' in fields.get('ReqTRES','')+fields.get('AllocTRES',''):
        raise ValueError('Declared059 allocation differs')
    return job


class Memory:
    def __init__(self,job):
        lines=Path('/proc/self/cgroup').read_text().splitlines()
        relative=next((line.split(':',2)[2] for line in lines if line.startswith('0::')),None)
        if relative is None:
            raise ValueError('059 requires observed unified job cgroup memory files')
        current=Path('/sys/fs/cgroup')/relative.lstrip('/')
        group=next((p for p in (current,*current.parents) if p.name==f'job_{job}'),None)
        if group is None:
            raise ValueError('Cannot identify enclosing Slurm-job memory cgroup')
        self.path=group
        for name in ('memory.current','memory.peak','memory.max','memory.stat'):
            if not (group/name).is_file():raise ValueError('Missing aggregate job cgroup memory evidence')
        if (group/'memory.max').read_text().strip()!=str(32*1024**3):
            raise ValueError('Actual job cgroup limit differs from declared32G')
        self.samples=[]

    def sample(self,scope,deadline):
        if time.monotonic()>=deadline:raise TimeoutError('Original059 cap reached')
        values={name:int((self.path/('memory.'+name)).read_text()) for name in ('current','peak','max')}
        if values['peak']>32*1024**3:raise MemoryError('Original aggregate32G cap exceeded')
        stats=dict(line.split() for line in (self.path/'memory.stat').read_text().splitlines())
        values.update(scope=scope,anon=int(stats['anon']),file=int(stats['file']),cgroup=str(self.path))
        self.samples.append(values)
        return values
