"""Native observed Galvani hierarchical cgroup-v1 job memory accounting."""
from pathlib import Path
import time
from research.full_constraint_resources import allocation


class Memory:
    def __init__(self,job):
        lines=Path('/proc/self/cgroup').read_text().splitlines()
        relative=next((line.split(':',2)[2] for line in lines
                       if 'memory' in line.split(':',2)[1].split(',')),None)
        if relative is None:raise ValueError('Observed native memory controller unavailable in allocation')
        current=Path('/sys/fs/cgroup/memory')/relative.lstrip('/')
        group=next((p for p in (current,*current.parents) if p.name==f'job_{job}'),None)
        if group is None:raise ValueError('Cannot identify enclosing own Slurm job memory cgroup')
        for name in ('memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes',
                     'memory.use_hierarchy','memory.stat'):
            if not (group/name).is_file():raise ValueError('Missing native aggregate job memory evidence')
        if int((group/'memory.limit_in_bytes').read_text())!=32*1024**3:
            raise ValueError('Actual job cgroup memory limit differs from original32G')
        if (group/'memory.use_hierarchy').read_text().strip()!='1':
            raise ValueError('Hierarchical job memory accounting required')
        self.path=group;self.samples=[]

    def sample(self,scope,deadline):
        if time.monotonic()>=deadline:raise TimeoutError('Original fullsupport cap reached')
        values={key:int((self.path/name).read_text()) for key,name in
                [('current','memory.usage_in_bytes'),('peak','memory.max_usage_in_bytes'),
                 ('max','memory.limit_in_bytes')]}
        if values['peak']>32*1024**3:raise MemoryError('Original aggregate32G cap exceeded')
        stats=dict(line.split() for line in (self.path/'memory.stat').read_text().splitlines())
        values.update(scope=scope,anon=int(stats['total_rss']),file=int(stats['total_cache']),
                      cgroup=str(self.path),controller='native_v1_hierarchical')
        self.samples.append(values)
        return values
