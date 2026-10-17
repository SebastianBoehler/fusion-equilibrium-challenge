"""One exclusive persistent local MAST016 execution with retained full logs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from participant.artifacts import digest, write_json

ROOT = Path('run_data/mast-016')


def main():
    started = time.monotonic(); at = datetime.now(timezone.utc).isoformat()
    (ROOT/'supervisor.lock').mkdir()
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    if digest(ROOT/'predeclared-plan.json') != freeze['plan_sha256']:
        raise ValueError('016plan changed before launch')
    for name, expected in freeze['source_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('016frozen source changed before launch: '+name)
    receipt = {'started_at': at, 'supervisor_PID': os.getpid(), 'attempts': 1,
               'command': [sys.executable, '-B', '-m', 'research.mast_full_hybrid_check'],
               'hard_seconds': plan['resources']['hard_local_seconds']}
    write_json(ROOT/'started-execution.json', receipt)
    with (ROOT/'execution.stdout.txt').open('xb') as out, (ROOT/'execution.stderr.txt').open('xb') as err:
        remaining = receipt['hard_seconds']-(time.monotonic()-started)
        try:
            process = subprocess.Popen(receipt['command'], stdout=out, stderr=err)
            receipt['worker_PID'] = process.pid
            write_json(ROOT/'started-execution.json', receipt)
            rc = process.wait(timeout=remaining); timeout = False
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); rc = process.returncode; timeout = True
    receipt.update(ended_at=datetime.now(timezone.utc).isoformat(), returncode=rc,
        timed_out=timeout, wall_seconds=time.monotonic()-started,
        stdout_bytes=(ROOT/'execution.stdout.txt').stat().st_size,
        stderr_bytes=(ROOT/'execution.stderr.txt').stat().st_size,
        stdout_sha256=digest(ROOT/'execution.stdout.txt'), stderr_sha256=digest(ROOT/'execution.stderr.txt'))
    write_json(ROOT/'execution-receipt.json', receipt)


if __name__ == '__main__': main()
