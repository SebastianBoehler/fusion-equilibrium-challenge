"""Exclusive foreground MAST017 execution with complete exit/signal evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from participant.artifacts import digest, write_json

ROOT = Path('run_data/mast-017')


def main():
    start = time.monotonic(); (ROOT/'launch.lock').mkdir()
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    ps = subprocess.run(['ps', '-p', '8133,8134', '-o', 'pid=,command='], capture_output=True, text=True, timeout=5)
    write_json(ROOT/'old-process-check.json', {'at': datetime.now(timezone.utc).isoformat(),
        'returncode': ps.returncode, 'stdout': ps.stdout, 'stderr': ps.stderr})
    if ps.returncode != 1 or ps.stdout.strip(): raise ValueError('Original016 process absence not verified')
    if digest(ROOT/'predeclared-plan.json') != freeze['plan_sha256']: raise ValueError('017plan changed')
    for name, expected in freeze['source_sha256'].items():
        if digest(Path(name)) != expected: raise ValueError('017source changed: '+name)
    receipt = {'started_at': datetime.now(timezone.utc).isoformat(), 'launcher_PID': os.getpid(),
               'attempts': 1, 'hard_seconds': plan['resources']['hard_local_seconds'], 'signal_events': [],
               'command': [sys.executable, '-B', '-u', '-m', 'research.mast_completion_check']}
    process = None

    def handle(sig, _frame):
        receipt['signal_events'].append({'signal': sig, 'at': datetime.now(timezone.utc).isoformat()})
        write_json(ROOT/'signal-evidence.json', receipt)
        if process is not None and process.poll() is None: process.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, handle)
    write_json(ROOT/'started-execution.json', receipt)
    with (ROOT/'execution.stdout.txt').open('xb') as out, (ROOT/'execution.stderr.txt').open('xb') as err:
        process = subprocess.Popen(receipt['command'], stdout=out, stderr=err)
        receipt['worker_PID'] = process.pid; write_json(ROOT/'started-execution.json', receipt)
        print(json.dumps({'worker_PID': process.pid, 'launcher_PID': os.getpid(), 'attempts': 1}), flush=True)
        try:
            rc = process.wait(timeout=receipt['hard_seconds']-(time.monotonic()-start)); timeout = False
        except subprocess.TimeoutExpired:
            process.kill(); rc = process.wait(); timeout = True
    receipt.update(ended_at=datetime.now(timezone.utc).isoformat(), returncode=rc, timed_out=timeout,
        wall_seconds=time.monotonic()-start,
        stdout_bytes=(ROOT/'execution.stdout.txt').stat().st_size,
        stderr_bytes=(ROOT/'execution.stderr.txt').stat().st_size,
        stdout_sha256=digest(ROOT/'execution.stdout.txt'), stderr_sha256=digest(ROOT/'execution.stderr.txt'))
    write_json(ROOT/'execution-receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__': main()
