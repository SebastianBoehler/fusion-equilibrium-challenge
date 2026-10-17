"""Explicit one-attempt045 recovery launcher; no training or job dispatch."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from research.finite_recovery_evidence import decode_full_recovery, require_terminal_snapshot, sha

ROOT = Path('run_data/analysis-054')
REMOTE = Path('run_data/analysis-045/monitor-source/remote-full-recovery.py')


def require_preparation():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT / 'source-freeze.json').read_text())
    if sha((ROOT / 'predeclared-plan.json').read_bytes()) != freeze['plan_sha256']:
        raise ValueError('Frozen recovery preparation changed')
    for name, expected in plan['source_sha256'].items() | freeze['source_sha256'].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError(f'Pinned recovery source/receipt changed: {name}')
    source = REMOTE.read_bytes()
    if sha(source) != plan['remote_full_recovery_sha256']:
        raise ValueError('Original remote full-recovery source changed')
    return source


def recover(snapshot, source):
    # Claim the exclusive attempt only after authentic completed accounting.
    Path('run_data/analysis-045/full-recovery-attempt.lock').mkdir()
    at = datetime.now().astimezone()
    prefix = Path('run_data/analysis-045') / ('full-recovery-' + at.strftime('%Y%m%d-%H%M%S'))
    attempt = {'started_at': at.isoformat(), 'snapshot': snapshot,
               'source_sha256': sha(source), 'SSH_attempts': 1, 'new_allocations': 0}
    with Path(str(prefix) + '-attempt.json').open('x') as stream:
        json.dump(attempt, stream, indent=2)
    started = time.monotonic()
    command = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', '<cluster>',
               'python3 -B -u -']
    try:
        process = subprocess.run(command, input=source, capture_output=True, timeout=120)
        out, err, code, timed_out = process.stdout, process.stderr, process.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, err, code, timed_out = exc.stdout or b'', exc.stderr or b'', None, True
    # Always retain complete received bytes and return code BEFORE parsing.
    Path(str(prefix) + '.jsonl').write_bytes(out)
    Path(str(prefix) + '.stderr.txt').write_bytes(err)
    receipt = {'started_at': at.isoformat(), 'ended_at': datetime.now().astimezone().isoformat(),
               'command': command, 'returncode': code, 'timeout': timed_out,
               'wall_seconds': time.monotonic() - started, 'stdout_bytes': len(out),
               'stderr_bytes': len(err), 'stdout_sha256': sha(out), 'stderr_sha256': sha(err)}
    Path(str(prefix) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
    review = {'all_bytes_recovered': False, 'targets_ready_for_numerical_read': False}
    try:
        review = decode_full_recovery(out, Path(str(prefix) + '-files'))
        if code != 0 or timed_out or not review['all_bytes_recovered']:
            raise ValueError('Full recovery incomplete; preserve accepted-job guard')
    except Exception as exc:
        review['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        Path(str(prefix) + '-receipt.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'receipt': str(prefix) + '-receipt.json',
                      'all_bytes_recovered': True, 'original_gate_review_still_required': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'recover'))
    parser.add_argument('saved_snapshot', type=Path)
    args = parser.parse_args()
    source = require_preparation()
    try:
        snapshot = require_terminal_snapshot(args.saved_snapshot)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({'status': 'BLOCKED_NO_REMOTE_ACTION', 'reason': str(exc),
                          'SSH_attempts': 0, 'new_allocations': 0}))
        raise SystemExit(2)
    if args.action == 'check':
        print(json.dumps({'status': 'TERMINAL_SUCCESS_OBSERVED_RECOVERY_ELIGIBLE',
                          'SSH_attempts': 0, 'new_allocations': 0}))
    else:
        recover(snapshot, source)


if __name__ == '__main__':
    main()
