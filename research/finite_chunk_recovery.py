"""One explicit, bounded phase of read-only045 recovery; never auto-retry."""

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from research.finite_recovery_evidence import sha, require_terminal_accounting

ROOT = Path('run_data/analysis-056')
SOURCE = ROOT / 'frozen-source/remote-chunk-recovery.py'


def capture(phase):
    freeze = json.loads((ROOT / 'execution-freeze.json').read_text())
    for name, expected in freeze['source_sha256'].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError(f'Frozen chunk recovery changed: {name}')
    plan = json.loads((ROOT / 'execution-plan.json').read_text())
    if sha((ROOT / 'execution-plan.json').read_bytes()) != freeze['plan_sha256']:
        raise ValueError('Chunk recovery execution plan changed')
    for previous in range(phase):
        review = json.loads((ROOT / f'phase-{previous}-review.json').read_text())
        if not review['complete']:
            raise ValueError('Earlier phase incomplete; no automatic retry')
    (ROOT / f'phase-{phase}.lock').mkdir()
    command = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', '<cluster>',
               f'python3 -B -u - {phase}']
    started = time.monotonic()
    at = datetime.now(timezone.utc).isoformat()
    try:
        process = subprocess.run(command, input=SOURCE.read_bytes(), capture_output=True, timeout=120)
        out, err, code, timeout = process.stdout, process.stderr, process.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, err, code, timeout = exc.stdout or b'', exc.stderr or b'', None, True
    prefix = ROOT / f'phase-{phase}'
    Path(str(prefix) + '.jsonl').write_bytes(out)
    Path(str(prefix) + '.stderr.txt').write_bytes(err)
    receipt = {'started_at': at, 'returncode': code, 'timeout': timeout,
               'wall_seconds': time.monotonic() - started, 'command': command,
               'source_sha256': sha(SOURCE.read_bytes()), 'stdout_bytes': len(out),
               'stdout_sha256': sha(out), 'stderr_bytes': len(err), 'stderr_sha256': sha(err)}
    Path(str(prefix) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
    review = {'complete': False, 'files': {}, 'chunks': [], 'errors': []}
    try:
        records = []
        for line in out.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                review['errors'].append(str(exc))
        by = {r['kind']: r['value'] for r in records if r['kind'] not in ('file', 'chunk')}
        require_terminal_accounting(by['accounting'])
        for kind, count in (('source_hashes', 25), ('input_hashes', 5)):
            value = by[kind]
            lines = value['stdout'].splitlines()
            if value['returncode'] != 0 or len(lines) != count or not all(s.endswith(': OK') for s in lines):
                raise ValueError(f'Original {kind} failed')
        for record in records:
            kind, value = record['kind'], record['value']
            if kind not in ('file', 'chunk'):
                continue
            data = base64.b64decode(value['base64'], validate=True)
            if len(data) != value['bytes'] or sha(data) != value['sha256']:
                raise ValueError('Chunk/smallfile byte identity failed')
            relative = Path(value['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unexpected recovered path')
            if kind == 'file':
                if phase != 0 or value['path'] not in plan['small_files']:
                    raise ValueError('Unexpected smallfile')
                destination = ROOT / 'recovered-files' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                review['files'][value['path']] = {'path': str(destination), 'bytes': len(data), 'sha256': sha(data)}
            else:
                if value['path'] not in plan['arrays'] or len(data) > 2097152:
                    raise ValueError('Unexpected/oversized array chunk')
                destination = ROOT / 'chunks' / f"{relative.name}-{value['offset']:09d}.bin"
                destination.parent.mkdir(exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(data)
                review['chunks'].append({k: v for k, v in value.items() if k != 'base64'} | {'local_path': str(destination)})
        if phase == 0 and set(review['files']) != set(plan['small_files']):
            raise ValueError('Missing required smallfile')
        completed = by['completed']
        if (code != 0 or timeout or review['errors'] or not completed['complete']
                or completed['phase'] != phase or not completed['unchanged_during_read']):
            raise ValueError('Phase incomplete; no automatic retry')
        for name, bounds in plan['arrays'].items():
            if by['catalog'][name] != bounds:
                raise ValueError('Original target catalog changed')
        review.update(complete=True, catalog=by['catalog'], records_without_payloads=
            [r for r in records if r['kind'] not in ('chunk', 'file')])
    except Exception as exc:
        review['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        Path(str(prefix) + '-review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'phase': phase, 'complete': True, 'chunks': len(review['chunks'])}))


def assemble():
    plan = json.loads((ROOT / 'execution-plan.json').read_text())
    reviews = [json.loads((ROOT / f'phase-{i}-review.json').read_text()) for i in range(4)]
    if not all(r['complete'] for r in reviews):
        raise ValueError('Four complete phases required')
    files = dict(reviews[0]['files'])
    raw_parts = []
    for phase in range(4):
        prefix = ROOT / f'phase-{phase}'
        raw = Path(str(prefix) + '.jsonl').read_bytes()
        receipt = json.loads(Path(str(prefix) + '.json').read_text())
        if (receipt['returncode'] != 0 or receipt['timeout'] or len(raw) != receipt['stdout_bytes']
                or sha(raw) != receipt['stdout_sha256']):
            raise ValueError('Original complete phase capture changed')
        raw_parts.append(raw)
    for name, expected in plan['arrays'].items():
        chunks = sorted([c for r in reviews for c in r['chunks'] if c['path'] == name], key=lambda c: c['offset'])
        destination = ROOT / 'recovered-files' / name
        offset = 0
        with destination.open('xb') as stream:
            for chunk in chunks:
                data = Path(chunk['local_path']).read_bytes()
                if chunk['offset'] != offset or len(data) != chunk['bytes'] or sha(data) != chunk['sha256']:
                    raise ValueError('Array chunk gap/order/byte identity failed')
                stream.write(data)
                offset += len(data)
        if offset != expected['bytes'] or sha(destination.read_bytes()) != expected['sha256']:
            raise ValueError('Whole recovered array identity failed')
        files[name] = expected | {'path': str(destination)}
    prefix = ROOT / 'complete-recovery'
    raw = b''.join(raw_parts)
    Path(str(prefix) + '.jsonl').write_bytes(raw)
    Path(str(prefix) + '.json').write_text(json.dumps({'returncode': 0, 'timeout': False,
        'stdout_bytes': len(raw), 'stdout_sha256': sha(raw), 'kind': 'four_complete_disjoint_phase_captures',
        'phase_receipts': [str(ROOT / f'phase-{i}.json') for i in range(4)]}, indent=2) + '\n')
    review = {'all_bytes_recovered': True, 'errors': [], 'files': files,
        'total_bytes': sum(v['bytes'] for v in files.values()), 'transport': 'four_disjoint_bounded_phases',
        'records_without_file_payloads': reviews[-1]['records_without_payloads'],
        'original_scientific_gates_verified': False, 'targets_ready_for_numerical_read': False}
    Path(str(prefix) + '-receipt.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'complete_recovery_receipt': str(prefix) + '-receipt.json', 'files': len(files)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('0', '1', '2', '3', 'assemble'))
    action = parser.parse_args().action
    assemble() if action == 'assemble' else capture(int(action))
