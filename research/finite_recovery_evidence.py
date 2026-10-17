"""Byte recovery and terminal-state guards; never decode numerical arrays."""

import base64
import hashlib
import json
from pathlib import Path

JOB = '2955114'
SMALL = ('accepted-job-id.txt', 'srun-client.pid', 'srun-client.log',
         'client_dispatch.json', 'accepted-job-resource-receipt.json',
         'accepted_resources.json', 'accepted_no_requeue.json', 'job-id.txt',
         'allocation-receipt.json', 'step-exit.txt', 'full-targets.json',
         'train-targets/index.json', 'train-targets/validation.json',
         'train-targets/runtime-validation.json', 'logs/2955114.out', 'logs/2955114.err')
ARRAYS = ('train-targets/values.npy', 'train-targets/valid.npy')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def one_record(records, kind):
    found = [r['value'] for r in records if r['kind'] == kind]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one {kind} record')
    return found[0]


def require_terminal_accounting(value):
    if value['returncode'] != 0 or value['timed_out']:
        raise ValueError('Successful complete accounting required')
    rows = [s.split('|') for s in value['stdout'].splitlines()]
    own = [r for r in rows if len(r) >= 16 and
           (r[0] == JOB or r[0].startswith(JOB + '.'))]
    parents = [r for r in own if r[0] == JOB]
    steps = [r for r in own if r[0].startswith(JOB + '.')]
    if len(parents) != 1 or not steps:
        raise ValueError('Completed accepted045 parent and step accounting required')
    if any(r[2] != 'COMPLETED' or r[3] != '0:0' for r in own):
        raise ValueError('Accepted045 parent and steps must all be COMPLETED0:0')
    return own


def require_terminal_snapshot(parsed_path):
    """Read original raw evidence, verifying its receipt before trusting state."""
    path = Path(parsed_path)
    if not path.name.endswith('-parsed.json'):
        raise ValueError('Saved parsed045 snapshot path required')
    prefix = path.with_name(path.name.removesuffix('-parsed.json'))
    raw = Path(str(prefix) + '.jsonl').read_bytes()
    receipt = json.loads(Path(str(prefix) + '.json').read_text())
    if (receipt.get('returncode') != 0 or receipt.get('timeout')
            or len(raw) != receipt['stdout_bytes'] or sha(raw) != receipt['stdout_sha256']):
        raise ValueError('Complete successful saved SSH evidence required')
    records = [json.loads(line) for line in raw.splitlines()]
    if one_record(records, 'start')['job_id'] != JOB:
        raise ValueError('Snapshot belongs to another job')
    own = require_terminal_accounting(one_record(records, 'accounting'))
    if not one_record(records, 'completed')['terminal_accounting_observed']:
        raise ValueError('Completed collector marker required')
    return {'snapshot_path': str(path), 'raw_sha256': sha(raw), 'accounting': own}


def decode_full_recovery(raw, destination):
    """Retain verified complete bytes; scientific target validation is separate."""
    records = [json.loads(line) for line in raw.splitlines()]
    if one_record(records, 'start')['job_id'] != JOB:
        raise ValueError('Recovery belongs to another job')
    recovered, errors, total = {}, [], 0
    destination = Path(destination)
    destination.mkdir()
    for record in records:
        if record['kind'] != 'file':
            continue
        value = record['value']
        name = value['path']
        if name not in SMALL + ARRAYS or name in recovered:
            errors.append(f'Unexpected/duplicate file: {name}')
            continue
        if not value.get('exists') or 'base64' not in value:
            errors.append(f'Incomplete file: {name}')
            continue
        data = base64.b64decode(value['base64'], validate=True)
        limit = 134217728 if name in ARRAYS else 16777216
        if (len(data) != value['bytes'] or sha(data) != value['sha256']
                or len(data) > limit or total + len(data) > 134217728):
            errors.append(f'Byte identity or declared bound failed: {name}')
            continue
        total += len(data)
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        if sha(path.read_bytes()) != value['sha256']:
            raise ValueError(f'Local recovered byte identity changed: {name}')
        recovered[name] = {'path': str(path), 'bytes': len(data), 'sha256': sha(data)}
        if not value.get('fully_recovered') or not value.get('unchanged_during_read'):
            errors.append(f'Changed during remote read: {name}')
    for name in SMALL + ARRAYS:
        if name not in recovered:
            errors.append(f'Required file unrecovered: {name}')
    require_terminal_accounting(one_record(records, 'accounting'))
    gate = one_record(records, 'recovery_gate')
    if set(gate) != {'commands_succeeded', 'parent_present', 'step_present',
                     'all_accounted_rows_completed_zero'} or not all(v is True for v in gate.values()):
        errors.append('Remote terminal/hash recovery gate failed')
    for kind, count in (('source_hashes', 25), ('input_hashes', 5)):
        value = one_record(records, kind)
        lines = value['stdout'].splitlines()
        if (value['returncode'] != 0 or value['timed_out'] or len(lines) != count
                or any(not line.endswith(': OK') for line in lines)):
            errors.append(f'Original hash command failed: {kind}')
    completed = one_record(records, 'completed')
    if (not completed.get('fully_recovered') or completed.get('errors')
            or completed.get('new_allocations') != 0
            or completed.get('array_decoding') is not False
            or completed.get('numerical_execution') is not False):
        errors.append('Remote completion marker failed')
    return {'all_bytes_recovered': not errors, 'files': recovered, 'errors': errors,
            'total_bytes': total, 'original_scientific_gates_verified': False,
            'targets_ready_for_numerical_read': False, 'records_without_file_payloads':
            [r for r in records if r['kind'] != 'file']}
