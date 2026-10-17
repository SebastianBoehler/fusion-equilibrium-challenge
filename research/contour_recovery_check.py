"""Verify the completed024 recovery before any candidate comparison."""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from research.contour_comparison import validate
from research.inductance_comparison import RECEIPT, sha256


ROOT = Path('run_data/analysis-024')
RECOVERED = Path('run_data/contour-024')
MAPPED = {'train-contours/index.json': 'index.json',
          'repo/source-sha256.txt': 'source-sha256.txt',
          'logs/2950821.out': '2950821.out', 'logs/2950821.err': '2950821.err'}


def verify_files(remote):
    for name, digest in remote.items():
        if sha256(RECOVERED / MAPPED.get(name, name)) != digest:
            raise ValueError(f'Remote/local artifact differs: {name}')


def verify_source(remote, proposal):
    if remote['source.tar.gz'] != proposal['source_archive_sha256']:
        raise ValueError('Recovered source archive differs from submitted source')
    for name, digest in proposal['source_sha256'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Local evaluation source differs from submitted source: {name}')


def checkpoint_recovery():
    """Record checkpoint provenance while the large exports finish separately."""
    output = ROOT / 'checkpoint-recovery.json'
    if output.exists():
        raise FileExistsError('Preserve existing checkpoint recovery receipt')
    remote = json.loads((ROOT / 'remote-artifact-sha256.json').read_text())
    proposal = json.loads((ROOT / 'proposal/proposal-receipt.json').read_text())
    received = {name: sha for name, sha in remote.items() if not name.endswith('development.npz')}
    verify_files(received)
    verify_source(received, proposal)
    incumbent = json.loads(RECEIPT.read_text())
    hashes = {}
    for arm in ('aligned-li', 'aligned-li-kappa'):
        hashes[arm] = remote[f'{arm}/model.pt']
        validate(RECOVERED / arm / 'model.pt', hashes[arm], arm, incumbent)
    result = {'checkpoint_metadata_and_source_verified': True,
              'checkpoint_sha256': hashes, 'verified_recovered_artifact_sha256': received,
              'development_export_recovery_pending': True, 'locked_shots_read': False}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('Both actual024 checkpoints, metadata, reports, logs and source provenance verified.', flush=True)


def main():
    torch.set_num_threads(2)
    output = ROOT / 'completed-recovery.json'
    if output.exists():
        raise FileExistsError('Preserve the completed recovery receipt')
    remote = json.loads((ROOT / 'remote-artifact-sha256.json').read_text())
    proposal = json.loads((ROOT / 'proposal/proposal-receipt.json').read_text())
    verify_files(remote)
    verify_source(remote, proposal)
    incumbent = json.loads(RECEIPT.read_text())
    index = json.loads(Path('run_data/nn-008/development/index.json').read_text())
    paired_path = ROOT / 'paired-decision.json'
    paired = json.loads(paired_path.read_text())
    if not paired['complete'] or paired['candidate_promoted']:
        raise ValueError('Expected the completed024 paired rejection')
    if len(index['names']) != 256 or index['frames'] != 57315:
        raise ValueError('Expected the entire frozen development group')
    arms = {}
    for arm in ('aligned-li', 'aligned-li-kappa'):
        path = RECOVERED / arm
        digest = remote[f'{arm}/model.pt']
        validate(path / 'model.pt', digest, arm, incumbent)
        history = json.loads((path / 'history.json').read_text())
        if [row['epoch'] for row in history] != list(range(10)):
            raise ValueError('Incomplete actual training history')
        if not all(np.isfinite(list(row.values())).all() for row in history):
            raise ValueError('Nonfinite training history')
        keys = [f'shot_{i:04d}_{key}' for i in range(256)
                for key in ('psirz', 'q95', 'betaN')]
        with np.load(path / 'development.npz', allow_pickle=False) as exported:
            if exported.files != keys:
                raise ValueError('Development export keys or order changed')
            for i, (a, b) in enumerate(zip(index['offsets'][:-1], index['offsets'][1:], strict=True)):
                for key in ('psirz', 'q95', 'betaN'):
                    value = exported[f'shot_{i:04d}_{key}']
                    shape = (b - a, 65, 65) if key == 'psirz' else (b - a,)
                    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f'Invalid actual development export: {arm}/{i}/{key}')
        arms[arm] = {'checkpoint_sha256': digest, 'recipe_metadata_verified': True,
                     'frozen_input_statistics_parent_scalar_weights_exact': True,
                     'epochs': 10, 'development_shots': 256, 'development_frames': 57315,
                     'development_export_float32_keys_order_shapes_finiteness_passed': True,
                     'development': json.loads((path / 'development.scores.json').read_text()),
                     'regression': json.loads((path / 'regression.scores.json').read_text())}
    result = {'observed_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),
              'gpu_job_id': 2950821, 'state': 'COMPLETED', 'exit': '0:0',
              'start': '2026-10-09T08:47:27+02:00', 'end': '2026-10-09T09:16:56+02:00',
              'actual_allocated_gpu_seconds': 1769, 'time_limit_seconds': 3600,
              'total_campaign_allocated_a100_seconds': 2974 + 1769,
              'includes_allocation_preparation_training_scoring': True,
              'remote_source_and_input_checks_after_completion_passed': True,
              'source_archive_sha256': remote['source.tar.gz'],
              'all_remote_local_artifact_hashes_match': True, 'artifact_sha256': remote,
              'arms': arms, 'complete': True, 'ensemble_comparison_complete': True,
              'paired_decision_sha256': sha256(paired_path),
              'locked_shots_read': False, 'candidate_promoted': False}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('Completed024 recovery verified; both actual exports and checkpoint metadata pass.', flush=True)
    for arm, data in arms.items():
        print(arm, data['checkpoint_sha256'], flush=True)


if __name__ == '__main__':
    main()
