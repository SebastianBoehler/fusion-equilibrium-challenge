"""Freeze the tested contour repair and its provenance before remote dispatch."""

from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('run_data/analysis-024/proposal')
REMOTE = '<cluster-work>/fusion/contour-024-20261009'
PARENT_SHA = 'fda8f02587063b305787728b78376c979b0e21e7d7fd9a25471671a9d3c17a95'
SPLIT_SHA = '09660b12c7335441bf958abc0ddb58d63bd2f97e3da8631acb28dab443d6210f'


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def main():
    checks = json.loads(Path('run_data/analysis-023/recipe-check.json').read_text())
    validation = json.loads(Path('run_data/analysis-023/target-validation.json').read_text())
    authorization = json.loads(Path('run_data/analysis-020/follow-up-authorization.json').read_text())
    if not checks['complete'] or not validation['complete'] or checks['frames'] != 508:
        raise ValueError('Completed real-data recipe checks required')
    parent, split = Path('run_data/nn-008/mlp-shape/model.pt'), Path('run_data/full-split.json')
    if digest(parent) != PARENT_SHA or digest(split) != SPLIT_SHA:
        raise ValueError('Frozen parent or split changed')
    ROOT.mkdir(parents=True, exist_ok=False)
    files = sorted(Path('participant').glob('*.py')) + sorted(Path('vendor').rglob('*.py'))
    files += [Path('vendor/fusion_starter/fusion_scoring/masks/d3d_envelope.npz'),
              Path('ops/contour-024.sbatch'), Path('ops/contour-024-prepare.sbatch')]
    hashes = {str(p): digest(p) for p in files}
    source_sha = ROOT / 'source-sha256.txt'
    source_sha.write_text(''.join(f'{sha}  {name}\n' for name, sha in hashes.items()))
    archive = ROOT / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for path in files:
            handle.add(path, arcname=str(path), recursive=False)
        handle.add(source_sha, arcname='source-sha256.txt')
    shutil.copyfile(parent, ROOT / 'parent-model.pt')
    shutil.copyfile(split, ROOT / 'full-split.json')
    (ROOT / 'input-sha256.txt').write_text(
        f'{PARENT_SHA}  {REMOTE}/parent-model.pt\n{SPLIT_SHA}  {REMOTE}/full-split.json\n')
    result = {'status': 'Prepared, CPU validated; not dispatched', 'remote_root': REMOTE,
              'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_archive_sha256': digest(archive), 'source_sha256': hashes,
              'parent_sha256': PARENT_SHA, 'split_sha256': SPLIT_SHA,
              'human_authorization': authorization,
              'recipe_check_sha256': digest(Path('run_data/analysis-023/recipe-check.json')),
              'target_validation_sha256': digest(Path('run_data/analysis-023/target-validation.json')),
              'experiment': {'arms_in_order': ['aligned-li', 'aligned-li-kappa'],
                             'epochs_each': 10, 'batch': 2048, 'seed': 20261024,
                             'lr': 4e-6, 'li_weight': 0.05, 'kappa_weight_second_arm': 0.04419808835,
                             'parent_and_scalars_frozen': True, 'correction_grid': 17,
                             'normal_displacement_bound': 'one physical grid cell'},
              'allocation': {'cpu_preparation_gpu_count': 0, 'cpu_preparation_limit_seconds': 3600,
                             'training_gpu_count': 1, 'gpu_type': 'A100-PCIE-40GB',
                             'maximum_allocated_gpu_seconds': 3600, 'cpus': 16, 'host_mem_GB': 64,
                             'requeue': False, 'execution': 'GPU afterok CPU preparation; arms serial'},
              'success_check': 'Full float32 dev256 and old8/8 ensemble controls beat eleven; all components, failures and regimes assessed',
              'limitations': 'Only training-only CPU checks completed; full-corpus target preparation and CUDA are unverified.'}
    (ROOT / 'proposal-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'remote_root', 'source_archive_sha256', 'allocation')}))


if __name__ == '__main__':
    main()
