"""Validate the input-driven MAST candidate and preserve scored DIII-D predictions."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .mast_green import predict


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def validate(data, n):
    expected = {'psirz': (n, 65, 65), 'q95': (n,), 'betaN': (n,)}
    if set(data) != set(expected):
        raise ValueError('Unexpected prediction keys')
    for key, shape in expected.items():
        values = data[key]
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f'Invalid saved prediction: {key}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--order-manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--d3d-dir', type=Path, required=True)
    parser.add_argument('--d3d-id', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    order = json.loads(args.order_manifest.read_text())
    entries = manifest['shots']
    if (manifest['config'] != 'mast_public_test' or len(entries) != 1206
            or order['dataset_revision'] != manifest['revision']
            or order['ordered_paths'] != [e['path'] for e in entries]
            or len({e['path'] for e in entries}) != len(entries)):
        raise ValueError('The complete official MAST stream order is required')
    source_files = ('mast_inputs.py', 'mast_green.py', 'mast_submit.py')
    source_hashes = {f: digest(Path(__file__).parent / f) for f in source_files}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'mast_public_test.npz'
    zipped = args.out / 'fusion-mast-green-001.zip'
    if target.exists() or zipped.exists():
        raise FileExistsError('Use a fresh output directory')
    started, counts = time.monotonic(), []
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for i, entry in enumerate(entries):
            path = args.cache / 'inputs' / (Path(entry['path']).stem + '.npz')
            deadline = time.monotonic() + 900
            while not path.exists() or not zipfile.is_zipfile(path):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Pinned input preparation did not finish: {entry["path"]}')
                time.sleep(.5)
            with np.load(path, allow_pickle=False) as inputs:
                if (str(inputs['source_hash']) != entry['sha256']
                        or str(inputs['revision']) != manifest['revision']):
                    raise ValueError('Prepared MAST inputs differ from the pinned source')
                row = {k: inputs[k] for k in inputs.files}
            n = len(row['efit_times'])
            result = predict(row)
            validate(result, n)
            counts.append(n)
            for key, values in result.items():
                with archive.open(f'shot_{i:04d}_{key}.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
            if (i + 1) % 50 == 0 or i + 1 == len(entries):
                print(f'Built MAST {i + 1}/{len(entries)}; '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    with np.load(target, allow_pickle=False) as saved:
        keys = {f'shot_{i:04d}_{k}' for i in range(len(entries))
                for k in ('psirz', 'q95', 'betaN')}
        if set(saved.files) != keys:
            raise ValueError('Saved MAST archive has missing or extra keys')
        for i, n in enumerate(counts):
            validate({k: saved[f'shot_{i:04d}_{k}'] for k in ('psirz', 'q95', 'betaN')}, n)

    d3d_root = args.d3d_dir
    d3d_receipt = json.loads((d3d_root / 'build-receipt.json').read_text())
    d3d_file = d3d_root / 'diii_d_public_test.npz'
    d3d_archives = list(d3d_root.glob('fusion-*.zip'))
    if len(d3d_archives) != 1 or d3d_receipt['dataset_config'] != 'diii_d_public_test':
        raise ValueError('Exactly one verified DIII-D source archive is required')
    d3d_zip = d3d_archives[0]
    if digest(d3d_zip) != d3d_receipt['zip_sha256']:
        raise ValueError('The scored DIII-D submission archive changed')
    d3d_hash = digest(d3d_file)
    with zipfile.ZipFile(d3d_zip) as source:
        with source.open(d3d_file.name) as handle:
            if file_digest(handle, 'sha256').hexdigest() != d3d_hash:
                raise ValueError('DIII-D predictions differ from their scored source archive')
    if source_hashes != {f: digest(Path(__file__).parent / f) for f in source_files}:
        raise ValueError('Inference source changed during the build')
    provenance = {
        'method': 'Thomson-pressure circular plasma current + measured coil Green fields',
        'dataset_revision': manifest['revision'], 'mast_shots': len(entries),
        'mast_frames': sum(counts), 'MAST_equilibrium_data_used': False,
        'MAST_envelope_used': False, 'MAST_leaderboard_feedback_used': False,
        'DIII_D_source_submission': args.d3d_id, 'DIII_D_predictions_sha256': d3d_hash,
        'source_sha256': source_hashes,
        'assumptions': ['circular Gaussian current density from electron-pressure moments',
                        'midplane current centre; equal ion and electron temperatures',
                        'vacuum toroidal field from the 24-turn TF circuit'],
        'limitations': ['no nonlinear force-balance closure or passive vessel eddy currents',
                        'cylindrical approximation for q95; accuracy only known after official scoring'],
        'TF_hardware_reference': 'https://ccfe.ukaea.uk/wp-content/uploads/2019/12/MAST-U_RP_2019_v1.pdf',
        'AI_assistance': 'Codex implemented and verified this input-driven baseline',
    }
    (args.out / 'methods.json').write_text(json.dumps(provenance, indent=2))
    with zipfile.ZipFile(zipped, 'x', zipfile.ZIP_STORED) as archive:
        archive.write(d3d_file, d3d_file.name)
        archive.write(target, target.name)
        archive.writestr('manifest.json', json.dumps(provenance, indent=2))
    receipt = dict(provenance, zip_sha256=digest(zipped), zip_bytes=zipped.stat().st_size,
                   mast_predictions_sha256=digest(target),
                   manifest_sha256=sha256(args.manifest.read_bytes()).hexdigest(),
                   order_manifest_sha256=sha256(args.order_manifest.read_bytes()).hexdigest(),
                   validation='every saved key, shape, float32 dtype and finite value; exact DIII-D copy',
                   elapsed_seconds=time.monotonic() - started)
    (args.out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
