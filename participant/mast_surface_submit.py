"""Frozen MAST iteration 002: replace q95, preserve flux, betaN and DIII-D."""

from hashlib import sha256
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .mast_submit import digest, validate
from .mast_surface import predict_q95


def main():
    baseline = Path('run_data/submission-mast-001')
    cache = Path('run_data/mast-001')
    out = Path('run_data/submission-mast-002')
    manifest_path = cache / 'data.json'
    manifest = json.loads(manifest_path.read_text())
    order = json.loads((cache / 'stream-order-verified.json').read_text())
    original = json.loads((baseline / 'build-receipt.json').read_text())
    protocol = json.loads(Path('protocols/mast-surface-002.json').read_text())
    entries = manifest['shots']
    if (manifest['config'] != 'mast_public_test' or len(entries) != 1206
            or order['dataset_revision'] != manifest['revision']
            or order['ordered_paths'] != [entry['path'] for entry in entries]):
        raise ValueError('The complete pinned official MAST order is required')
    source = baseline / 'mast_public_test.npz'
    if digest(source) != original['mast_predictions_sha256']:
        raise ValueError('The official MAST baseline predictions changed')
    source_files = ('mast_surface.py', 'mast_surface_submit.py', 'mast_inputs.py',
                    'mast_green.py', 'mast_submit.py')
    hashes = {name: digest(Path(__file__).parent / name) for name in source_files}
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'mast_public_test.npz'
    zipped = out / 'fusion-mast-surface-002.zip'
    if target.exists() or zipped.exists():
        raise FileExistsError('Iteration 002 requires a fresh output directory')
    started, counts, ratios = time.monotonic(), [], []
    with np.load(source, allow_pickle=False) as predictions, \
            zipfile.ZipFile(source) as old, \
            zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for index, entry in enumerate(entries):
            path = cache / 'inputs' / (Path(entry['path']).stem + '.npz')
            with np.load(path, allow_pickle=False) as inputs:
                row = dict(inputs)
            if (str(row['source_hash']) != entry['sha256']
                    or str(row['revision']) != manifest['revision']):
                raise ValueError('Input cache differs from the pinned source')
            prefix = f'shot_{index:04d}_'
            psi, beta = predictions[prefix + 'psirz'], predictions[prefix + 'betaN']
            q95 = predict_q95(row, psi)
            validate(dict(psirz=psi, q95=q95, betaN=beta), len(row['efit_times']))
            for key in ('psirz', 'betaN'):
                name = prefix + key + '.npy'
                payload = old.read(name)
                new.writestr(name, payload)
            with new.open(prefix + 'q95.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, q95, allow_pickle=False)
            counts.append(len(q95))
            ratios.extend((q95 / predictions[prefix + 'q95']).tolist())
            if (index + 1) % 50 == 0 or index + 1 == len(entries):
                print(f'MAST surface {index + 1}/{len(entries)}; '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target) as new, \
            np.load(target, allow_pickle=False) as saved:
        if set(new.namelist()) != set(old.namelist()):
            raise ValueError('Iteration 002 keys differ from the full baseline')
        for index, frames in enumerate(counts):
            prefix = f'shot_{index:04d}_'
            validate({key: saved[prefix + key] for key in ('psirz', 'q95', 'betaN')}, frames)
            for key in ('psirz', 'betaN'):
                name = prefix + key + '.npy'
                if sha256(old.read(name)).digest() != sha256(new.read(name)).digest():
                    raise ValueError('A frozen MAST flux or betaN array changed')
    d3d = Path('run_data/submission-010/diii_d_public_test.npz')
    if digest(d3d) != original['DIII_D_predictions_sha256']:
        raise ValueError('The frozen official DIII-D predictions changed')
    if hashes != {name: digest(Path(__file__).parent / name) for name in source_files}:
        raise ValueError('Inference source changed during the build')
    provenance = dict(
        protocol, dataset_revision=manifest['revision'], source_sha256=hashes,
        mast_shots=len(entries), mast_frames=sum(counts),
        DIII_D_source_submission=973093, DIII_D_predictions_sha256=digest(d3d),
        MAST_equilibrium_data_used=False, MAST_envelope_used=False,
        MAST_leaderboard_feedback_used=True,
        feedback_use='Official 973259 component scores identified q95 as the first improvement target.',
        AI_assistance='Codex implemented and checked the input-driven surface integral.')
    with zipfile.ZipFile(zipped, 'x', zipfile.ZIP_STORED) as archive:
        archive.write(d3d, d3d.name)
        archive.write(target, target.name)
        archive.writestr('manifest.json', json.dumps(provenance, indent=2))
    receipt = dict(provenance, zip_sha256=digest(zipped), zip_bytes=zipped.stat().st_size,
                   mast_predictions_sha256=digest(target),
                   manifest_sha256=digest(manifest_path),
                   validation='All arrays finite float32; every MAST psi/betaN payload byte-identical; exact DIII-D copy',
                   q95_ratio_percentiles=np.percentile(ratios, [0, 25, 50, 75, 100]).tolist(),
                   elapsed_seconds=time.monotonic() - started)
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
