"""Build a complete DIII-D submission from one or more neural checkpoints (averaged)."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .data import read_shot
from .flux_head import flux_delta, parameters
from .nn_train import Network, neighbor_index
from .rich import INPUT_COLUMNS, canonicalize_currents, rich_features


class Predictor:
    def __init__(self, paths: list[Path], device: str = 'cpu'):
        self.device = torch.device(device)
        self.members = []
        for path in paths:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            cfg = ck['config']
            net = Network(cfg.get('network_inputs', cfg['inputs']), cfg['components'] + cfg['scalars'], cfg['width'], cfg['depth'])
            net.load_state_dict(ck['state'])
            net.eval().to(self.device)
            self.members.append((net, parameters(ck, self.device), cfg))

    @torch.no_grad()
    def predict_inputs(self, x: np.ndarray) -> dict:
        """Predict every EFIT frame of one shot; x must be the complete shot in time order."""
        psi, scal = 0.0, 0.0
        for net, p, cfg in self.members:
            raw = canonicalize_currents(x) if cfg.get('canon_ip') else np.asarray(x, dtype=np.float32)
            z = ((torch.from_numpy(raw).to(self.device) - p['x_mean'])
                 / p['x_std']).clamp(-cfg['clip'], cfg['clip'])
            context = cfg.get('context', 0)
            if context:
                z = z[torch.from_numpy(neighbor_index([0, len(z)], context)).to(self.device)].flatten(1)
            out = net(z)
            k = cfg['components']
            psi = psi + p['psi_mean'] + flux_delta(out, p, cfg)
            scal = scal + out[:, k:k + 2] * p['s_std'][:2] + p['s_mean'][:2]
        n = len(self.members)
        psi, scal = (psi / n).cpu().numpy(), (scal / n).cpu().numpy()
        return {'psirz': psi.reshape(-1, 65, 65), 'q95': scal[:, 0], 'betaN': scal[:, 1]}

    def predict(self, row: dict) -> dict:
        return self.predict_inputs(rich_features({k: row[k] for k in INPUT_COLUMNS}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--order-manifest', type=Path, required=True)
    parser.add_argument('--expected-shots', type=int, required=True)
    parser.add_argument('--archive-name', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    order = json.loads(args.order_manifest.read_text())
    entries, config = manifest['shots'], manifest['config']
    if (not config.startswith('diii_d_') or not args.archive_name.endswith('.zip')
            or Path(args.archive_name).name != args.archive_name):
        raise ValueError('DIII-D configuration and a basename ending in .zip required')
    if (len(entries) != args.expected_shots or order['shot_count'] != args.expected_shots
            or order['dataset_revision'] != manifest['revision']
            or order['ordered_paths'] != [e['path'] for e in entries]):
        raise ValueError('The complete official test stream order was not verified')
    args.out.mkdir(parents=True, exist_ok=True)
    target, zipped = args.out / f'{config}.npz', args.out / args.archive_name
    if target.exists() or zipped.exists():
        raise FileExistsError('Use a fresh output directory for immutable submission artifacts')
    model = Predictor(args.checkpoints, args.device)
    started = time.monotonic()
    frames = []
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, entry in enumerate(entries):
            row = read_shot(entry, manifest['revision'], args.cache / 'raw', columns=INPUT_COLUMNS)
            count = len(row['efit_times'])
            prediction = model.predict(row)
            frames.append(count)
            for key, expected in (('psirz', (count, 65, 65)), ('q95', (count,)), ('betaN', (count,))):
                # Float16 rounding measurably degrades contour-derived metrics.
                values = np.asarray(prediction[key], dtype=np.float32)
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f'Invalid prediction: shot {index}, {key}')
                with archive.open(f'shot_{index:04d}_{key}.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
            if (index + 1) % 100 == 0 or index + 1 == len(entries):
                print(f'Built {index + 1}/{len(entries)} shots', flush=True)
    with np.load(target, allow_pickle=False) as upload:
        expected_keys = {f'shot_{i:04d}_{k}' for i in range(args.expected_shots) for k in ('psirz', 'q95', 'betaN')}
        if set(upload.files) != expected_keys:
            raise ValueError('Wrong submission keys')
        for i, count in enumerate(frames):
            for key, shape in (('psirz', (count, 65, 65)), ('q95', (count,)), ('betaN', (count,))):
                values = upload[f'shot_{i:04d}_{key}']
                if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError('Saved submission validation failed')
    with zipfile.ZipFile(zipped, 'w', zipfile.ZIP_STORED) as archive:
        archive.write(target, target.name)
    with zipped.open('rb') as handle:
        zipped_hash = file_digest(handle, 'sha256').hexdigest()
    receipt = {'model': 'rich-mlp-ensemble',
               'checkpoints': {str(p): sha256(p.read_bytes()).hexdigest() for p in args.checkpoints},
               'dataset_revision': manifest['revision'], 'dataset_config': config, 'shots': len(entries),
               'frames': sum(frames), 'test_data_used_only_for_inference': True,
               'manifest_sha256': sha256(args.manifest.read_bytes()).hexdigest(),
               'order_manifest_sha256': sha256(args.order_manifest.read_bytes()).hexdigest(),
               'flux_dtype': 'float32', 'zip_sha256': zipped_hash, 'zip_bytes': zipped.stat().st_size,
               'validation': 'all keys, shapes, float32 dtypes, finite values',
               'elapsed_seconds': time.monotonic() - started}
    (args.out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
