"""Serial031 training of the validated local critical-level geometry surrogate."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from .contour_training import KAPPA_WEIGHT
from .critical_artifact import save_correction
from .critical_training import ARMS, CriticalBatch, objective
from .finetune_common import prepare_inputs
from .flux_head import export, parameters
from .nn_train import Network, chunks, load
from .smooth_correction import SmoothCorrection, fold_checkpoint


def sha256(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the authorized allocated A100')
    if args.out.exists():
        raise FileExistsError('Preserve existing or active training output')
    validation = json.loads(args.validation.read_text())
    if not validation['complete'] or not validation['all_usable_shapes_finiteness_weights_indices_levels_passed']:
        raise ValueError('Full-corpus critical target validation required')
    for name, digest in validation['file_sha256'].items():
        if sha256(args.critical / name) != digest:
            raise ValueError('Validated critical target changed before training')
    started = time.monotonic()
    device = torch.device('cuda')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    parent_sha = sha256(args.checkpoint)
    train_index, xt, pt, _ = load(args.data / 'train')
    dev_index, xd, _, _ = load(args.data / 'development')
    batch = CriticalBatch(args.contours, args.critical, train_index)
    if len(batch.valid_indices) != validation['usable_frames']:
        raise ValueError('Validated critical target usable count changed')
    params = parameters(parent, device)
    xt, xd = (prepare_inputs(value, params, parent['config'], device) for value in (xt, xd))
    pt = torch.from_numpy(np.asarray(pt)).to(device)
    r, z = (torch.from_numpy(value).to(device) for value in (batch.r, batch.z))
    model = SmoothCorrection(parent).to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    variance = parent['coef_scale'].square().sum().to(device)
    eligible = batch.valid_indices
    args.out.mkdir(parents=True)
    history = []
    print(f'{args.arm}: total{len(xt)},usable{len(eligible)},trainable'
          f'{sum(p.numel() for p in model.correction.parameters())},'
          f'allocated{torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
    for epoch in range(10):
        model.train()
        order = eligible[torch.randperm(len(eligible), device='cpu').numpy()]
        sums = torch.zeros(4, device=device)
        batches = 0
        for sl in chunks(len(eligible), 2048):
            picks = order[sl]
            indices, weights = batch.critical_get(picks, device)
            points, quadrature, interior, li_target = batch.get(picks, device)
            rows = torch.from_numpy(picks).to(device)
            value, pixel, li, kappa = objective(model, xt[rows], pt[rows], r, z,
                                               (points, quadrature, interior), li_target,
                                               variance, indices, weights, args.arm)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite critical correction gradient')
            optimizer.step()
            sums += torch.stack((pixel, li, kappa, value)).detach()
            batches += 1
        history.append(dict(zip(('pixel', 'li_mse', 'kappa_mse', 'objective'), (sums / batches).tolist())) |
                       {'epoch': epoch, 'elapsed_seconds': time.monotonic() - started})
        print(history[-1], flush=True)
    if any(p.grad is not None for p in model.parent.parameters()):
        raise AssertionError('Frozen parent received gradients')
    ck = fold_checkpoint(parent, model)
    ck['config'] = ck['config'] | {'out': str(args.out), 'critical_tune': {
        'parent_sha256': parent_sha, 'target_manifest_sha256': batch.manifest_sha,
        'critical_validation_sha256': sha256(args.validation),
        'critical_index_sha256': validation['file_sha256']['index.json'],
        'arm': args.arm, 'seed': args.seed, 'epochs': 10, 'lr': 4e-6,
        'batch': 2048, 'scalar_head_frozen': True, 'parent_frozen': True,
        'li_weight': .05 if args.arm == 'critical-li-kappa' else 0,
        'kappa_weight': KAPPA_WEIGHT, 'training_frames': len(xt), 'usable_frames': len(eligible),
        'original_reference_excluded_frames': validation['original_reference_excluded_frames'],
        'unsupported_critical_events': validation['unsupported_critical_events'],
        'target_contour': batch.critical_description, 'local_scorer_version': '3.2.0',
        'movement_bound_grid_cells': 1, 'critical_event_training_only': True,
        'level_derivative': 'live interpolation at true active event; local surrogate, not exact future closure derivative'}}
    torch.save(ck, args.out / 'model.pt')
    save_correction(model, ck['config']['critical_tune'], args.out / 'correction-state.pt')
    cfg = ck['config']
    exported = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                       cfg['width'], cfg['depth'], cfg['dropout']).to(device).eval()
    exported.load_state_dict(ck['state'])
    with torch.no_grad():
        output = torch.cat([exported(xd[sl]) for sl in chunks(len(xd), 2048)])
    export(output, parameters(ck, device), cfg, dev_index, args.out / 'development.npz')
    (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    print(f'Finished{args.arm},elapsed{time.monotonic()-started:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'data', 'contours', 'critical', 'validation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
