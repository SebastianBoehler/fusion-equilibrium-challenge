"""Matched frozen-parent smooth corrections, with or without fixed-contour li loss."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from .finetune_common import prepare_inputs
from .flux_head import export, parameters
from .inductance_batch import TargetBatch
from .inductance_loss import inductance
from .nn_train import Network, chunks, load
from .smooth_correction import SmoothCorrection, fold_checkpoint


def objective(model, x, truth, r, z, geometry, target_li, variance, li_weight):
    psi, _ = model(x)
    psi = psi.reshape(-1, 65, 65)
    pixel = (psi - truth.reshape(-1, 65, 65)).square().sum((1, 2)).mean() / variance
    li = (inductance(psi, r, z, *geometry) - target_li).square().mean()
    loss = pixel + li_weight * li
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite smooth-correction objective')
    return loss, pixel, li


def run(args):
    from .contour_training import ARMS, KAPPA_WEIGHT, ScoredContourBatch, contour_objective

    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the approved allocated A100')
    if args.out.exists():
        raise FileExistsError('Use an immutable fresh training directory')
    started = time.monotonic()
    device = torch.device('cuda')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with args.checkpoint.open('rb') as handle:
        parent_sha = file_digest(handle, 'sha256').hexdigest()
    train_index, xt, pt, _ = load(args.data / 'train')
    dev_index, xd, _, _ = load(args.data / 'development')
    aligned = args.arm in ARMS
    targets = (ScoredContourBatch if aligned else TargetBatch)(args.targets, train_index)
    params = parameters(parent, device)
    xt, xd = (prepare_inputs(v, params, parent['config'], device) for v in (xt, xd))
    pt = torch.from_numpy(np.asarray(pt)).to(device)
    r, z = (torch.from_numpy(value).to(device) for value in (targets.r, targets.z))
    model = SmoothCorrection(parent).to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    variance = parent['coef_scale'].square().sum().to(device)
    li_weight = 0.05 if aligned or args.arm == 'pixel-li' else 0.0
    kappa_weight = KAPPA_WEIGHT if args.arm == 'aligned-li-kappa' else 0.0
    eligible = targets.valid_indices if aligned else np.arange(len(xt))
    history = []
    args.out.mkdir(parents=True)
    print(f'{args.arm}: train{len(xt)}, usable{len(eligible)}, trainable{sum(p.numel() for p in model.correction.parameters())}, '
          f'allocated{torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
    for epoch in range(10):
        model.train()
        order = eligible[torch.randperm(len(eligible), device='cpu').numpy()]
        keys = ('pixel', 'li_mse', 'kappa_mse', 'objective') if aligned else ('pixel', 'li_mse', 'objective')
        sums = torch.zeros(len(keys), device=device)
        batches = 0
        for sl in chunks(len(eligible), 2048):
            cpu_indices = order[sl]
            idx = torch.from_numpy(cpu_indices).to(device)
            points, weights, interior, li_target = targets.get(cpu_indices, device)
            if aligned:
                loss, pixel, li, kappa = contour_objective(model, xt[idx], pt[idx], r, z,
                    (points, weights, interior), li_target, variance, kappa_weight)
                terms = (pixel, li, kappa, loss)
            else:
                loss, pixel, li = objective(model, xt[idx], pt[idx], r, z,
                    (points, weights, interior), li_target, variance, li_weight)
                terms = (pixel, li, loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite correction gradient')
            optimizer.step()
            sums += torch.stack(terms).detach()
            batches += 1
        history.append(dict(zip(keys, (sums / batches).tolist())) |
                       {'epoch': epoch, 'elapsed_seconds': time.monotonic() - started})
        print(history[-1], flush=True)
    if any(p.grad is not None for p in model.parent.parameters()):
        raise AssertionError('Frozen parent received gradients')
    ck = fold_checkpoint(parent, model)
    ck['config'] = ck['config'] | {'out': str(args.out), 'inductance_tune': {
        'parent_sha256': parent_sha, 'target_manifest_sha256': targets.manifest_sha,
        'arm': args.arm, 'seed': args.seed, 'epochs': 10, 'lr': 4e-6, 'li_weight': li_weight,
        'batch': 2048, 'scalar_head_frozen': True, 'kappa_weight': kappa_weight,
        'training_frames': len(xt), 'usable_frames': len(eligible),
        'target_contour': targets.description if aligned else 'raw EFIT',
        'local_scorer_version': '3.2.0', 'normal_movement_bound_grid_cells': 1 if aligned else None}}
    torch.save(ck, args.out / 'model.pt')
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
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('pixel', 'pixel-li', 'aligned-li', 'aligned-li-kappa'), required=True)
    parser.add_argument('--seed', type=int, default=20261021)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
