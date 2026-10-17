"""Matched PCA/direct-grid fine-tuning with contour, axis and field-gradient losses."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import array_value
from .field_loss import loss as field_loss
from .finetune_common import geometry_targets, prepare_inputs, scalar_loss
from .flux_head import decode, direct_grid, export, parameters
from .geometry_loss import losses as geometry_losses
from .nn_train import Network, chunks, gather, load, neighbor_index


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Field training requires the allocated A100')
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    started = time.monotonic()
    device = torch.device('cuda')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_index, xt, pt, st = load(args.data / 'train')
    dev_index, xd, _, _ = load(args.data / 'development')
    if args.grid_shot.stem not in train_index['names']:
        raise ValueError('Grid metadata must come from a training shot')
    table = pq.read_table(args.grid_shot, columns=['efit_grid_R', 'efit_grid_Z'])
    r, z = [torch.tensor(array_value(table[key][0]), dtype=torch.float32, device=device)
            for key in ('efit_grid_R', 'efit_grid_Z')]
    if (r.shape != (65,) or z.shape != (65,) or not torch.all(r > 0)
            or not torch.all(torch.diff(r) > 0) or not torch.all(torch.diff(z) > 0)):
        raise ValueError('Invalid physical flux grid')
    ck = direct_grid(parent) if args.representation == 'grid' else dict(parent)
    cfg, params = ck['config'], parameters(ck, device)
    parent_basis = parent['basis'].to(device)
    parent_scale = parent['coef_scale'].to(device)
    variance = parent_scale.square().sum()
    geom = geometry_targets(args.geometry, train_index, device)
    xt, xd = (prepare_inputs(v, params, cfg, device) for v in (xt, xd))
    context = cfg.get('context', 0)
    rt = torch.from_numpy(neighbor_index(train_index['offsets'], context)).to(device) if context else None
    rd = torch.from_numpy(neighbor_index(dev_index['offsets'], context)).to(device) if context else None
    # Keep physical targets once; standardizing the entire grid would duplicate 25 GB.
    pt = torch.from_numpy(np.asarray(pt)).to(device)
    st = (torch.from_numpy(np.asarray(st)).to(device) - params['s_mean']) / params['s_std']
    k = cfg['components']
    model = Network(cfg.get('network_inputs', cfg['inputs']), k + cfg['scalars'],
                    cfg['width'], cfg['depth'], cfg['dropout']).to(device)
    model.load_state_dict(ck['state'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scalar_weights = torch.tensor([cfg['scalar_weight']] * 2 +
                                  [cfg['aux_weight']] * (cfg['scalars'] - 2), device=device)
    args.out.mkdir(parents=True)
    history = []
    print(f'{args.representation}: {len(xt)} training frames, {sum(p.numel() for p in model.parameters())} '
          f'parameters, batch2048, allocated {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        sums, batches = torch.zeros(5, device=device), 0
        for sl in chunks(len(xt), 2048):
            idx = order[sl]
            out = model(gather(xt, rt[idx]) if rt is not None else xt[idx])
            psi = decode(out, params, cfg)
            error = psi - pt[idx]
            pixel = error.square().sum(1).mean() / variance
            shape = ((error @ parent_basis) / parent_scale).square().mean()
            scal = scalar_loss(out[:, k:], st[idx], scalar_weights)
            boundary, axis = geometry_losses(psi, *(v[idx] for v in geom))
            gradient = field_loss(psi, pt[idx], r, z)
            objective = (pixel + cfg['shape_weight'] * shape + scal
                         + 0.1 * boundary + 0.1 * axis + 0.05 * gradient)
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite field training objective')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += torch.stack((pixel, boundary, axis, gradient, objective)).detach()
            batches += 1
        schedule.step()
        record = dict(zip(('pixel', 'boundary', 'axis', 'gradient', 'objective'), (sums / batches).tolist()))
        history.append(record | {'epoch': epoch, 'elapsed_seconds': time.monotonic() - started})
        print(history[-1], flush=True)
    model.eval()
    with torch.no_grad():
        output = torch.cat([model(gather(xd, rd[sl]) if rd is not None else xd[sl])
                            for sl in chunks(len(xd), 2048)])
    ck['state'] = {key: value.cpu() for key, value in model.state_dict().items()}
    with args.checkpoint.open('rb') as handle:
        parent_sha = file_digest(handle, 'sha256').hexdigest()
    ck['config'] = cfg | {'out': str(args.out), 'field_tune': {
        'parent_sha256': parent_sha, 'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
        'boundary_weight': 0.1, 'axis_weight': 0.1, 'gradient_weight': 0.05, 'batch': 2048}}
    torch.save(ck, args.out / 'model.pt')
    export(output, params, cfg, dev_index, args.out / 'development.npz')
    (args.out / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'Finished {args.representation}, {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--grid-shot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--representation', choices=('pca', 'grid'), required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()
    if args.epochs <= 0 or args.lr <= 0:
        raise ValueError('Epochs and learning rate must be positive')
    run(args)
