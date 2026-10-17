"""Fine-tune a neural checkpoint against training-only contour and axis errors."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from .geometry_loss import losses as geometry_losses
from .finetune_common import geometry_targets, prepare_inputs, scalar_loss
from .nn_train import Network, chunks, export, gather, load, neighbor_index, project


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded training experiment requires the allocated CUDA GPU')
    if args.out.exists():
        raise FileExistsError('Use a fresh directory for each geometry experiment')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')
    started = time.monotonic()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    train_index, xt, pt, st = load(args.data / 'train')
    dev_index, xd, _, _ = load(args.data / 'development')
    args.out.mkdir(parents=True)
    params = {k: ck[k].to(device) for k in
              ('x_mean', 'x_std', 'psi_mean', 'basis', 'coef_scale', 's_mean', 's_std')}
    geom = geometry_targets(args.geometry, train_index, device)
    xt, xd = (prepare_inputs(v, params, cfg, device) for v in (xt, xd))
    context = cfg.get('context', 0)
    rt = torch.from_numpy(neighbor_index(train_index['offsets'], context)).to(device) if context else None
    rd = torch.from_numpy(neighbor_index(dev_index['offsets'], context)).to(device) if context else None
    ct = project(pt, params['psi_mean'], params['basis'], device) / params['coef_scale']
    sample = torch.from_numpy(np.asarray(pt[:200000])).to(device)
    variance = ((sample - params['psi_mean']).square().sum(1).mean()).item()
    del sample
    st = (torch.from_numpy(np.asarray(st)).to(device) - params['s_mean']) / params['s_std']
    k = cfg['components']
    model = Network(cfg.get('network_inputs', cfg['inputs']), k + cfg['scalars'],
                    cfg['width'], cfg['depth'], cfg['dropout']).to(device)
    model.load_state_dict(ck['state'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pixel_weights = params['coef_scale'].square() / variance
    scalar_weights = torch.tensor([cfg['scalar_weight']] * 2 +
                                  [cfg['aux_weight']] * (cfg['scalars'] - 2), device=device)
    history = []
    print(f'geometry fine-tune: {len(xt)} train frames, {int(geom[-1].sum())} valid geometry, '
          f'{args.epochs} epochs on {torch.cuda.get_device_name()}', flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        sums = torch.zeros(4, device=device)
        batches = 0
        for sl in chunks(len(xt), cfg['batch']):
            idx = order[sl]
            out = model(gather(xt, rt[idx]) if rt is not None else xt[idx])
            residual = out[:, :k] - ct[idx]
            pixel = (residual.square() * pixel_weights).sum(1).mean()
            shape = residual.square().mean()
            scal = scalar_loss(out[:, k:], st[idx], scalar_weights)
            psi = params['psi_mean'] + (out[:, :k] * params['coef_scale']) @ params['basis'].T
            boundary, axis = geometry_losses(psi, *(v[idx] for v in geom))
            loss = (pixel + cfg['shape_weight'] * shape + scal
                    + args.boundary_weight * boundary + args.axis_weight * axis)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite geometry training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += torch.stack((pixel, boundary, axis, loss)).detach()
            batches += 1
        schedule.step()
        record = dict(zip(('pixel', 'boundary', 'axis', 'loss'), (sums / batches).tolist()))
        history.append(record | {'epoch': epoch, 'elapsed_seconds': time.monotonic() - started})
        print(history[-1], flush=True)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(gather(xd, rd[sl]) if rd is not None else xd[sl])
                         for sl in chunks(len(xd), 8192)])
    ck['state'] = {name: value.cpu() for name, value in model.state_dict().items()}
    with args.checkpoint.open('rb') as handle:
        parent_sha = file_digest(handle, 'sha256').hexdigest()
    ck['config'] = cfg | {'out': str(args.out), 'geometry_tune': {
        'parent_sha256': parent_sha, 'epochs': args.epochs, 'lr': args.lr,
        'boundary_weight': args.boundary_weight, 'axis_weight': args.axis_weight, 'seed': args.seed}}
    torch.save(ck, args.out / 'model.pt')
    export(out, k, params['coef_scale'], params['basis'], params['psi_mean'],
           params['s_std'], params['s_mean'], dev_index, args.out / 'development.npz')
    (args.out / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'saved geometry model; elapsed {time.monotonic() - started:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--boundary-weight', type=float, default=0.1)
    parser.add_argument('--axis-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or min(args.boundary_weight, args.axis_weight) < 0:
        raise ValueError('Epochs and learning rate must be positive; geometry weights nonnegative')
    run(args)


if __name__ == '__main__':
    main()
