"""Residual MLP from rich inputs to a PCA flux basis plus scalar heads.

The flux basis is orthonormal, so squared coefficient error equals squared pixel error
inside the retained subspace: the main loss term is exactly 1 - R2_psi (up to the
truncation residual). q95 and betaN are separate standardized heads; li and the axis
position are auxiliary supervision only.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .rich import canonicalize_currents

SUBMITTED = 2  # q95, betaN are the first two packed scalars


class Block(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, x):
        return x + self.net(x)


class Network(nn.Module):
    def __init__(self, inputs, outputs, width=1024, depth=6, dropout=0.0):
        super().__init__()
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.Sequential(*[Block(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, outputs))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def load(packed: Path):
    index = json.loads((packed / 'index.json').read_text())
    return (index, np.load(packed / 'x.npy', mmap_mode='r'), np.load(packed / 'psi.npy', mmap_mode='r'),
            np.load(packed / 'scalars.npy', mmap_mode='r'))


def neighbor_index(offsets, context):
    """Row indices of the K previous and K next EFIT frames within each shot, edge-repeated."""
    rows = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        local = np.arange(b - a)[:, None] + np.arange(-context, context + 1)[None, :]
        rows.append(a + np.clip(local, 0, b - a - 1))
    return np.concatenate(rows).astype(np.int64)


def gather(x, rows):
    return x[rows].flatten(1) if rows is not None else x


def chunks(n, size=65536):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def pca(psi, device, components):
    n = len(psi)
    mean = torch.zeros(4225, dtype=torch.float64, device=device)
    for s in chunks(n):
        mean += torch.from_numpy(np.asarray(psi[s])).to(device, torch.float64).sum(0)
    mean /= n
    cov = torch.zeros(4225, 4225, dtype=torch.float64, device=device)
    for s in chunks(n):
        block = torch.from_numpy(np.asarray(psi[s])).to(device, torch.float64) - mean
        cov += block.T @ block
    values, vectors = torch.linalg.eigh(cov / n)
    values, vectors = values.flip(0), vectors.flip(1)
    explained = (values.cumsum(0) / values.sum()).cpu().numpy()
    return mean.float(), vectors[:, :components].float().contiguous(), values[:components].float(), explained


def project(psi, mean, basis, device):
    out = torch.empty(len(psi), basis.shape[1], dtype=torch.float32, device=device)
    for s in chunks(len(psi)):
        out[s] = (torch.from_numpy(np.asarray(psi[s])).to(device) - mean) @ basis
    return out


def standardize_inputs(x, device, transform=np.asarray):
    x64 = torch.from_numpy(transform(x)).to(device, torch.float64)
    mean, std = x64.mean(0), x64.std(0)
    std = torch.where(std > 1e-9, std, torch.ones_like(std))
    return mean.float(), std.float()


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    started = time.time()
    train_index, x_train, psi_train, s_train = load(args.data / 'train')
    dev_index, x_dev, psi_dev, s_dev = load(args.data / 'development')
    if args.max_train_frames:
        if args.context:
            raise ValueError('Frame truncation is incompatible with temporal context')
        x_train, psi_train, s_train = (a[:args.max_train_frames] for a in (x_train, psi_train, s_train))
    print(f'train frames {len(x_train)}, development frames {len(x_dev)}, device {device}', flush=True)

    stat_device = device if device.type == 'cuda' else torch.device('cpu')  # float64 statistics
    psi_mean, basis, eig, explained = pca(psi_train, stat_device, args.components)
    psi_mean, basis, eig = psi_mean.to(device), basis.to(device), eig.to(device)
    print('PCA explained:', {k: float(1 - explained[k - 1]) for k in (8, 16, 32, 64, 128, 256, 512)
                             if k <= len(explained)}, flush=True)
    c_train = project(psi_train, psi_mean, basis, device)
    c_dev = project(psi_dev, psi_mean, basis, device)
    total_var = float(((torch.from_numpy(np.asarray(psi_train[:200000])).to(device) - psi_mean) ** 2)
                      .sum(1).mean())
    transform = canonicalize_currents if args.canon_ip else np.asarray
    x_mean, x_std = (v.to(device) for v in standardize_inputs(x_train, stat_device, transform))

    def prep_x(x):
        return ((torch.from_numpy(transform(x)).to(device) - x_mean) / x_std).clamp(-args.clip, args.clip)

    xt, xd = prep_x(x_train), prep_x(x_dev)
    rows_t = rows_d = None
    if args.context:
        rows_t = torch.from_numpy(neighbor_index(train_index['offsets'], args.context)).to(device)
        rows_d = torch.from_numpy(neighbor_index(dev_index['offsets'], args.context)).to(device)
    st = torch.from_numpy(np.asarray(s_train)).to(device)
    sd = torch.from_numpy(np.asarray(s_dev)).to(device)
    s_mean = torch.nanmean(st, 0)
    s_std = torch.sqrt(torch.nanmean((st - s_mean) ** 2, 0))
    coef_scale = eig.sqrt()
    # Partially whitened coefficient targets keep small-variance shape modes trainable.
    ct, cd = c_train / coef_scale, c_dev / coef_scale
    weight_pixel = eig / total_var  # per standardized coefficient, sums to explained fraction
    weight_shape = torch.full_like(eig, 1.0 / args.components)

    k, n_scalar = args.components, s_train.shape[1]
    model = Network(xt.shape[1] * (2 * args.context + 1), k + n_scalar, args.width, args.depth, args.dropout).to(device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(xt) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                   pct_start=0.05, div_factor=20, final_div_factor=200)
    scalar_weights = torch.tensor([args.scalar_weight] * SUBMITTED + [args.aux_weight] * (n_scalar - SUBMITTED),
                                  device=device)

    def losses(out, c, s):
        dc = out[:, :k] - c
        pixel = (dc ** 2 * weight_pixel).sum(1).mean()
        shape = (dc ** 2 * weight_shape).sum(1).mean()
        z = (out[:, k:] - (s - s_mean) / s_std)
        finite = torch.isfinite(z)
        scal = torch.where(finite, z, torch.zeros_like(z)) ** 2
        scal = (scal.sum(0) / finite.sum(0).clamp(min=1) * scalar_weights).sum()
        return pixel + args.shape_weight * shape + scal, pixel

    def evaluate():
        model.eval()
        outs = []
        with torch.no_grad():
            for s in chunks(len(xd)):
                outs.append(model(gather(xd, rows_d[s]) if rows_d is not None else xd[s]))
        model.train()
        out = torch.cat(outs)
        psi_err = ((out[:, :k] - cd) * coef_scale)
        # residual outside the retained subspace counts against R2_psi as well
        resid = 0.0
        sse = 0.0
        sst = 0.0
        dev_psi_mean = 0.0
        n = 0
        for s in chunks(len(psi_dev)):
            block = torch.from_numpy(np.asarray(psi_dev[s])).to(device)
            recon = psi_mean + (c_dev[s]) @ basis.T
            resid += float(((block - recon) ** 2).sum())
            dev_psi_mean += float(block.sum())
            n += block.numel()
        dev_psi_mean /= n
        for s in chunks(len(psi_dev)):
            block = torch.from_numpy(np.asarray(psi_dev[s])).to(device)
            sst += float(((block - dev_psi_mean) ** 2).sum())
        sse = float((psi_err ** 2).sum()) + resid
        pred_s = out[:, k:] * s_std + s_mean
        r2s = []
        for j in range(n_scalar):
            ok = torch.isfinite(sd[:, j])
            y, p = sd[ok, j], pred_s[ok, j]
            r2s.append(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        return 1 - sse / sst, r2s, 1 - resid / sst, out

    step = 0
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(xt), device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x_batch = gather(xt, rows_t[idx]) if rows_t is not None else xt[idx]
            if args.noise > 0:
                x_batch = x_batch + args.noise * torch.randn_like(x_batch)
            loss, pixel = losses(model(x_batch), ct[idx], st[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running += float(pixel)
        r2_psi, r2s, ceiling, _ = evaluate()
        history.append({'epoch': epoch, 'train_pixel_frac': running / steps_per_epoch, 'dev_r2_psi': r2_psi,
                        'dev_r2_scalars': r2s, 'elapsed': time.time() - started})
        print(f'epoch {epoch:3d} train(1-R2psi) {running / steps_per_epoch:.3e} dev R2psi {r2_psi:.6f} '
              f'(PCA ceiling {ceiling:.6f}) scalars {np.round(r2s, 4).tolist()} {time.time() - started:.0f}s',
              flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = {'state': model.state_dict(), 'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                  'psi_mean': psi_mean.cpu(), 'basis': basis.cpu(), 'coef_scale': coef_scale.cpu(),
                  's_mean': s_mean.cpu(), 's_std': s_std.cpu(), 'config': vars(args) | {
                      'data': str(args.data), 'out': str(args.out), 'inputs': xt.shape[1],
                      'network_inputs': xt.shape[1] * (2 * args.context + 1),
                      'scalars': n_scalar}}
    torch.save(checkpoint, args.out / 'model.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    _, _, _, out = evaluate()
    export(out, k, coef_scale, basis, psi_mean, s_std, s_mean, dev_index, args.out / 'development.npz')
    print('saved', args.out, f'{time.time() - started:.0f}s', flush=True)


def export(out, k, coef_scale, basis, psi_mean, s_std, s_mean, index, path):
    offsets = index['offsets']
    arrays = {}
    with torch.no_grad():
        for i in range(len(index['names'])):
            a, b = offsets[i], offsets[i + 1]
            psi = psi_mean + (out[a:b, :k] * coef_scale) @ basis.T
            scal = out[a:b, k:k + 2] * s_std[:2] + s_mean[:2]
            arrays[f'shot_{i:04d}_psirz'] = psi.reshape(-1, 65, 65).cpu().numpy().astype(np.float16)
            arrays[f'shot_{i:04d}_q95'] = scal[:, 0].cpu().numpy().astype(np.float32)
            arrays[f'shot_{i:04d}_betaN'] = scal[:, 1].cpu().numpy().astype(np.float32)
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--components', type=int, default=256)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scalar-weight', type=float, default=0.1)
    parser.add_argument('--aux-weight', type=float, default=0.02)
    parser.add_argument('--shape-weight', type=float, default=0.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--max-train-frames', type=int, default=0)
    parser.add_argument('--context', type=int, default=0)
    parser.add_argument('--canon-ip', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
