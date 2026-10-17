"""Check geometry coordinates, precision, and gradients on a real training shot."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from participant.data import array_value
from participant.geometry_loss import losses
from participant.geometry_targets import COLUMNS, targets


def main():
    path = next(Path('run_data/local-eightshot').rglob('*.parquet'))
    table = pq.read_table(path, columns=list(COLUMNS))
    row = {k: array_value(table[k][0]) for k in COLUMNS}
    points, values, span, valid = targets(row)
    assert valid.sum() > len(valid) * 0.9
    psi = torch.from_numpy(np.asarray(row['efit_psirz'], dtype=np.float32)).requires_grad_()
    args = [torch.from_numpy(a) for a in (points, values, span, valid)]
    boundary, axis = losses(psi, *args)
    assert float(boundary.detach()) < 1e-10
    assert float(axis.detach()) < 1e-10
    # A constant gauge shift preserves these geometric constraints.
    shifted = losses(psi + 0.2, *args)
    assert all(float(v.detach()) < 1e-10 for v in shifted)
    # A spatial perturbation must produce a finite, nonzero training gradient.
    ramp = torch.linspace(-1, 1, 65)[None, None, :]
    perturbed = sum(losses(psi + 0.01 * ramp, *args))
    perturbed.backward()
    assert float(perturbed.detach()) > 1e-5
    assert torch.isfinite(psi.grad).all() and float(psi.grad.abs().sum()) > 0
    print(f'geometry checks passed: {path.name}, {int(valid.sum())}/{len(valid)} frames')


if __name__ == '__main__':
    main()
