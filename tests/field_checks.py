"""Check direct-head equivalence and differentiable field loss on real released data."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from participant.data import array_value
from participant.field_loss import loss
from participant.flux_head import decode, direct_grid, parameters
from participant.nn_train import Network


def main():
    ck = torch.load('run_data/nn-008/mlp-shape/model.pt', map_location='cpu', weights_only=False)
    x = np.load('run_data/nn-008/development/x.npy', mmap_mode='r')
    values = torch.from_numpy(np.array(x[:256], copy=True))
    normalized = ((values - ck['x_mean']) / ck['x_std']).clamp(-ck['config']['clip'], ck['config']['clip'])
    outputs = []
    for checkpoint in (ck, direct_grid(ck)):
        config = checkpoint['config']
        model = Network(config['inputs'], config['components'] + config['scalars'],
                        config['width'], config['depth'])
        model.load_state_dict(checkpoint['state'])
        model.eval()
        with torch.no_grad():
            out = model(normalized)
            outputs.append((decode(out, parameters(checkpoint, 'cpu'), config),
                            out[:, config['components']:]))
    # Folding two matrix products changes float32 summation order (measured ≤1.2e-6).
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    path = next(Path('run_data/local-eightshot').rglob('*.parquet'))
    table = pq.read_table(path, columns=['efit_psirz', 'efit_grid_R', 'efit_grid_Z'])
    row = {key: array_value(table[key][0]) for key in table.column_names}
    truth = torch.tensor(np.asarray(row['efit_psirz'][:8]), dtype=torch.float32)
    r, z = [torch.tensor(row[key], dtype=torch.float32) for key in ('efit_grid_R', 'efit_grid_Z')]
    assert float(loss(truth, truth, r, z)) == 0
    assert float(loss(truth + .2, truth, r, z)) < 1e-9
    prediction = (truth + .01 * torch.linspace(-1, 1, 65)[None, None, :]).requires_grad_()
    objective = loss(prediction, truth, r, z)
    objective.backward()
    assert float(objective.detach()) > 0
    assert torch.isfinite(prediction.grad).all() and float(prediction.grad.abs().sum()) > 0
    print('Field checks passed: PCA/direct-grid initial outputs agree; real-field gradients finite')


if __name__ == '__main__':
    main()
