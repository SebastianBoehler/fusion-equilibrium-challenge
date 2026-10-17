"""Check the actual target pack, objective and exported checkpoint on training frames."""

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.inductance_tune import TargetBatch, objective
from participant.nn_train import Network
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.functional_batch_probe import PARENT_SHA


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    root = Path('run_data/analysis-020')
    output = root / 'recipe-cpu-check.json'
    assert not output.exists()
    parent_path = Path('run_data/nn-008/mlp-shape/model.pt')
    with parent_path.open('rb') as handle:
        assert file_digest(handle, 'sha256').hexdigest() == PARENT_SHA
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    original = json.loads((root / 'batch-and-correction-probe.json').read_text())
    index = json.loads((root / 'packing-check/train/index.json').read_text())
    offsets = index['offsets']
    indices = np.concatenate([np.asarray(receipt['frames']) + a for receipt, a in
                              zip(original['training_shots'], offsets[:-1], strict=True)])
    raw = np.load(root / 'training-target-pack.npz')
    packed = TargetBatch(root / 'packing-check/targets', index)
    points, weights, interior, li = packed.get(indices, 'cpu')
    torch.testing.assert_close(points, torch.from_numpy(raw['points']), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.from_numpy(raw['boundary_weights']), rtol=0, atol=0)
    torch.testing.assert_close(interior, torch.from_numpy(raw['interior_weights']), rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(li, torch.from_numpy(raw['target_li']), rtol=2e-6, atol=2e-6)
    x = prepare_inputs(raw['x'], parameters(parent, 'cpu'), parent['config'], 'cpu')
    truth = torch.from_numpy(raw['psi'])
    r, z = (torch.from_numpy(v) for v in (packed.r, packed.z))
    results = {}
    for arm, weight in (('pixel', 0), ('pixel-li', 0.05)):
        model = SmoothCorrection(parent).train()
        initial, initial_scalars = model(x)
        with torch.no_grad():
            source = model.parent(x)
            assert torch.equal(initial, decode(source, parameters(parent, 'cpu'), parent['config']))
        optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
        for _ in range(6):
            loss, pixel, field = objective(model, x, truth, r, z, (points, weights, interior),
                                           li, parent['coef_scale'].square().sum(), weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1.0)
            assert torch.isfinite(norm)
            optimizer.step()
        ck = fold_checkpoint(parent, model)
        cfg = ck['config']
        folded = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                         cfg['width'], cfg['depth'], cfg['dropout']).eval()
        folded.load_state_dict(ck['state'])
        with torch.no_grad():
            actual_psi, actual_scalar = model(x)
            out = folded(x)
            maximum_error = float((actual_psi - decode(out, parameters(ck, 'cpu'), cfg)).abs().max())
            assert maximum_error < 3e-6
            assert torch.equal(actual_scalar, initial_scalars)
            assert torch.equal(actual_scalar, out[:, 4225:])
        assert all(value.grad is None for value in model.parent.parameters())
        assert torch.equal(ck['state']['head.1.weight'][4225:], parent['state']['head.1.weight'][parent['config']['components']:])
        results[arm] = {'finite_updates': 6, 'zero_initial_equivalence': True,
                        'scalar_outputs_and_weights_exact': True, 'frozen_parent_gradients': True,
                        'export_flux_max_absolute_error': maximum_error, 'trainable_parameters': 296225}
    sources = {str(path): file_digest(path.open('rb'), 'sha256').hexdigest() for path in
               map(Path, ('participant/inductance_loss.py', 'participant/inductance_targets.py',
                          'participant/inductance_tune.py', 'participant/smooth_correction.py'))}
    receipt = {'training_only': True, 'frames': len(indices), 'full_training_pack_frames': index['frames'],
               'target_pack_order_and_masks_verified': True, 'parent_sha256': PARENT_SHA,
               'source_sha256': sources, 'controls': results, 'gpu_scheduled': False,
               'limitations': 'CPU only,3 training shots; full corpus preparation and CUDA not run'}
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
