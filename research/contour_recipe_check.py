"""Check the actual scorer-contour target pack, paired objective and grid export."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.contour_normal_loss import loss, physical_points, projected, targets
from participant.contour_training import ARMS, KAPPA_WEIGHT, ScoredContourBatch, contour_objective
from participant.data import read_shot
from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.nn_train import Network
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error

ROOT = Path('run_data/analysis-023')


def check_arm(parent, x, truth, fixed, li, r, z, reference, arm):
    model = SmoothCorrection(parent).train()
    initial, scalars = model(x)
    variance = parent['coef_scale'].square().sum()
    with torch.no_grad():
        assert torch.equal(initial, decode(model.parent(x), parameters(parent, 'cpu'), parent['config']))
    weight = KAPPA_WEIGHT if arm == 'aligned-li-kappa' else 0
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    history = []
    for _ in range(6):
        value, pixel, inductance, kappa = contour_objective(model, x, truth, r, z, fixed, li, variance, weight)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
        assert torch.isfinite(norm)
        optimizer.step()
        history.append({'pixel': float(pixel), 'li': float(inductance), 'kappa': float(kappa),
                        'gradient_norm': float(norm)})
    ck = fold_checkpoint(parent, model)
    cfg = ck['config']
    folded = Network(cfg['inputs'], cfg['components'] + cfg['scalars'], cfg['width'], cfg['depth'], cfg['dropout']).eval()
    folded.load_state_dict(ck['state'])
    with torch.no_grad():
        final, final_scalars = model(x)
        out = folded(x)
        maximum = float((final - decode(out, parameters(ck, 'cpu'), cfg)).abs().max())
    assert maximum < 3e-6
    assert torch.equal(final_scalars, scalars) and torch.equal(out[:, 4225:], scalars)
    assert all(p.grad is None for p in model.parent.parameters())
    assert torch.equal(ck['state']['head.1.weight'][4225:], parent['state']['head.1.weight'][parent['config']['components']:])
    actual, boundary, failed = derive_predictions(final.numpy().reshape(-1, 65, 65), reference[0], fast_score._geometry())
    return {'kappa_weight': weight, 'six_updates': history, 'export_max_difference': maximum,
            'zero_initial_equivalence': True, 'scalar_outputs_weights_exact': True,
            'parent_gradients_absent': True, 'extraction_failures': failed,
            'boundary_mean': float(np.mean(boundary)),
            'derived_errors': {name: error(actual[:, j], reference[1][:, j])
                               for j, name in enumerate(fast_score.official.CONS_SCALARS)}}


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    receipt = ROOT / 'recipe-check.json'
    if receipt.exists():
        raise FileExistsError('Preserve completed recipe checks')
    old = Path('run_data/analysis-020/packing-check')
    index = json.loads((old / 'train/index.json').read_text())
    manifest = json.loads((old / 'training-only-manifest.json').read_text())
    rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for e in manifest['train']]
    x = np.concatenate([rich_features(row) for row in rows])
    fields = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    assert len(x) == index['frames'] == len(fields) == 508
    for key in ('points', 'weights', 'mask_bits', 'target_li', 'r', 'z'):
        a, b = (np.load(path / f'{key}.npy') for path in (old / 'targets', ROOT / 'raw-target-migration'))
        assert np.array_equal(a, b), key
    batch = ScoredContourBatch(ROOT / 'scored-targets', index)
    picks = batch.valid_indices
    assert len(picks) == 508
    p, w, interior, li = batch.get(picks, 'cpu')
    r, z = (torch.from_numpy(v) for v in (batch.r, batch.z))
    truth = torch.from_numpy(fields[picks])
    reference = fast_score.official.build_reference(fields[picks].astype(float), *fast_score._geometry())
    contours = physical_points(p, r, z)
    inverse_error = float(abs(contours.numpy() - np.asarray(reference[0])).max())
    assert inverse_error < 3e-7
    normal = targets(truth, r, z, p, contours, interior)
    assert float(loss(truth, p, normal)) < 1e-10
    path, sha = MODELS['parent']
    assert digest(path) == sha
    parent = torch.load(path, map_location='cpu', weights_only=False)
    inputs = prepare_inputs(x[picks], parameters(parent, 'cpu'), parent['config'], 'cpu')
    base = SmoothCorrection(parent)(inputs)[0].reshape(-1, 65, 65)
    base_loss = loss(base, p, normal)
    invariance = {name: float(abs(loss(value, p, normal) - base_loss)) for name, value in
                  [('weak', base * 1e-12), ('sign', -base), ('gauge', base + 1), ('amplitude', base * 1.1)]}
    assert max(invariance.values()) < 2e-5
    displacement = float((projected(base, p, normal) - contours).norm(dim=2).max())
    assert displacement <= float(normal[-1]) + 1e-6
    try:
        loss(torch.zeros_like(base), p, normal)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Undefined zero field was accepted')
    arms = {arm: check_arm(parent, inputs, truth, (p, w, interior), li, r, z,
                          reference, arm) for arm in ARMS}
    result = {'complete': True, 'training_only': True, 'frames': len(picks), 'train_shots': 3,
              'parent_sha256': sha, 'target_manifest_sha256': batch.manifest_sha,
              'raw_packer_migration_exact': True, 'physical_contour_inverse_max_difference': inverse_error,
              'normal_loss_invariance_errors': invariance, 'zero_field_rejected': True,
              'max_displacement_m': displacement, 'movement_bound_m': float(normal[-1]),
              'arms': arms, 'elapsed_cpu_seconds': time.monotonic() - started,
              'limitations': 'Three training shots only; no corpus/CUDA/held-out improvement established.'}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
