"""Training-only CPU diagnostic of matched PCA/grid objectives and AdamW updates."""

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from participant.data import read_shot
from participant.field_loss import loss as field_loss
from participant.finetune_common import prepare_inputs, scalar_loss
from participant.flux_head import decode, direct_grid, parameters
from participant.geometry_loss import losses as geometry_losses
from participant.geometry_targets import COLUMNS, targets
from participant.nn_prepare import TARGET_COLUMNS
from participant.nn_train import Network
from participant.rich import INPUT_COLUMNS, rich_features


def training_batch():
    split = json.loads(Path('run_data/full-split.json').read_text())
    rng = np.random.default_rng(20261019)
    parts, receipts = [], []
    r0 = z0 = None
    for entry in split['train'][:3]:
        cache = Path('run_data/analysis-019/training-raw')
        columns = tuple(dict.fromkeys(INPUT_COLUMNS + TARGET_COLUMNS + COLUMNS))
        row = read_shot(entry, split['revision'], cache, columns)
        path = cache / Path(entry['path']).name
        with path.open('rb') as handle:
            digest = file_digest(handle, 'sha256').hexdigest()
        assert digest == entry['sha256']
        geom = targets(row)
        valid = np.flatnonzero(geom[-1])
        assert len(valid) >= 32
        idx = rng.choice(valid, 32, replace=False)
        x = rich_features({key: row[key] for key in INPUT_COLUMNS})[idx]
        psi = np.asarray(row['efit_psirz'], np.float32)[idx].reshape(32, -1)
        scalars = np.column_stack([row[key] for key in TARGET_COLUMNS[1:]])[idx]
        r, z = (np.asarray(row[key], np.float32) for key in ('efit_grid_R', 'efit_grid_Z'))
        if r0 is not None:
            assert np.array_equal(r, r0) and np.array_equal(z, z0)
        r0, z0 = r, z
        parts.append((x, psi, scalars, *(values[idx] for values in geom)))
        receipts.append({'path': str(path), 'sha256': digest, 'frame_indices': idx.tolist()})
        if len(parts) == 3:
            break
    assert len(parts) == 3, 'Three verified training shots required'
    batch = [torch.from_numpy(np.concatenate(values).copy()) for values in zip(*parts, strict=True)]
    return batch, torch.from_numpy(r0), torch.from_numpy(z0), receipts


def objective(model, x, psi, scalars, geom, params, cfg, parent, r, z):
    out = model(x)
    prediction = decode(out, params, cfg)
    error = prediction - psi
    pixel = error.square().sum(1).mean() / parent['coef_scale'].square().sum()
    shape = ((error @ parent['basis']) / parent['coef_scale']).square().mean()
    weights = torch.tensor([cfg['scalar_weight']] * 2 + [cfg['aux_weight']] * (cfg['scalars'] - 2))
    scalar = scalar_loss(out[:, cfg['components']:], scalars, weights)
    boundary, axis = geometry_losses(prediction, *geom)
    gradient = field_loss(prediction, psi, r, z)
    terms = {'pixel': pixel, 'shape': cfg['shape_weight'] * shape, 'scalar': scalar,
             'boundary': 0.1 * boundary, 'axis': 0.1 * axis, 'gradient': 0.05 * gradient}
    return prediction, terms


def gradient_diagnostics(terms, model):
    tracked = (model.stem.weight, model.head[1].weight)
    vectors = {}
    for key, term in terms.items():
        grad = torch.autograd.grad(term, tracked, retain_graph=True)
        vectors[key] = torch.cat([value.detach().flatten() for value in grad])
    norms = {key: float(value.norm()) for key, value in vectors.items()}
    cosine = {a + '/' + b: float(torch.dot(vectors[a], vectors[b]) /
                               (vectors[a].norm() * vectors[b].norm()).clamp(min=1e-30))
              for i, a in enumerate(vectors) for b in list(vectors)[i + 1:]}
    return {'parameter_scope': 'stem.weight and head.1.weight', 'weighted_term_norms': norms,
            'weighted_term_cosines': cosine}


def probe(parent, representation, batch, r, z):
    ck = direct_grid(parent) if representation == 'grid' else parent
    cfg, params = ck['config'], parameters(ck, 'cpu')
    assert cfg.get('context', 0) == 0
    x, psi, scalars, *geom = batch
    x = prepare_inputs(x.numpy(), params, cfg, 'cpu')
    scalars = (scalars - params['s_mean']) / params['s_std']
    model = Network(cfg.get('network_inputs', cfg['inputs']), cfg['components'] + cfg['scalars'],
                    cfg['width'], cfg['depth'], cfg['dropout'])
    model.load_state_dict(ck['state'])
    model.train()
    torch.manual_seed(20261011)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=cfg['weight_decay'])
    with torch.no_grad():
        initial = decode(model(x), params, cfg).clone()
    records = []
    for step in range(3):
        prediction, terms = objective(model, x, psi, scalars, geom, params, cfg, parent, r, z)
        diagnostics = gradient_diagnostics(terms, model) if step == 0 else None
        value = sum(terms.values())
        assert torch.isfinite(value)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            updated = decode(model(x), params, cfg)
            change = updated - prediction.detach()
            outside = change - (change @ parent['basis']) @ parent['basis'].T
            grid = change.reshape(-1, 65, 65)
            second = grid.diff(n=2, dim=1).square().mean() + grid.diff(n=2, dim=2).square().mean()
            records.append({'step': step + 1, 'weighted_losses_before': {k: float(v) for k, v in terms.items()},
                            'total_parameter_gradient_norm_before_clip': float(unclipped),
                            'flux_step_rms': float(change.square().mean().sqrt()),
                            'flux_target_error_rms_after': float((updated - psi).square().mean().sqrt()),
                            'outside_parent_pca_energy_fraction': float(outside.square().sum() /
                                                                      change.square().sum().clamp(min=1e-30)),
                            'second_difference_energy_per_step_energy': float(second /
                                                                             change.square().mean().clamp(min=1e-30)),
                            'gradient_diagnostics': diagnostics})
    return initial, records


def main():
    torch.set_num_threads(2)
    out = Path('run_data/analysis-019/optimizer-probe.json')
    assert not out.exists(), 'Do not repeat the frozen diagnostic'
    parent_path = Path('run_data/nn-008/mlp-shape/model.pt')
    with parent_path.open('rb') as handle:
        digest = file_digest(handle, 'sha256').hexdigest()
    assert digest == 'fda8f02587063b305787728b78376c979b0e21e7d7fd9a25471671a9d3c17a95'
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    batch, r, z, receipts = training_batch()
    initial_pca, pca = probe(parent, 'pca', batch, r, z)
    initial_grid, grid = probe(parent, 'grid', batch, r, z)
    difference = float((initial_pca - initial_grid).abs().max())
    assert difference < 3e-6
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'purpose': 'CPU mechanism diagnostic; no checkpoint or submission changed',
              'frames': 96, 'training_only': True, 'locked_shots_read': False,
              'parent_sha256': digest, 'training_shots': receipts,
              'limitations': 'Three fixed-batch CPU steps; not the shuffled A100 trajectory or a score comparison',
              'initial_mapping_max_absolute_difference': difference, 'pca': pca, 'grid': grid}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('frames', 'initial_mapping_max_absolute_difference')}, indent=2))
    for step, (a, b) in enumerate(zip(pca, grid, strict=True), 1):
        print('step', step, 'grid/pca physical step RMS', b['flux_step_rms'] / a['flux_step_rms'])


if __name__ == '__main__':
    main()
