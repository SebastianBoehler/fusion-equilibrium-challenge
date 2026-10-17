"""One fixed CPU-only Galvani TRAIN throughput and projection/export check."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from participant.correction_constraint import correction_rms, interpolation_gram, project_correction
from participant.finite_metrics import measure_derived
from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.nn_train import Network
from participant.smooth_correction import SmoothCorrection, fold_checkpoint


def sha(path):
    with Path(path).open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(root):
    plan_path = root / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    inputs = plan['CPU_only_inputs']
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(plan_path), 'optimizer_steps': 0, 'GPU_count': 0,
        'locked_or_development_read': False, 'true_fields_or_MAST_labels_read': False,
        'prediction_or_learned_artifacts_saved': False, 'fixture_only': True,
        'host': platform.node(), 'torch_version': torch.__version__, 'numpy_version': np.__version__}
    (root / 'running.lock').mkdir()
    output = root / 'cluster-cost.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        for key in ('parent', 'fixture', 'validation'):
            if sha(inputs[key]) != inputs[key + '_sha256']:
                raise ValueError(f'Pinned CPU input changed: {key}')
        contours = Path(inputs['contours'])
        if sha(contours / 'valid.npy') != inputs['valid_sha256'] or sha(contours / 'index.json') != inputs['contour_index_sha256']:
            raise ValueError('Validated TRAIN mask/index changed')
        validation = json.loads(Path(inputs['validation']).read_text())
        index = json.loads((Path(inputs['packed']) / 'index.json').read_text())
        reference_index = json.loads((contours / 'index.json').read_text())
        if (not validation['complete'] or not validation['training_only'] or
                reference_index['manifest_sha256'] != inputs['manifest_sha256'] or
                index['names'] != reference_index['names'] or index['offsets'] != reference_index['offsets']):
            raise ValueError('Packed TRAIN order/manifest differs from complete024 validation')
        valid = np.load(contours / 'valid.npy', allow_pickle=False)
        eligible = np.flatnonzero(valid)
        if len(eligible) != plan['fixed_checks']['original_target_mask_count']:
            raise ValueError('Validated TRAIN mask count changed')
        chosen = np.sort(np.random.default_rng(plan['seed']).choice(eligible, plan['frames'], replace=False))
        source = Path(inputs['packed']) / 'x.npy'
        raw = np.load(source, mmap_mode='r', allow_pickle=False)
        if raw.dtype != np.float32 or len(raw) != index['frames']:
            raise ValueError('Invalid packed permitted TRAIN inputs')
        actual = np.asarray(raw[chosen])
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite permitted CPU fixture input')
        result.update(packed_x_sha256=sha(source), packed_index_sha256=sha(Path(inputs['packed']) / 'index.json'),
            sampled_frame_indices=chosen.tolist(), sampled_frames=len(chosen),
            sampled_shots=int(len(np.unique(np.searchsorted(index['offsets'][1:], chosen, side='right')))))
        parent = torch.load(inputs['parent'], map_location='cpu', weights_only=False)
        model = SmoothCorrection(parent).eval()
        x = prepare_inputs(actual, parameters(parent, 'cpu'), parent['config'], 'cpu')
        tick = time.monotonic()
        features, bases, scalar_parts = [], [], []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                f = model.parent.head[0](model.parent.blocks(model.parent.stem(x[start:start + 512])))
                values = model.parent.head[1](f)
                features.append(f)
                bases.append(decode(values, parameters(parent, 'cpu'), model.config))
                scalar_parts.append(values[:, model.config['components']:])
        features, base, scalars = torch.cat(features), torch.cat(bases), torch.cat(scalar_parts)
        result['CPU_parent_cache_seconds'] = time.monotonic() - tick
        std = base.std(1, unbiased=False)
        if torch.any(std <= 0) or not torch.isfinite(std).all():
            raise ValueError('Undefined CPU parent field standard deviation')
        fixture = torch.load(inputs['fixture'], map_location='cpu', weights_only=False)
        model.correction.load_state_dict(fixture['state'])
        gram = interpolation_gram(model)
        with torch.no_grad():
            latent = model.correction(features)
            physical = F.interpolate(latent.reshape(-1, 1, 17, 17), size=(65, 65),
                mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
            actual_rms = ((base + physical) - base).double().square().mean(1).sqrt()
            discrepancy = float(((correction_rms(latent, gram) - actual_rms).abs() / std).max())
        if discrepancy > plan['fixed_checks']['maximum_normalized_Gram_native_RMS_discrepancy']:
            raise ValueError('Broader Gram/native discrepancy exceeds declared tolerance')
        tick = time.monotonic()
        projection = project_correction(model, features, std, gram, plan['global_projection_bound'])
        projection['CPU_seconds'] = time.monotonic() - tick
        with torch.no_grad():
            latent = model.correction(features)
            native = base + F.interpolate(latent.reshape(-1, 1, 17, 17), size=(65, 65),
                mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
        relative = float(((native - base).square().mean(1).sqrt() / std).max())
        if relative > plan['fixed_checks']['maximum_actual_native_relative_RMS']:
            raise ValueError('Broader actual native correction bound failed')
        result['constraint'] = projection | {'maximum_normalized_discrepancy': discrepancy,
                                             'actual_native_relative_rms_max': relative}
        generator = torch.Generator().manual_seed(plan['seed'])
        signs = torch.randint(0, 2, (len(x), 289), generator=generator).float() * 2 - 1
        step = plan['functional_scale_relative_parent_std'] * std / model.correction_scale
        perturbation = F.interpolate((signs * step[:, None]).reshape(-1, 1, 17, 17),
            size=(65, 65), mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
        fixtures = {'native': native.numpy(), 'plus': (native + perturbation).numpy(),
                    'minus': (native - perturbation).numpy()}
        tick = time.monotonic()
        serial = measure_derived(fixtures['native'][:2048])
        result['CPU_serial_prefix_seconds'] = time.monotonic() - tick
        result['timings'] = {}
        with ProcessPoolExecutor(max_workers=plan['cpu_workers']) as pool:
            for key, fields in fixtures.items():
                tick = time.monotonic()
                values = measure_derived(fields, pool)
                result['timings'][key] = {'seconds': time.monotonic() - tick, 'frames': len(x),
                                          'all_seven_valid': True, 'extraction_failures': 0}
                if key == 'native' and not np.array_equal(values[:2048], serial):
                    raise ValueError('Galvani serial/parallel prefix is not bit equal')
                save(output, result)
            ck = fold_checkpoint(parent, model)
            cfg = ck['config']
            folded_model = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                                   cfg['width'], cfg['depth'], cfg['dropout']).eval()
            folded_model.load_state_dict(ck['state'])
            with torch.no_grad():
                parts = [folded_model(x[start:start + 512]) for start in range(0, len(x), 512)]
                out = torch.cat(parts)
                folded = decode(out, parameters(ck, 'cpu'), cfg)
            if not torch.equal(scalars, out[:, cfg['components']:]):
                raise ValueError('Broader folded scalar outputs changed')
            maximum = float((folded - native).abs().max())
            if maximum > plan['fixed_checks']['maximum_folded_field_difference']:
                raise ValueError('Broader folded flux difference exceeds declared tolerance')
            tick = time.monotonic()
            folded_values = measure_derived(folded.numpy(), pool)
            result['timings']['folded'] = {'seconds': time.monotonic() - tick, 'frames': len(x),
                                          'all_seven_valid': True, 'extraction_failures': 0}
        for key, value in parent['state'].items():
            if not torch.equal(model.parent.state_dict()[key], value):
                raise ValueError('Frozen parent changed in numerical fixture')
        for key in ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis', 's_mean', 's_std'):
            if not torch.equal(getattr(model, key), parent[key]):
                raise ValueError('Frozen normalization/decode statistic changed')
        result['export'] = {'maximum_field_difference': maximum, 'scalars_exact': True,
                            'all_actual_derived_finite': bool(np.isfinite(folded_values).all())}
        rates = [len(x) / result['timings'][key]['seconds'] for key in ('plus', 'minus')]
        result['cost_estimate'] = {'warm_frames_per_second': rates,
            'optimistic_one_full_finite_pass_seconds': 2 * plan['full_usable_training_frames'] / max(rates),
            'slower_measured_one_full_finite_pass_seconds': 2 * plan['full_usable_training_frames'] / min(rates),
            'setup_export_scoring_reserve_seconds': plan['setup_export_full_scoring_GPU_reserve_seconds'],
            'full_TRAIN_constraint_support_validated': False, 'A100_projection_runtime_measured': False,
            'production_recipe_or_GPU_feasibility_established': False}
        result.update(complete=True, all_numerical_checks_passed=True,
            actual_CPU_seconds=time.monotonic() - started, completed_at=datetime.now(timezone.utc).isoformat(),
            limitations=plan['limitations'])
        save(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      actual_CPU_seconds=time.monotonic() - started,
                      failed_at=datetime.now(timezone.utc).isoformat())
        save(output, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
