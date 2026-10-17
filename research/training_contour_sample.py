"""Load the existing three complete pinned training shots and scorer targets."""

import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score
from participant.contour_normal_loss import physical_points, targets
from participant.contour_training import ScoredContourBatch
from participant.data import read_shot
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.rich import INPUT_COLUMNS, rich_features
from research.inductance_contour_audit import MODELS, digest


def load():
    old = Path('run_data/analysis-020/packing-check')
    manifest_path = old / 'training-only-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((old / 'train/index.json').read_text())
    assert index['names'] == [Path(e['path']).stem for e in manifest['train']]
    rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for e in manifest['train']]
    raw = np.concatenate([rich_features(row) for row in rows])
    fields = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    assert len(fields) == len(raw) == index['frames'] == 508
    root = Path('run_data/analysis-023/scored-targets')
    validation = json.loads(Path('run_data/analysis-023/target-validation.json').read_text())
    for name, sha in validation['file_sha256'].items():
        assert digest(root / name) == sha
    batch = ScoredContourBatch(root, index)
    assert len(batch.valid_indices) == 508
    p, w, interior, li = batch.get(batch.valid_indices, 'cpu')
    r, z = (torch.from_numpy(v) for v in (batch.r, batch.z))
    truth = torch.from_numpy(fields)
    normal = targets(truth, r, z, p, physical_points(p, r, z), interior)
    path, sha = MODELS['parent']
    assert digest(path) == sha
    parent = torch.load(path, map_location='cpu', weights_only=False)
    inputs = prepare_inputs(raw, parameters(parent, 'cpu'), parent['config'], 'cpu')
    reference = fast_score.official.build_reference(fields.astype(float), *fast_score._geometry())
    return {'parent': parent, 'inputs': inputs, 'truth': truth, 'normal': normal,
            'geometry': (p, w, interior), 'li': li, 'r': r, 'z': z, 'reference': reference,
            'manifest_sha256': digest(manifest_path), 'target_manifest_sha256': batch.manifest_sha}
