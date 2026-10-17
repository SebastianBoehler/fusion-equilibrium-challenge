"""Fixed044 CPU cost revision; original037 control source stays immutable."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from participant import correction_separable
from research import finite_cluster_cost as control


def paired_projection(root, plan, dense_project):
    settings = plan['paired_constraint_check']

    @torch.no_grad()
    def measure(model, features, parent_std, gram, bound):
        if features.device.type != 'cpu' or len(features) != plan['frames']:
            raise ValueError('Fixed CPU constraint support differs')
        initial = {k: v.clone() for k, v in model.correction.state_dict().items()}
        tick = time.monotonic()
        factor = correction_separable.interpolation_factor(model)
        factor_seconds = time.monotonic() - tick
        result = {'complete': False, 'support_frames': len(features),
                  'factor_setup_wall_seconds': factor_seconds, 'pairs': [],
                  'full_corpus_support_validated': False, 'optimizer_steps': 0}
        output = root / 'constraint-cost.json'
        with output.open('x') as handle:
            json.dump(result, handle)
        rounds = settings['warm_pairs'] + settings['timed_pairs']
        for pair in range(rounds):
            results, states, seconds = {}, {}, {}
            order = ('dense', 'separable') if pair % 2 == 0 else ('separable', 'dense')
            for name in order:
                model.correction.load_state_dict(initial)
                tick = time.monotonic()
                results[name] = (dense_project(model, features, parent_std, gram, bound)
                    if name == 'dense' else correction_separable.project_correction(
                        model, features, parent_std, factor, bound))
                seconds[name] = time.monotonic() - tick
                states[name] = {k: v.clone() for k, v in model.correction.state_dict().items()}
            differences = {k: abs(results['dense'][k] - results['separable'][k])
                           for k in ('factor', 'relative_rms_before', 'relative_rms_after')}
            if max(differences.values()) > settings['maximum_relative_RMS_or_factor_difference']:
                raise ValueError('Paired dense/separable numerical gate failed')
            if not all(torch.equal(states['dense'][k], states['separable'][k]) for k in initial):
                raise ValueError('Paired globally projected parameters are not bit equal')
            if max(results[n]['relative_rms_after'] for n in order) > bound:
                raise ValueError('Paired correction bound failed')
            result['pairs'].append({'pair': pair, 'warm': pair < settings['warm_pairs'],
                'order': list(order), 'wall_seconds': seconds,
                'differences': differences, 'projected_parameters_bit_equal': True})
            control.save(output, result)
        model.correction.load_state_dict(states['dense'])
        timed = [p for p in result['pairs'] if not p['warm']]
        medians = {n: statistics.median(p['wall_seconds'][n] for p in timed)
                   for n in ('dense', 'separable')}
        result.update(complete=True, timed_pairs=len(timed), median_wall_seconds=medians,
                      separable_over_dense_ratio=medians['separable'] / medians['dense'],
                      final_dense_control_state_restored=True)
        control.save(output, result)
        return results['dense'] | {'paired_constraint_cost': result,
            'CPU_seconds_scope': 'Complete twelve-projection paired block; per-call medians are separate'}

    return measure


def run(root):
    plan = json.loads((root / 'predeclared-plan.json').read_text())
    dense_project = control.project_correction
    # Instrument only this control's projection call; preserve its immutable code.
    control.project_correction = paired_projection(root, plan, dense_project)
    try:
        control.run(root)
    finally:
        control.project_correction = dense_project


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
