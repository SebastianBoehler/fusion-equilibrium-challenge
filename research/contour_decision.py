"""Record full024 component and regime comparisons against the frozen eleven."""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from research.inductance_comparison import RECEIPT, sha256


ROOT = Path('run_data/analysis-024')
BASELINE = Path('run_data/analysis-017/canon-b-comparison.json')
ARMS = ('aligned-li', 'aligned-li-kappa')
FAILURES = ('lcfs_fail_frac', 'psi_nonfinite_frac', 'cons_fail_frac',
            'ref_lcfs_dropped', 'n_shots_missing')
COMPONENTS = ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency')


def differences(scores, baseline):
    if scores.keys() != baseline.keys():
        raise ValueError('Incomplete development and old regression groups')
    result = {}
    for group, score in scores.items():
        base = baseline[group]
        if score['n_shots'] != base['n_shots']:
            raise ValueError('Comparison shot count differs from incumbent')
        value = {key: score[key] - base[key] for key in COMPONENTS}
        for key in ('r2_each', 'r2_cons_each', 'cons_fail_each'):
            if score[key].keys() != base[key].keys():
                raise ValueError(f'Comparison component set differs: {key}')
            value[key] = {name: v - base[key][name] for name, v in score[key].items()}
        value['failures'] = {key: score[key] - base[key] for key in FAILURES}
        result[group] = value
    return result


def regimes(values, baseline):
    if values.keys() != baseline.keys() or values['all']['frames'] != 57315:
        raise ValueError('Incomplete frozen development input regimes')
    result = {}
    for name, value in values.items():
        if value['frames'] != baseline[name]['frames']:
            raise ValueError(f'Input regime frame count differs: {name}')
        result[name] = {'frames': value['frames'], 'flux_squared_error_ratio':
                        value['flux_squared_error'] / baseline[name]['flux_squared_error']}
    return result


def decide(arm, report, baseline, baseline_regimes):
    if not report.get('complete') or len(report['scores']) != 7:
        raise ValueError(f'Comparison remains incomplete: {arm}')
    if report['upload_dtype'] != 'float32' or report['locked_shots_read']:
        raise ValueError('Comparison dtype or locked-shot contract differs')
    controls = {}
    all_failures_zero = True
    for name, scores in report['scores'].items():
        delta = differences(scores, baseline)
        controls[name] = {
            'all_three_composites_beat_eleven': all(v['S'] > 0 for v in delta.values()),
            'component_differences_vs_eleven': delta,
            'complete_development_input_regimes_vs_eleven':
                regimes(report['development_regimes'][name], baseline_regimes)}
        for score in scores.values():
            all_failures_zero &= all(score[key] == 0 for key in FAILURES)
            all_failures_zero &= all(v == 0 for v in score['cons_fail_each'].values())
    if any(v['all_three_composites_beat_eleven'] for v in controls.values()):
        raise ValueError('Composite gate passed; manual full component review required')
    return {'complete': True, 'arm': arm, 'candidate_promoted': False,
            'prediction_dtype': 'float32', 'incumbent': 'original8+s05-e80+canonA+canonB',
            'reason': 'No control improves all three incumbent composites.',
            'report_sha256': sha256(ROOT / f'{arm}-comparison.json'),
            'incumbent_report_sha256': sha256(BASELINE),
            'incumbent_checkpoint_hashes_match_archive010': True,
            'all_extraction_missing_nonfinite_derived_failures_zero': all_failures_zero,
            'controls': controls, 'locked_shots_read': False}


def main():
    output = ROOT / 'paired-decision.json'
    if output.exists():
        raise FileExistsError('Preserve the completed paired decision')
    saved = json.loads(BASELINE.read_text())
    incumbent = json.loads(RECEIPT.read_text())
    if saved['checkpoint_sha256'] != incumbent['checkpoints']:
        raise ValueError('Incumbent hashes differ from archive010')
    baseline = saved['scores']['ensemble_plus_candidate']
    baseline_regimes = saved['development_regimes']['ensemble_plus_candidate']
    decisions = {}
    for arm in ARMS:
        report = json.loads((ROOT / f'{arm}-comparison.json').read_text())
        decisions[arm] = decide(arm, report, baseline, baseline_regimes)
    for arm, decision in decisions.items():
        path = ROOT / f'{arm}-decision.json'
        if not path.exists():
            with path.open('x') as handle:
                json.dump(decision, handle, indent=2)
                handle.write('\n')
    result = {'complete': True, 'observed_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),
              'arms': decisions, 'controls': 14, 'group_combinations': 42,
              'candidate_promoted': False, 'incumbent_preserved': True,
              'locked_shots_read': False, 'official_rank_forecast': False}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('Both024 arms fail all-three-composite promotion gates; eleven preserved.', flush=True)
    for arm, decision in decisions.items():
        print(arm, 'all failures zero:',
              decision['all_extraction_missing_nonfinite_derived_failures_zero'], flush=True)


if __name__ == '__main__':
    main()
