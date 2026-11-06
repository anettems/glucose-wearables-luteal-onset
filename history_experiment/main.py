#### Run the separate 21-day glucose experiment using the verified 2022 preparation.
#### Freeze the paired cohort, feature definition, fold checks and settings before fitting.
#### Save held-out probabilities, validation and training records, and a comparison report.

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from history_experiment.analysis import (VARIANTS, bootstrap, fit_folds, intervals,
    pair_predictions, paired_metrics)
from history_experiment.features import FEATURE, MIN_HISTORY, MIN_WEEK, build_history
from onset_assessment.history_main import validate_inputs
from onset_assessment.main import load_inputs
from onset_prediction import protocol
from onset_prediction.evaluation import check_shared_replacements
from onset_prediction.features import validate_cohort
from onset_prediction.folds import audit_folds, represented_events
from onset_prediction.freezing import read_json, verify_hashes, versions
from onset_prediction.logistic import create_classifier
from onset_prediction.metrics import calibration_bins, participant_draws
from onset_prediction.workflow import check_destinations, write_json
from preprocessing import file_sha256


def hashes(paths):
    #### Save absolute file names and content hashes for reproducibility checks.
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths)}


def prepare():
    #### Verify the existing outcomes and fold records, then build the shared experiment cohort.
    preparation = protocol.ROOT.parent / 'results/onset_prediction_2022_v1'
    (audit, observations, links), inputs = load_inputs(
        protocol.ROOT / 'data/processed/onset_prediction_2022_v1', preparation)
    validate_inputs(audit, observations, links)
    validate_cohort(observations)
    paths = [preparation / name for name in ('participant_folds.csv',
             'validation_fold_assignments.csv', 'evaluation_protocol.json')]
    if read_json(paths[2]) != protocol.record():
        raise ValueError('Expected the agreed three-day outcome preparation')
    test, validation = [pd.read_csv(path) for path in paths[:2]]
    inputs.update(hashes(paths + [protocol.ROOT.parent / 'EVALUATION_PROTOCOL.md']))
    history = build_history(audit)
    candidates = observations.merge(history, on=protocol.KEY, validate='one_to_one')
    data = candidates.loc[candidates[FEATURE].notna()].copy()
    test = test.loc[test.id.isin(data.id)]
    validation = validation.loc[validation.id.isin(data.id)]
    folds, observed = audit_folds(data, links, test, validation)
    if not folds.feasible.all():
        raise ValueError('At least one saved fold or training complement is infeasible')
    return data, test, validation, history, folds, observed, candidates, links, inputs


def cohort_table(candidates, data, links):
    #### Count participants, days and distinct starts at each experiment eligibility step.
    rows = []
    for stage, subset in [('original_shared', candidates), ('15_history_days',
            candidates.loc[candidates.history_days.ge(MIN_HISTORY)]), ('paired_experiment', data)]:
        rows.append({'stage': stage, 'participants': subset.id.nunique(), 'days': len(subset),
                     'positive_days': int(subset[protocol.TARGET].sum()),
                     'negative_days': int(subset[protocol.TARGET].eq(0).sum()),
                     'represented_starts': len(represented_events(subset, links))})
    return pd.DataFrame(rows)


def freeze(results, processed, prepared):
    #### Save and verify the exact inputs, assignments and analysis decisions before fitting.
    data, test, validation, history, folds, observed, candidates, links, inputs = prepared
    draws = participant_draws(data.id.unique())
    tables = {'observations': data, 'history_audit': history, 'participant_folds': test,
              'validation_folds': validation, 'bootstrap_participants': draws}
    for name, table in tables.items():
        table.to_csv(processed / f'{name}.csv', index=False)
    cohort = cohort_table(candidates, data, links)
    cohort.to_csv(results / 'cohort.csv', index=False)
    folds.to_csv(results / 'fold_feasibility.csv', index=False)
    observed.to_csv(results / 'existing_predictor_availability.csv', index=False)
    sources = list(Path(__file__).parent.rglob('*.py')) + list((protocol.ROOT / 'onset_prediction').glob('*.py'))
    sources += list((protocol.ROOT / 'onset_assessment').glob('*.py'))
    sources += list(protocol.ROOT.glob('*.py')) + [Path(__file__).with_name('README.md')]
    record = {'version': '2026-11-06-glucose-21d-v1', 'base_protocol': protocol.VERSION,
              'variants': {name: protocol.feature_columns('combined') + ([FEATURE] if index else [])
                           for index, name in enumerate(VARIANTS)},
              'recent_aligned_offsets': [-7, -1], 'oldest_aligned_offsets': [-21, -15],
              'feature': 'median of recent daily glucose medians minus median of oldest daily medians',
              'minimum_history_days': MIN_HISTORY, 'minimum_days_per_end_week': MIN_WEEK,
              'glucose_calendar_offset': -1, 'prediction_cutoff': '14:00',
              'logistic_settings': protocol.record()['models']['logistic_regression'],
              'estimator': create_classifier(1).get_params(),
              'selection': 'mean validation AP, exact ties favour smaller C',
              'primary_metric': 'paired difference in pooled test average precision',
              'test_seed': protocol.SEED, 'validation_seeds': protocol.record()['validation_seeds'],
              'bootstrap_seed': protocol.SEED + 1, 'bootstrap_resamples': len(draws),
              'versions': versions(), 'inputs': inputs, 'sources': hashes(sources),
              'prepared_files': hashes(list(processed.iterdir()) + list(results.iterdir()))}
    write_json(results / 'freeze.json', record)
    saved = read_json(results / 'freeze.json')
    for name in ('inputs', 'sources', 'prepared_files'):
        verify_hashes(saved[name])
    return draws, cohort


def report(results, cohort, estimates, fold_metrics):
    #### Explain the paired comparison and report the uncertainty conditional on these fits.
    sample = cohort.iloc[-1]
    difference = estimates.loc[estimates.variant.eq('difference')
                               & estimates.metric.eq('average_precision')].iloc[0]
    wins = int(fold_metrics.loc[fold_metrics.variant.eq('difference'), 'average_precision'].gt(0).sum())
    lines = ['# 21-day glucose-history experiment', '',
        f'The matched cohort contains {sample.days:,} prediction days from {sample.participants} '
        f'participants, including {sample.positive_days} positive days representing '
        f'{sample.represented_starts} supplied reference luteal starts. The positive-day '
        f'proportion is {sample.positive_days / sample.days:.1%}.', '',
        'Both logistic models use the same days and participant folds. The added predictor is the '
        'difference between typical daily glucose in the most recent and oldest weeks of a 21-day '
        'history. Each of these two weeks needs at least four observed days, and the full window '
        'needs at least 15. The current aligned glucose measurement is separate from both weeks.', '',
        '| Variant | Metric | Estimate | 95% interval |', '| --- | --- | ---: | ---: |']
    for row in estimates.itertuples(index=False):
        lines.append(f'| {row.variant} | {row.metric} | {row.estimate:.4f} | '
                     f'[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |')
    lines += ['', f'The main comparison is the change in pooled average precision '
              f'({difference.estimate:+.4f}). Average precision increased in {wins} of the five test folds.',
              'Intervals use 2,000 paired resamples of participants and are conditional on the fitted '
              'cross-validation models. Positive days can refer to the same onset. The selected days '
              'have relatively dense glucose records, which limits conclusions about days with poorer coverage.']
    (results / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(results, processed):
    #### Execute the frozen comparison and save one probability per variant and held-out day.
    check_destinations(processed, results)
    prepared = prepare()
    data, test, validation = prepared[:3]
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    draws, cohort = freeze(results, processed, prepared)
    outputs = fit_folds(data, test, validation)
    for name in ('fitted_parameters', 'validation_parameters'):
        check_shared_replacements(outputs[name])
    paired = pair_predictions(outputs['predictions'], data, test)
    replicates = bootstrap(paired, draws)
    estimates = intervals(paired, replicates)
    folds = pd.DataFrame([{'test_fold': fold, 'variant': variant, **scores}
                          for fold, group in paired.groupby('test_fold')
                          for variant, scores in paired_metrics(group).items()])
    calibration = pd.concat([calibration_bins(group).assign(variant=variant)
                             for variant, group in outputs['predictions'].groupby('variant')])
    outputs.update(paired_predictions=paired, bootstrap_metrics=replicates, pooled_metrics=estimates,
                   fold_metrics=folds, calibration_bins=calibration)
    for name, table in outputs.items():
        table.to_csv(results / f'{name}.csv', index=False)
    report(results, cohort, estimates, folds)
    write_json(results / 'manifest.json', {'status': 'complete', 'files': hashes(list(results.iterdir()))})
    print(cohort.to_string(index=False))
    print(estimates.to_string(index=False))


def main():
    #### Accept fresh destinations so earlier analyses and runs are protected.
    parser = argparse.ArgumentParser(description='Compare a 21-day glucose feature with the baseline')
    parser.add_argument('--results-dir', type=Path,
                        default=protocol.ROOT.parent / 'results/glucose_history_experiment_2022_v1')
    parser.add_argument('--processed-output-dir', type=Path,
                        default=protocol.ROOT / 'data/processed/glucose_history_experiment_2022_v1')
    args = parser.parse_args()
    run(args.results_dir.resolve(), args.processed_output_dir.resolve())


if __name__ == '__main__':
    main()
