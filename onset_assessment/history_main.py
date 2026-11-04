#### Run the 14-, 21- and 28-day history coverage audit from verified 2022 prepared data.
#### Save per-day histories under processed data and descriptive tables with provenance under
#### results. Recheck outcomes and the original 14-day counts; fit no models or transformations.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from onset_assessment.history_coverage import (FRACTIONS, WINDOWS, availability, build_windows,
    coverage_summaries, distribution_summaries, fold_summaries, participant_summaries)
from onset_assessment.main import check_destination, load_inputs, write_json
from onset_prediction.freezing import verify_hashes
from onset_prediction.outcomes import event_links, future_outcomes, start_statuses
from onset_prediction.protocol import KEY, ROOT, TARGET
from preprocessing import file_sha256


def validate_inputs(audit, observations, links):
    #### Reconstruct outcomes from all labels and verify the saved shared-cohort keys and links.
    rebuilt = future_outcomes(start_statuses(audit))
    actual = audit.sort_values(KEY)[TARGET].to_numpy(float)
    expected = rebuilt.sort_values(KEY)[TARGET].to_numpy(float, na_value=np.nan)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    pd.testing.assert_frame_equal(
        audit.loc[audit.supervised_eligible, KEY].sort_values(KEY).reset_index(drop=True),
        observations[KEY].sort_values(KEY).reset_index(drop=True))
    pd.testing.assert_frame_equal(event_links(rebuilt), links.sort_values(
        KEY + ['reference_day']).reset_index(drop=True), check_dtype=False)
    available = availability(audit)
    if not audit.glucose_median_mmol_l.notna().equals(audit.glucose_cv_percent.notna()):
        raise ValueError('Glucose summary availability differs; audit signals separately')
    if not audit.loc[available.wearable_two, 'main_sleep_end_seconds'].le(
            audit.loc[available.wearable_two, 'prediction_cutoff_seconds']).all():
        raise ValueError('Wearable data extend beyond the prediction cutoff')


def check_reference_counts(windows, audit):
    #### Confirm the calendar audit exactly reproduces the existing 14-day glucose counts.
    selected = windows.loc[windows.signal.eq('glucose') & windows.window_days.eq(14)]
    joined = selected.merge(audit[KEY + ['glucose_median_mmol_l__reference_count']],
                            on=KEY, validate='one_to_one')
    if not joined.observed_days.eq(joined.glucose_median_mmol_l__reference_count).all():
        raise ValueError('Fourteen-day coverage differs from existing personal histories')


def write_report(destination, audit, summary, distributions, people, folds):
    #### Explain retention and gap sensitivities without implying predictive performance.
    shared = summary.loc[summary.scope.eq('shared_cohort') & summary.signal.eq('glucose')]
    lines = ['# Longer glucose-history coverage audit', '',
        f'The full timeline contains {len(audit):,} days from {audit.id.nunique()} participants. '
        f'The unchanged shared cohort contains {int(audit.supervised_eligible.sum()):,} days.', '',
        '## Definitions', '',
        'At prediction day t, a length-L history uses aligned days t-L through t-1. '
        'For glucose these represent source calendar days t-L-1 through t-2. The current '
        'aligned glucose value from t-1 is separate and excluded from history. Wearable '
        'values on each aligned day come from sleep ending by 14:00. Missing calendar days '
        'count as missing slots; no interpolation or replacement is used.',
        'The existing daily glucose quality rule is retained (at least 70% within-day coverage '
        'and no gap longer than four hours). The percentages below instead describe the '
        'number of quality-approved daily summaries across a multi-day history.',
        'Histories use all quality-approved measurements before outcome or paired-cohort '
        'filtering. Shared-cohort results retain the existing measurement/history eligibility '
        'and known three-day outcome. Current-glucose and full-timeline results are also saved. '
        'There is no retrospective pre-luteal or complete-cycle restriction.', '',
        '## Glucose history retained within the shared cohort', '',
        '| History | Minimum observed days | Days retained | Participants | Positive days | Distinct starts |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in shared.itertuples():
        lines.append(f'| {row.window_days} days | {row.minimum_days} ({row.minimum_fraction:.0%}) | '
                     f'{row.days:,} ({row.retained_fraction:.1%}) | {row.participants} | '
                     f'{row.positive_days} | {row.represented_starts} |')
    lines += ['', 'Coverage requirements are descriptive sensitivities, not chosen model settings. '
              'Distinct starts require at least one retained positive day and are not independent '
              'counts of every positive window.', '', '## Gaps and recording boundaries', '',
              '| History | Observed days, median [IQR] | Longest gap, median [IQR] | '
              'Full elapsed history span | Oldest week has at least 4 days |',
              '| --- | ---: | ---: | ---: | ---: |']
    chosen = distributions.loc[distributions.scope.eq('shared_cohort')
                               & distributions.signal.eq('glucose')]
    for row in chosen.itertuples():
        lines.append(f'| {row.window_days} days | {row.observed_days_median:g} '
                     f'[{row.observed_days_q25:g}, {row.observed_days_q75:g}] | '
                     f'{row.longest_gap_days_median:g} [{row.longest_gap_days_q25:g}, '
                     f'{row.longest_gap_days_q75:g}] | {row.full_span_days} / {row.days} | '
                     f'{row.oldest_week_at_least_four_days} / {row.days} |')
    lines += ['', 'IQR is the middle half of days. Longest gaps include leading and trailing '
              'missing slots. Per-day exports distinguish slots before the available timeline '
              'from missing measurements within it. A full elapsed span does not imply observed glucose.',
              '', '## Participant and fold support at 70% history coverage', '']
    for length in (21, 28):
        retained = people.loc[people.signal.eq('glucose') & people.window_days.eq(length)
                              & people.minimum_fraction.eq(.7) & people.baseline_days.gt(0)]
        counts = folds.loc[folds.signal.eq('glucose') & folds.window_days.eq(length)
                           & folds.minimum_fraction.eq(.7)]
        assessment = counts.loc[counts.stage.isin(['test', 'validation'])]
        lines.append(f'- {length} days: {int(retained.days.ge(20).sum())} participants retain '
                     f'at least 20 days; {int(retained.represented_starts.ge(2).sum())} retain '
                     f'at least two distinct starts. Across saved test and validation assessments, '
                     f'the smallest positive-day count is {assessment.positive_days.min()} and '
                     f'the smallest start count is {assessment.represented_starts.min()}. '
                     f'{int((~counts.outcome_support).sum())} of {len(counts)} assessment/training '
                     'subsets lack both classes or a represented start.')
    lines += ['', 'All threshold sensitivities and all saved test, development, validation and '
              'training folds are exported. Folds are not reallocated. These are outcome-count '
              'checks only; preprocessing feasibility for hypothetical new features is not established.',
              '', '## Interpretation and limits', '',
              'Coverage can establish whether longer histories are available, not whether they '
              'predict the next cycle. Windows overlap heavily, so thousands of windows are not '
              'thousands of independent trajectories. A 21- or 28-day history is not necessarily '
              'a complete menstrual cycle. Sparse older weeks can leave a nominally long history '
              'dominated by recent measurements.',
              'The wearable and combined tables count days with at least two available wearable '
              'signals; the two signals may differ between days. They do not guarantee complete '
              'sequences for every wearable channel. Raw-value units and quality assumptions are '
              'inherited from the verified preparation. Causes of gaps, sensor changes, food and '
              'activity are not determined by this audit.',
              'No model is fitted, no new predictors are constructed, and primary feature windows, '
              'outcomes and folds are unchanged. A later modelling proposal should specify its '
              'missing-history handling and repeat these cohort checks before fitting.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(results, processed_output):
    #### Verify historical inputs, calculate coverage and save reproducible fresh outputs.
    check_destination(results, ROOT.parent / 'results')
    check_destination(processed_output, ROOT / 'data/processed')
    preparation = ROOT.parent / 'results/onset_prediction_2022_v1'
    (audit, observations, links), inputs = load_inputs(
        ROOT / 'data/processed/onset_prediction_2022_v1', preparation)
    validate_inputs(audit, observations, links)
    paths = [preparation / 'participant_folds.csv', preparation / 'validation_fold_assignments.csv',
             ROOT.parent / 'EVALUATION_PROTOCOL.md']
    inputs.update({str(path.resolve()): file_sha256(path) for path in paths})
    tests, validations = [pd.read_csv(path) for path in paths[:2]]
    sources = list(Path(__file__).parent.glob('history*.py'))
    sources += [Path(__file__).parent / 'tests/test_history_coverage.py',
                Path(__file__).parent / 'HISTORY_COVERAGE.md']
    sources += list((ROOT / 'onset_prediction').glob('*.py'))
    sources += [Path(__file__).with_name('main.py'), ROOT / 'model_inputs.py', ROOT / 'preprocessing.py']
    results.mkdir(parents=True, exist_ok=True)
    processed_output.mkdir(parents=True, exist_ok=True)
    write_json(results / 'audit_protocol.json', {
        'version': '2026-11-04-history-coverage-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'sources': {str(path): file_sha256(path) for path in sources},
        'windows': WINDOWS, 'coverage_fractions': FRACTIONS,
        'window': 'aligned t-L <= s < t; glucose source days t-L-1 through t-2',
        'missingness': 'no interpolation or imputation; absent calendar slots remain missing',
        'models_fitted': False, 'randomness': None,
        'versions': {'python': platform.python_version(), **{
            name: version(name) for name in ['numpy', 'pandas', 'scikit-learn']}}})
    windows = build_windows(audit)
    check_reference_counts(windows, audit)
    summary = coverage_summaries(windows, links)
    distributions = distribution_summaries(windows)
    people = participant_summaries(windows, links)
    folds = fold_summaries(windows, links, tests, validations)
    windows.to_csv(processed_output / 'history_windows.csv', index=False)
    for name, table in [('coverage_summary', summary), ('gap_summary', distributions),
                        ('participant_coverage', people), ('fold_coverage', folds)]:
        table.to_csv(results / (name + '.csv'), index=False)
    write_report(results, audit, summary, distributions, people, folds)
    verify_hashes(inputs)
    files = [path for folder in [results, processed_output] for path in folder.iterdir() if path.is_file()]
    write_json(results / 'manifest.json', {'status': 'complete',
               'files': {str(path): file_sha256(path) for path in files}})
    print(summary.loc[summary.scope.eq('shared_cohort') & summary.signal.eq('glucose')].to_string(index=False))


def main():
    #### Resolve fresh output locations for a descriptive audit with no model changes.
    parser = argparse.ArgumentParser(description='Audit 21-28-day history coverage')
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT.parent / 'results/history_coverage_2022_v1')
    parser.add_argument('--processed-output-dir', type=Path,
                        default=ROOT / 'data/processed/history_coverage_2022_v1')
    args = parser.parse_args()
    run(args.results_dir.resolve(), args.processed_output_dir.resolve())


if __name__ == '__main__':
    main()
