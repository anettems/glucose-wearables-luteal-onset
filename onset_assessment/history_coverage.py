#### Audit longer calendar histories using quality-approved 2022 daily measurements.
#### Count observed days, gaps and older-week support before any cohort filtering.
#### Current aligned values and future measurements never enter a history. Coverage rules
#### are descriptive sensitivities, not new model features or changes to the primary cohort.

import math

import numpy as np
import pandas as pd

from model_inputs import GLUCOSE_FEATURES, WEARABLE_FEATURES
from onset_prediction.folds import represented_events
from onset_prediction.outcomes import validate_keys
from onset_prediction.protocol import KEY, TARGET


WINDOWS = (14, 21, 28)
FRACTIONS = (.5, .7, .8, .9, 1.)
SIGNALS = ('glucose', 'wearable_two', 'combined')


def longest_gap(observed):
    #### Count the longest run of missing calendar days, including both window edges.
    longest, current = 0, 0
    for available in observed:
        current = 0 if available else current + 1
        longest = max(longest, current)
    return longest


def availability(audit):
    #### Keep individual sensor quality checks and distinguish jointly observed days.
    validate_keys(audit)
    columns = list(GLUCOSE_FEATURES) + list(WEARABLE_FEATURES)
    if np.isinf(audit[columns].to_numpy(float)).any():
        raise ValueError('Infinite measurement in coverage input')
    observed = audit[KEY].copy()
    observed['glucose'] = audit[list(GLUCOSE_FEATURES)].notna().all(axis=1)
    observed['wearable_two'] = audit[list(WEARABLE_FEATURES)].notna().sum(axis=1).ge(2)
    observed['combined'] = observed.glucose & observed.wearable_two
    if not audit.loc[observed.glucose, 'prior_day_source_day'].eq(
            audit.loc[observed.glucose, 'day_in_study'] - 1).all():
        raise ValueError('Glucose does not come from the preceding calendar day')
    return observed


def window_record(days, observed, day, length):
    #### Use exactly length prior calendar slots, retaining absent records as missing days.
    slots = np.arange(day - length, day)
    present = np.isin(slots, days[observed])
    inside = slots >= days.min()
    return {
        'observed_days': int(present.sum()), 'coverage_fraction': float(present.mean()),
        'longest_gap_days': longest_gap(present),
        'oldest_week_observed': int(present[:7].sum()),
        'extra_beyond_14_observed': int(present[:-14].sum()),
        'before_timeline_days': int((~inside).sum()),
        'missing_within_timeline_days': int((inside & ~present).sum()),
        'full_history_span': bool(inside.all()),
    }


def build_windows(audit):
    #### Calculate every history before restricting to current glucose or model eligibility.
    available = availability(audit)
    records = []
    for (person, interval), group in available.groupby(['id', 'study_interval']):
        days = group.day_in_study.to_numpy(int)
        for signal in SIGNALS:
            observed = group[signal].to_numpy(bool)
            for length in WINDOWS:
                for day in days:
                    record = {'id': person, 'study_interval': interval, 'day_in_study': day,
                              'signal': signal, 'window_days': length}
                    record.update(window_record(days, observed, day, length))
                    records.append(record)
    metadata = audit[KEY + [TARGET, 'history_eligible', 'supervised_eligible']].merge(
        available[KEY + ['glucose']], on=KEY, validate='one_to_one')
    return pd.DataFrame(records).merge(metadata, on=KEY, validate='many_to_one')


def cohort_counts(data, links):
    #### Separate positive prediction days from distinct supplied reference starts.
    return {'participants': data.id.nunique(), 'days': len(data),
            'positive_days': int(data[TARGET].eq(1).sum()),
            'negative_days': int(data[TARGET].eq(0).sum()),
            'unknown_outcomes': int(data[TARGET].isna().sum()),
            'represented_starts': len(represented_events(data, links))}


def scopes(data):
    #### Show the full timeline, glucose-available days and the unchanged shared cohort.
    return {'all_timeline': data, 'current_glucose': data.loc[data.glucose],
            'shared_cohort': data.loc[data.supervised_eligible]}


def coverage_summaries(windows, links):
    #### Compare fixed day-coverage requirements without selecting one from model performance.
    rows = []
    for (signal, length), group in windows.groupby(['signal', 'window_days']):
        for scope, baseline in scopes(group).items():
            for fraction in FRACTIONS:
                minimum = math.ceil(length * fraction)
                kept = baseline.loc[baseline.observed_days.ge(minimum)]
                row = {'signal': signal, 'window_days': length, 'scope': scope,
                       'minimum_fraction': fraction, 'minimum_days': minimum,
                       'baseline_days': len(baseline), **cohort_counts(kept, links)}
                row['retained_fraction'] = len(kept) / len(baseline) if len(baseline) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def distribution_summaries(windows):
    #### Describe boundaries, gaps and older-week coverage separately from threshold retention.
    rows = []
    for (signal, length), group in windows.groupby(['signal', 'window_days']):
        for scope, baseline in scopes(group).items():
            row = {'signal': signal, 'window_days': length, 'scope': scope,
                   'days': len(baseline), 'full_span_days': int(baseline.full_history_span.sum())}
            for column in ['observed_days', 'longest_gap_days', 'oldest_week_observed',
                           'extra_beyond_14_observed', 'missing_within_timeline_days']:
                for label, quantile in [('q25', .25), ('median', .5), ('q75', .75)]:
                    row[column + '_' + label] = baseline[column].quantile(quantile)
            row['oldest_week_at_least_four_days'] = int(baseline.oldest_week_observed.ge(4).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def participant_summaries(windows, links):
    #### Retain every participant, including those with no remaining shared-cohort days.
    rows = []
    for (signal, length, person), group in windows.groupby(['signal', 'window_days', 'id']):
        baseline = group.loc[group.supervised_eligible]
        for fraction in FRACTIONS:
            kept = baseline.loc[baseline.observed_days.ge(math.ceil(length * fraction))]
            rows.append({'id': person, 'signal': signal, 'window_days': length,
                         'minimum_fraction': fraction, 'baseline_days': len(baseline),
                         **cohort_counts(kept, links)})
    return pd.DataFrame(rows)


def fold_summaries(windows, links, test_folds, validation_folds):
    #### Audit outcome support in saved assessment and training folds without reallocating people.
    rows = []
    for (signal, length), group in windows.loc[windows.supervised_eligible].groupby(
            ['signal', 'window_days']):
        for fraction in FRACTIONS:
            kept = group.loc[group.observed_days.ge(math.ceil(length * fraction))]
            for fold in sorted(test_folds.test_fold.unique()):
                testing = test_folds.loc[test_folds.test_fold.eq(fold), 'id']
                development = test_folds.loc[test_folds.test_fold.ne(fold), 'id']
                subsets = [('test', 0, testing), ('development', 0, development)]
                assignments = validation_folds.loc[validation_folds.test_fold.eq(fold)]
                for validation in sorted(assignments.validation_fold.unique()):
                    subsets.extend([
                        ('validation', validation, assignments.loc[
                            assignments.validation_fold.eq(validation), 'id']),
                        ('training', validation, assignments.loc[
                            assignments.validation_fold.ne(validation), 'id'])])
                for stage, validation, people in subsets:
                    counts = cohort_counts(kept.loc[kept.id.isin(people)], links)
                    rows.append({'signal': signal, 'window_days': length,
                                 'minimum_fraction': fraction, 'test_fold': fold,
                                 'validation_fold': validation, 'stage': stage, **counts,
                                 'outcome_support': counts['positive_days'] > 0
                                 and counts['negative_days'] > 0 and counts['represented_starts'] > 0})
    return pd.DataFrame(rows)
