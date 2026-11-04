#### Check calendar history coverage, timing, gaps, cohort retention and event counting.
#### Synthetic examples have explicit expected counts and do not depend on observed results.

import unittest

import numpy as np
import pandas as pd

from model_inputs import GLUCOSE_FEATURES, WEARABLE_FEATURES
from onset_assessment.history_coverage import (availability, build_windows, cohort_counts,
    coverage_summaries, fold_summaries, longest_gap, window_record)
from onset_prediction.outcomes import event_links, future_outcomes, start_statuses
from onset_prediction.protocol import TARGET


def synthetic_audit():
    #### Create two people with distinct complete or entirely absent glucose histories.
    parts = []
    for person in (1, 2):
        data = pd.DataFrame({'id': person, 'study_interval': 2022,
                             'day_in_study': range(1, 36)})
        data['phase'] = np.where(data.day_in_study.between(20, 25), 'luteal', 'follicular')
        data = future_outcomes(start_statuses(data))
        for signal in list(GLUCOSE_FEATURES) + list(WEARABLE_FEATURES):
            data[signal] = 5. if person == 1 else np.nan
        data['prior_day_source_day'] = data.day_in_study - 1
        data['history_eligible'] = data.day_in_study.ge(8) & (person == 1)
        data['supervised_eligible'] = data.history_eligible & data[TARGET].notna()
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


class HistoryCoverageTests(unittest.TestCase):
    #### Keep the audit independent of row order, current values and future information.

    def test_calendar_slots_and_edges(self):
        #### Missing rows and both edge gaps count without extending the calendar window.
        days = np.array([1, 3, 4, 7, 8, 20])
        record = window_record(days, np.ones(len(days), bool), 8, 7)
        self.assertEqual(record['observed_days'], 4)
        self.assertEqual(record['longest_gap_days'], 2)
        self.assertEqual(record['before_timeline_days'], 0)
        self.assertEqual(record['missing_within_timeline_days'], 3)
        early = window_record(days, np.ones(len(days), bool), 3, 7)
        self.assertEqual(early['before_timeline_days'], 5)
        self.assertEqual(early['observed_days'], 1)
        self.assertEqual(longest_gap([False, False, True, False]), 2)
        self.assertEqual(longest_gap([False] * 7), 7)
        self.assertEqual(longest_gap([True] * 7), 0)

    def test_current_future_and_participant_separation(self):
        #### Histories exclude current/future values and cannot borrow from another person.
        data = synthetic_audit()
        windows = build_windows(data)
        selected = windows.loc[windows.day_in_study.eq(29) & windows.window_days.eq(28)
                               & windows.signal.eq('glucose')].sort_values('id')
        self.assertEqual(selected.observed_days.tolist(), [28, 0])
        changed = data.copy()
        changed.loc[changed.day_in_study.ge(29), list(GLUCOSE_FEATURES)] = np.nan
        after = build_windows(changed)
        compared = after.loc[after.day_in_study.eq(29) & after.window_days.eq(28)
                             & after.signal.eq('glucose')].sort_values('id')
        self.assertEqual(compared.observed_days.tolist(), [28, 0])

    def test_histories_before_cohort_filtering(self):
        #### Unlabelled and model-ineligible past days still supply quality-approved history.
        data = synthetic_audit()
        data.loc[data.day_in_study.lt(20), ['history_eligible', 'supervised_eligible']] = False
        data.loc[data.day_in_study.lt(20), 'phase'] = np.nan
        windows = build_windows(data)
        row = windows.loc[windows.id.eq(1) & windows.day_in_study.eq(29)
                          & windows.window_days.eq(28) & windows.signal.eq('glucose')].iloc[0]
        self.assertEqual(row.observed_days, 28)
        self.assertEqual(row.extra_beyond_14_observed, 14)
        self.assertEqual(row.oldest_week_observed, 7)

    def test_threshold_rounding_and_distinct_events(self):
        #### Seventy percent requires 15/21 or 20/28 days; three positives are one start.
        data = synthetic_audit()
        links = event_links(data)
        counts = cohort_counts(data.loc[data.id.eq(1) & data[TARGET].eq(1)], links)
        self.assertEqual(counts['positive_days'], 3)
        self.assertEqual(counts['represented_starts'], 1)
        summary = coverage_summaries(build_windows(data), links)
        chosen = summary.loc[summary.scope.eq('shared_cohort') & summary.signal.eq('glucose')
                             & summary.minimum_fraction.eq(.7)]
        self.assertEqual(chosen.minimum_days.tolist(), [10, 15, 20])
        self.assertEqual(chosen.days.tolist(), [22, 17, 12])

    def test_key_and_alignment_validation(self):
        #### Reject duplicated keys, disallowed years, infinite values and source-day errors.
        data = synthetic_audit()
        for kind in ('duplicate', 'year', 'source', 'infinite'):
            changed = data.copy()
            if kind == 'duplicate':
                changed = pd.concat([changed, changed.iloc[:1]])
            elif kind == 'year':
                changed.loc[0, 'study_interval'] = 2024
            elif kind == 'source':
                changed.loc[0, 'prior_day_source_day'] = 99
            else:
                changed.loc[0, GLUCOSE_FEATURES[0]] = np.inf
            with self.assertRaises(ValueError):
                availability(changed)

    def test_empty_fold_is_reported(self):
        #### Preserve empty assessment folds as failures instead of silently dropping them.
        data = synthetic_audit()
        tests = pd.DataFrame({'id': [1, 2], 'test_fold': [1, 2]})
        validations = pd.DataFrame({'id': [2, 1], 'test_fold': [1, 2],
                                    'validation_fold': [1, 1]})
        folds = fold_summaries(build_windows(data), event_links(data), tests, validations)
        empty = folds.loc[folds.stage.eq('test') & folds.test_fold.eq(2)]
        self.assertTrue(empty.days.eq(0).all())
        self.assertFalse(empty.outcome_support.any())


if __name__ == '__main__':
    unittest.main()
