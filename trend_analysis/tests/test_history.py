"""Meaningful calendar, causality and cohort checks for personal histories."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "tests")]
from model_inputs import FEATURE_SETS, glucose_features
from test_primary_evaluation import synthetic_observations
from trend_analysis.features import add_personal_features, select_history_cohort, window_statistics


def long_observations(participants=10):
    parts = []
    for offset in range(0, 30, 6):
        block = synthetic_observations(participants)
        for column in ["day_in_study", "prior_day_source_day", "sleeping_hr_source_day"]:
            block[column] += offset
        for column in ["prediction_cutoff_seconds", "main_sleep_start_seconds", "main_sleep_end_seconds"]:
            block[column] += offset * 86400
        parts.append(block)
    return pd.concat(parts, ignore_index=True).sort_values(["id", "day_in_study"]).reset_index(drop=True)


def history_from(observations):
    return observations[["id", "study_interval", "day_in_study", *FEATURE_SETS["combined"]]].copy()


class HistoryTests(unittest.TestCase):
    def test_window_median_and_slope_use_actual_days_and_exclude_current_value(self):
        days = np.array([1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])
        values = 10.0 + 2 * days
        values[days >= 15] = 9999
        result = window_statistics(days, values, 15)
        self.assertEqual(result["reference_count"], 11)
        self.assertEqual(result["trend_count"], 6)
        self.assertEqual(result["reference"], np.median(10 + 2 * days[:11]))
        self.assertAlmostEqual(result["slope"], 2.0)
        self.assertEqual(result["last_reference_day"], 14)

    def test_gaps_do_not_extend_calendar_window_to_old_rows(self):
        days = np.array([1, 2, 3, 4, 5, 6, 7, 28])
        result = window_statistics(days, days.astype(float), 28)
        self.assertEqual(result["reference_count"], 0)
        self.assertTrue(np.isnan(result["reference"]))
        self.assertTrue(np.isnan(result["slope"]))

    def test_glucose_history_retains_calendar_lag_and_excludes_current_cgm(self):
        rows = [{"id": 1, "study_interval": 2022, "day_in_study": day,
                 "glucose_value": 2.0 * day + offset, "audit_structurally_usable": True}
                for day in range(1, 21) for offset in [0, 2]]
        glucose = pd.DataFrame(rows)
        source = history_from(long_observations(1)).drop(columns=["glucose_median_mmol_l", "glucose_cv_percent"])
        def history(frame):
            values = source.merge(glucose_features(frame), on=["id", "study_interval", "day_in_study"], how="left")
            return add_personal_features(values).set_index("day_in_study")
        expected = history(glucose)
        self.assertEqual(expected.loc[10, "glucose_median_mmol_l"], 19)
        self.assertEqual(expected.loc[10, "glucose_median_mmol_l__reference"], 10)
        self.assertAlmostEqual(expected.loc[10, "glucose_median_mmol_l__slope"], 2)
        glucose.loc[glucose.day_in_study.ge(10), "glucose_value"] = 10000
        actual = history(glucose)
        pd.testing.assert_frame_equal(expected.loc[:10], actual.loc[:10])

    def test_future_changes_cannot_change_earlier_features(self):
        source = history_from(long_observations(2))
        expected = add_personal_features(source)
        changed = source.copy()
        changed.loc[changed.day_in_study.gt(15), FEATURE_SETS["combined"]] = 10000
        actual = add_personal_features(changed)
        pd.testing.assert_frame_equal(expected.loc[expected.day_in_study.le(15)], actual.loc[actual.day_in_study.le(15)])

    def test_current_change_affects_deviation_but_not_its_own_reference_or_slope(self):
        source = history_from(long_observations(1))
        expected = add_personal_features(source).set_index("day_in_study")
        source.loc[source.day_in_study.eq(15), "sleeping_hr_bpm"] += 10
        actual = add_personal_features(source).set_index("day_in_study")
        for suffix in ["reference", "slope"]:
            self.assertEqual(actual.loc[15, f"sleeping_hr_bpm__{suffix}"], expected.loc[15, f"sleeping_hr_bpm__{suffix}"])
        self.assertAlmostEqual(actual.loc[15, "sleeping_hr_bpm__deviation"] - expected.loc[15, "sleeping_hr_bpm__deviation"], 10)

    def test_histories_never_cross_participants_or_use_missing_values(self):
        source = history_from(long_observations(2))
        source.loc[source.id.eq(1) & source.day_in_study.le(10), "sleeping_hr_bpm"] = np.nan
        expected = add_personal_features(source)
        source.loc[source.id.eq(2), FEATURE_SETS["combined"]] = -1000
        actual = add_personal_features(source)
        pd.testing.assert_frame_equal(expected.loc[expected.id.eq(1)], actual.loc[actual.id.eq(1)])
        first = actual.loc[actual.id.eq(1) & actual.day_in_study.eq(11)].iloc[0]
        self.assertEqual(first.sleeping_hr_bpm__reference_count, 0)
        self.assertTrue(np.isnan(first.sleeping_hr_bpm__deviation))

    def test_invalid_interval_and_duplicate_keys_are_rejected(self):
        source = history_from(long_observations(1))
        with self.assertRaises(ValueError):
            add_personal_features(source.assign(study_interval=2024))
        with self.assertRaises(ValueError):
            add_personal_features(pd.concat([source, source.iloc[:1]], ignore_index=True))

    def test_history_from_days_outside_prediction_cohort_is_retained(self):
        observations = long_observations(2)
        source = history_from(observations)
        # The first seven days need not have prediction labels to supply histories.
        prediction_days = observations.loc[observations.day_in_study.ge(8)]
        enriched = add_personal_features(source)
        audit, cohort = select_history_cohort(prediction_days, enriched)
        self.assertEqual(len(cohort), len(prediction_days))
        self.assertEqual(cohort.day_in_study.min(), 8)
        self.assertTrue(audit.glucose_history_ready.all())

    def test_two_ready_wearables_allow_missing_third_history(self):
        observations = long_observations(1)
        source = history_from(observations)
        source.loc[source.day_in_study.lt(30), "log_median_rmssd_ms"] = np.nan
        enriched = add_personal_features(source)
        _, cohort = select_history_cohort(observations.loc[observations.day_in_study.eq(30)], enriched)
        self.assertEqual(len(cohort), 1)
        self.assertEqual(cohort.wearable_histories_ready.iloc[0], 2)
        self.assertTrue(np.isnan(cohort.log_median_rmssd_ms__deviation.iloc[0]))
        self.assertTrue(pd.notna(cohort.log_median_rmssd_ms.iloc[0]))


if __name__ == "__main__":
    unittest.main()
