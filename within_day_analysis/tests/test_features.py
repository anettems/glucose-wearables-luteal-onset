"""Known-signal, missing-time and calendar-boundary checks for glucose patterns."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from within_day_analysis import protocol
from within_day_analysis.features import daily_patterns, pair_features, period_coverage, select_cohort, summarise_day


def glucose_frame(seconds, values, participant=1, day=1):
    return pd.DataFrame({"id": participant, "study_interval": 2022, "day_in_study": day,
                         "audit_end_seconds": day * 86400 + np.asarray(seconds),
                         "glucose_value": values, "audit_structurally_usable": True})


class GlucosePatternTests(unittest.TestCase):
    def test_quantiles_and_rate_have_known_units(self):
        seconds = np.arange(30) * 300.0
        values = 4.0 + np.arange(30) * 0.1
        result = summarise_day(seconds, values)
        self.assertAlmostEqual(result["glucose_iqr_mmol_l"], 1.45)
        self.assertAlmostEqual(result["glucose_p90_minus_median_mmol_l"], 1.16)
        self.assertAlmostEqual(result["glucose_mean_absolute_rate_mmol_l_min"], 0.02)
        self.assertAlmostEqual(result["glucose_lag5min_correlation"], 1.0)
        self.assertEqual(result["glucose_pattern_pairs"], 29)

    def test_rate_uses_actual_elapsed_time_with_export_jitter(self):
        seconds = np.arange(30) * 301.0
        result = pair_features(seconds, 5 + seconds / 60 * 0.01)
        self.assertAlmostEqual(result["glucose_mean_absolute_rate_mmol_l_min"], 0.01)
        self.assertEqual(result["glucose_pattern_pairs"], 29)

    def test_gaps_and_oversampled_pairs_are_not_bridged(self):
        seconds = np.r_[np.arange(25) * 300, 9000, 9001, 9301]
        values = np.r_[5 + np.arange(25) * 0.1, 50, 5, 5.1]
        result = pair_features(seconds, values)
        self.assertEqual(result["glucose_pattern_pairs"], 25)
        self.assertAlmostEqual(result["glucose_mean_absolute_rate_mmol_l_min"], 0.02)

    def test_dynamic_features_need_24_pairs(self):
        result = pair_features(np.arange(24) * 300, 5 + np.arange(24) * 0.1)
        self.assertTrue(np.isnan(result["glucose_mean_absolute_rate_mmol_l_min"]))
        self.assertTrue(np.isnan(result["glucose_lag5min_correlation"]))
        result = pair_features(np.arange(25) * 300, 5 + np.arange(25) * 0.1)
        self.assertAlmostEqual(result["glucose_lag5min_correlation"], 1.0)

    def test_equal_distribution_can_have_different_dynamics(self):
        seconds = np.arange(288) * 300
        smooth = summarise_day(seconds, np.r_[np.full(144, 5.0), np.full(144, 6.0)])
        alternating = summarise_day(seconds, np.tile([5.0, 6.0], 144))
        self.assertEqual(smooth["glucose_iqr_mmol_l"], alternating["glucose_iqr_mmol_l"])
        self.assertGreater(alternating["glucose_mean_absolute_rate_mmol_l_min"], smooth["glucose_mean_absolute_rate_mmol_l_min"])
        self.assertAlmostEqual(alternating["glucose_lag5min_correlation"], -1.0)
        self.assertGreater(smooth["glucose_lag5min_correlation"], 0.9)

    def test_constant_signal_has_zero_rate_and_undefined_correlation(self):
        result = summarise_day(np.arange(288) * 300, np.full(288, 5.0))
        self.assertEqual(result["glucose_mean_absolute_rate_mmol_l_min"], 0)
        self.assertEqual(result["glucose_iqr_mmol_l"], 0)
        self.assertTrue(np.isnan(result["glucose_lag5min_correlation"]))

    def test_clock_period_contrast_and_separate_coverage(self):
        seconds = np.arange(288) * 300
        values = np.where(seconds < 21600, 5.0, 6.0)
        result = summarise_day(seconds, values)
        self.assertEqual(result["glucose_night_minus_day_mmol_l"], -1)
        self.assertEqual(result["glucose_night_coverage"], 1)
        self.assertEqual(result["glucose_day_coverage"], 1)
        # Overall coverage is 75%, but the entire night is absent.
        result = summarise_day(seconds[72:], values[72:])
        self.assertTrue(np.isnan(result["glucose_night_minus_day_mmol_l"]))

    def test_coverage_counts_union_and_clips_boundaries(self):
        seconds = np.array([0, 1, 2, 21450, 86350])
        self.assertAlmostEqual(period_coverage(seconds, 0, 21600), 452 / 21600)
        self.assertAlmostEqual(period_coverage(seconds, 21600, 86400), 200 / 64800)
        self.assertEqual(period_coverage(np.array([]), 0, 21600), 0)

    def test_calendar_lag_isolated_participants_and_future_values(self):
        seconds = np.arange(288) * 300
        first = glucose_frame(seconds, 5 + np.sin(np.arange(288) / 10), day=2)
        later = glucose_frame(seconds, 6 + np.cos(np.arange(288) / 10), day=5)
        other = glucose_frame(seconds, 8 + np.sin(np.arange(288) / 2), participant=2, day=2)
        source = pd.concat([first, later, other], ignore_index=True)
        result = daily_patterns(source.sample(frac=1, random_state=42))
        self.assertEqual(result.day_in_study.tolist(), [3, 6, 3])
        expected = result.loc[result.id.eq(1) & result.day_in_study.eq(3)]
        source.loc[source.day_in_study.ge(3) | source.id.eq(2), "glucose_value"] = 1000
        actual = daily_patterns(source)
        pd.testing.assert_frame_equal(expected, actual.loc[actual.id.eq(1) & actual.day_in_study.eq(3)])

    def test_invalid_records_excluded_and_invalid_keys_rejected(self):
        source = glucose_frame(np.arange(288) * 300, 5 + np.sin(np.arange(288) / 10))
        invalid = source.iloc[[0]].assign(glucose_value=1000, audit_structurally_usable=False)
        pd.testing.assert_frame_equal(daily_patterns(source), daily_patterns(pd.concat([source, invalid])))
        for changed in [source.assign(study_interval=2024), pd.concat([source, source.iloc[[0]]])]:
            with self.assertRaises(ValueError):
                daily_patterns(changed)

    def test_complete_case_selection_and_feature_columns(self):
        keys = pd.DataFrame({"id": [1, 1], "study_interval": 2022, "day_in_study": [2, 3]})
        patterns = keys.copy()
        for feature in protocol.PATTERN_FEATURES:
            patterns[feature] = [1.0, np.nan]
        audit, cohort = select_cohort(keys.assign(luteal_status=[0, 1]), patterns)
        self.assertEqual(cohort.day_in_study.tolist(), [2])
        self.assertEqual(audit.within_day_eligible.tolist(), [True, False])
        self.assertEqual(len(protocol.feature_columns("glucose", "within_day")), 11)
        self.assertEqual(len(protocol.feature_columns("combined", "within_day")), 20)
        self.assertEqual(protocol.feature_columns("wearable", "baseline"), protocol.feature_columns("wearable", "within_day"))


if __name__ == "__main__":
    unittest.main()
