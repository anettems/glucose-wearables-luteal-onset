"""Tests of the 14:00 cohort, feature values and training-only replacement."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import model_inputs as inputs
import preprocessing as prep
from test_preprocessing import write_temporal_exports


def feature_row(participant=1, day=2):
    return {"id": participant, "study_interval": 2022, "day_in_study": day,
            "phase": "luteal", "luteal_status": 1, "exact_luteal_onset": 1,
            "uncertain_luteal_entry": False, "prior_day_source_day": day - 1,
            "sleeping_hr_source_day": day, "prediction_cutoff_seconds": day * 86400 + 50400,
            "main_sleep_start_seconds": (day - 1) * 86400 + 23 * 3600,
            "main_sleep_end_seconds": day * 86400 + 7 * 3600,
            "valid_wearable_count": 3, "aligned_sleeping_hr": True,
            "aligned_hrv": True, "aligned_temperature": True,
            "sleeping_hr_bpm": 60., "log_median_rmssd_ms": np.log(20.),
            "skin_temperature_c": 33., "glucose_median_mmol_l": 5.,
            "glucose_cv_percent": 20.}


class ModelInputTests(unittest.TestCase):
    def test_any_two_wearables_and_current_day_label_are_required(self):
        rows = []
        for flags in [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0)]:
            row = feature_row(day=20 + len(rows) * 3)
            row.update(zip(inputs.WEARABLE_FLAGS, map(bool, flags)))
            row.update(quality_units_accepted=True, aligned_glucose=True, aligned_main_sleep=True)
            rows.append(row)
        result = prep.apply_prediction_requirements(pd.DataFrame(rows))
        self.assertEqual(result.eligible_prediction.tolist(), [True] * 4 + [False])
        self.assertNotIn("history_usable_days", result)
        for field, value in [("aligned_glucose", False), ("aligned_main_sleep", False),
                             ("quality_units_accepted", False), ("phase", None)]:
            changed = pd.DataFrame(rows).assign(**{field: value})
            self.assertFalse(prep.apply_prediction_requirements(changed).eligible_prediction.any())

    def test_feature_values_use_previous_glucose_and_matching_sleep(self):
        eligible = pd.DataFrame([feature_row()])
        glucose = pd.DataFrame({"id": [1] * 4, "study_interval": [2022] * 4,
            "day_in_study": [1, 1, 1, 2], "glucose_value": [4., 6., 999., 100.],
            "audit_structurally_usable": [True, True, False, True]})
        intervals = pd.DataFrame({"id": [1] * 3, "study_interval": [2022] * 3,
            "day_in_study": [1, 2, 2], "matched_sleep_end_day": [2, 2, 2],
            "rmssd": [10., 30., 999.], "usable_in_main_sleep": [True, True, False]})
        temp = eligible[prep.KEY].assign(nightly_temperature=33., review_selected=True)
        result = inputs.construct_features(eligible, glucose, intervals, temp)
        self.assertEqual(result.glucose_median_mmol_l.iloc[0], 5.)
        self.assertAlmostEqual(result.glucose_cv_percent.iloc[0], 100 * np.sqrt(2) / 5)
        self.assertAlmostEqual(result.log_median_rmssd_ms.iloc[0], np.log(20))
        self.assertEqual(result.skin_temperature_c.iloc[0], 33.)
        # Source values that fail a quality flag must not silently enter a model.
        eligible["aligned_hrv"] = False
        eligible["valid_wearable_count"] = 2
        result = inputs.construct_features(eligible, glucose, intervals, temp)
        self.assertTrue(result.log_median_rmssd_ms.isna().all())
        self.assertTrue(result.log_median_rmssd_ms_missing.all())

    def test_calendar_lag_does_not_bridge_missing_days_or_participants(self):
        glucose = pd.DataFrame({"id": [1, 2], "study_interval": [2022, 2022],
            "day_in_study": [1, 3], "glucose_value": [5., 9.],
            "audit_structurally_usable": [True, True]})
        summaries = inputs.glucose_features(glucose)
        target = pd.DataFrame({"id": [1, 2], "study_interval": [2022, 2022], "day_in_study": [4, 2]})
        result = prep.join_daily(target, summaries)
        self.assertTrue(result.glucose_median_mmol_l.isna().all())

    def test_replacement_uses_only_training_values_and_preserves_inputs(self):
        training = pd.DataFrame([feature_row(1), feature_row(2)])
        training.loc[1, "sleeping_hr_bpm"] = 80.
        evaluation = pd.DataFrame([feature_row(3)])
        evaluation["sleeping_hr_bpm"] = np.nan
        evaluation["aligned_sleeping_hr"] = False
        evaluation["valid_wearable_count"] = 2
        evaluation["sleeping_hr_bpm_missing"] = True
        evaluation["log_median_rmssd_ms"] = 1000.
        before = training.copy(deep=True)
        train, test, medians = inputs.replace_missing_wearables(training, evaluation)
        pd.testing.assert_frame_equal(before, training)
        self.assertTrue(evaluation.sleeping_hr_bpm.isna().all())
        self.assertEqual(test.sleeping_hr_bpm.iloc[0], 70.)
        self.assertEqual(medians.log_median_rmssd_ms, np.log(20.))
        self.assertEqual(test.log_median_rmssd_ms.iloc[0], 1000.)
        self.assertTrue(test.sleeping_hr_bpm_missing.iloc[0])
        pd.testing.assert_frame_equal(train, training)
        self.assertEqual([len(v) for v in inputs.FEATURE_SETS.values()], [2, 3, 5])

    def test_replacement_rejects_participant_overlap_and_empty_training_feature(self):
        training = pd.DataFrame([feature_row(1)])
        with self.assertRaisesRegex(ValueError, "participants must be separate"):
            inputs.replace_missing_wearables(training, training.copy())
        training["sleeping_hr_bpm"] = np.nan
        training["aligned_sleeping_hr"] = False
        training["valid_wearable_count"] = 2
        with self.assertRaisesRegex(ValueError, "no observed training values"):
            inputs.replace_missing_wearables(training, pd.DataFrame([feature_row(2)]))

    def test_invalid_outputs_are_rejected(self):
        row = pd.DataFrame([feature_row()])
        for changed in [pd.concat([row, row]), row.assign(study_interval=2024),
                        row.assign(glucose_cv_percent=np.inf),
                        row.assign(glucose_median_mmol_l=np.nan),
                        row.assign(prior_day_source_day=0),
                        row.assign(main_sleep_end_seconds=999999),
                        row.assign(aligned_hrv=False)]:
            with self.assertRaises(ValueError):
                inputs.validate_features(changed)

    def test_main_creates_reproducible_paired_datasets_without_global_imputation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            raw = root / "raw"
            write_temporal_exports(raw)
            # Remove HRV for day 4. The two remaining wearables must retain that onset.
            path = raw / "heart_rate_variability_details.csv"
            hrv = pd.read_csv(path)
            hrv.loc[~hrv.day_in_study.eq(4)].to_csv(path, index=False)
            hashes = {p.name: prep.file_sha256(p) for p in raw.glob("*.csv")}
            for run in [1, 2]:
                processed, report = root / f"processed{run}", root / f"report{run}"
                with contextlib.redirect_stdout(io.StringIO()):
                    main.main(["--raw-dir", str(raw), "--processed-dir", str(processed), "--report-dir", str(report)])
                data = pd.read_csv(processed / "observations_2022.csv")
                self.assertEqual(len(data), 15)
                self.assertEqual(int(data.exact_luteal_onset.eq(1).sum()), 3)
                self.assertEqual(int(data.log_median_rmssd_ms.isna().sum()), 3)
                for name, features in inputs.FEATURE_SETS.items():
                    frame = pd.read_csv(processed / f"{name}_features_2022.csv")
                    pd.testing.assert_frame_equal(frame[prep.KEY], data[prep.KEY])
                    self.assertEqual(frame.columns.tolist(), prep.KEY + features)
                config = json.loads((report / "run_config.json").read_text())
                self.assertTrue(config["features_ready"])
                self.assertFalse(config["imputation_fitted"])
                self.assertFalse(config["participant_splits_assigned"])
                self.assertEqual(config["prediction_cutoff"], "14:00:00")
                if run == 2:
                    for filename in ["observations_2022.csv", "glucose_features_2022.csv",
                                     "wearable_features_2022.csv", "combined_features_2022.csv", "outcomes_2022.csv"]:
                        self.assertEqual((root / "processed1" / filename).read_bytes(), (processed / filename).read_bytes())
            for p in raw.glob("*.csv"):
                self.assertEqual(prep.file_sha256(p), hashes[p.name])


if __name__ == "__main__":
    unittest.main()
