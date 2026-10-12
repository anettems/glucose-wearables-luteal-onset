"""Focused checks for the descriptive 2022 audit; run with unittest discover."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "preprocessing.py"
SPEC = importlib.util.spec_from_file_location("preprocessing", MODULE_PATH)
preprocessing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocessing)


def write_synthetic_exports(raw_dir):
    raw_dir.mkdir()
    for name, spec in preprocessing.TABLE_SPECS.items():
        records = []
        for day, interval in [(1, 2022), (3, 2022), (1, 2024)]:
            record = {"id": 1, "study_interval": interval, "day_in_study": day}
            record.update({column: 1.0 for column in spec["values"]})
            if name == "hormones_and_selfreport":
                record["phase"] = "Follicular" if day == 1 else "Luteal"
            if "timestamp" in spec:
                record["timestamp"] = "06:00:00"
            if spec.get("session"):
                record.pop("day_in_study")
                record.update({"sleep_start_day_in_study": day,
                               "sleep_end_day_in_study": day,
                               "sleep_start_timestamp": "00:00:00",
                               "sleep_end_timestamp": "07:00:00"})
            if name == "computed_temperature":
                record["type"] = "SKIN"
            if name == "sleep":
                record.update({"mainsleep": True, "duration": 25200000,
                               "minutesasleep": 420, "timeinbed": 420})
            records.append(record)
        pd.DataFrame(records).to_csv(raw_dir / f"{name}.csv", index=False)


class PreprocessingTests(unittest.TestCase):
    def test_filter_precedes_key_validation_and_preserves_raw(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            path = Path(directory) / "glucose.csv"
            path.write_text("id,study_interval,day_in_study,timestamp,glucose_value\n"
                            "1,2022,1,00:00:00,5\n"
                            "bad,2024,bad,invalid,100\n", encoding="utf-8")
            before = path.read_bytes()
            data, excluded = preprocessing.load_2022(path, preprocessing.TABLE_SPECS["glucose"])
            self.assertEqual(data["id"].tolist(), [1])
            self.assertEqual(excluded, 1)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_keys_and_missing_columns_fail(self):
        data = pd.DataFrame({"id": [1.5]})
        with self.assertRaises(ValueError):
            preprocessing.validate_integer_column(data, "id")
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            path = Path(directory) / "glucose.csv"
            path.write_text("id,study_interval\n1,2022\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                preprocessing.load_2022(path, preprocessing.TABLE_SPECS["glucose"])

    def test_duplicate_conflicts_are_not_averaged(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            path = Path(directory) / "glucose.csv"
            path.write_text("id,study_interval,day_in_study,timestamp,glucose_value\n"
                            "1,2022,1,00:00:00,5\n1,2022,1,00:00:00,5\n"
                            "1,2022,1,00:05:00,6\n1,2022,1,00:05:00,7\n"
                            "1,2022,1,00:10:00,-1\n", encoding="utf-8")
            data, summary, _ = preprocessing.audit_table(
                Path(directory), "glucose", preprocessing.TABLE_SPECS["glucose"])
            self.assertEqual(summary["exact_duplicates_removed"], 1)
            self.assertEqual(summary["conflicting_key_rows"], 2)
            self.assertEqual(data["audit_structurally_usable"].sum(), 1)
            self.assertEqual(data["glucose_value"].tolist(), [5, 6, 7, -1])

    def test_clock_validation_and_hrv_interval_end(self):
        seconds = preprocessing.clock_seconds(pd.Series([
            "23:59:59.5", "24:00:00", "00:60:00", "2022-01-01", None]))
        self.assertEqual(seconds.iloc[0], 86399.5)
        self.assertTrue(seconds.iloc[1:].isna().all())
        data = pd.DataFrame({"day_in_study": [1], "timestamp": ["23:58:00"]})
        timed = preprocessing.add_timing(data, preprocessing.TABLE_SPECS["heart_rate_variability_details"])
        self.assertEqual(timed["audit_end_seconds"].iat[0], 2 * 86400 + 180)

    def test_sessions_are_assigned_to_end_day_and_invalid_order_flagged(self):
        data = pd.DataFrame({"sleep_start_day_in_study": [1, 2],
                             "sleep_end_day_in_study": [2, 2],
                             "sleep_start_timestamp": ["23:00:00", "08:00:00"],
                             "sleep_end_timestamp": ["07:00:00", "07:00:00"]})
        timed = preprocessing.add_timing(data, preprocessing.TABLE_SPECS["sleep"])
        self.assertEqual(timed["audit_invalid_time"].tolist(), [False, True])
        self.assertEqual(timed["audit_end_seconds"].iat[0] - timed["audit_start_seconds"].iat[0], 8 * 3600)

    def test_calendar_gaps_and_boundary_onsets_remain_unknown(self):
        labels = pd.DataFrame({"id": [1] * 5, "study_interval": [2022] * 5,
                               "day_in_study": [1, 2, 4, 5, 6],
                               "phase": [" Luteal ", "Follicular", "Luteal", "Follicular", "Luteal"],
                               "audit_conflicting_key": [False] * 5})
        grid = preprocessing.create_daily_grid({"labels": labels})
        daily = preprocessing.add_labels(grid, labels)
        self.assertEqual(daily["day_in_study"].tolist(), list(range(1, 7)))
        self.assertTrue(pd.isna(daily.loc[2, "phase"]))
        self.assertEqual(daily["exact_luteal_onset"].fillna(-1).tolist(), [-1, 0, -1, -1, 0, 1])
        self.assertEqual(daily["uncertain_luteal_entry"].tolist(), [True, False, False, True, False, False])

    def test_conflicting_and_unknown_labels_have_no_outcome(self):
        labels = pd.DataFrame({"id": [1] * 4, "study_interval": [2022] * 4,
                               "day_in_study": [1, 2, 2, 3],
                               "phase": ["unrecognised", "luteal", "follicular", "luteal"],
                               "audit_conflicting_key": [False, True, True, False]})
        daily = preprocessing.add_labels(preprocessing.create_daily_grid({"labels": labels}), labels)
        self.assertTrue(daily["phase"].iloc[:2].isna().all())
        self.assertTrue(daily["exact_luteal_onset"].isna().all())

    def test_participant_boundaries_do_not_create_onsets(self):
        labels = pd.DataFrame({"id": [1, 2], "study_interval": [2022, 2022],
                               "day_in_study": [1, 1], "phase": ["follicular", "luteal"],
                               "audit_conflicting_key": [False, False]})
        daily = preprocessing.add_labels(preprocessing.create_daily_grid({"labels": labels}), labels)
        self.assertTrue(daily["exact_luteal_onset"].isna().all())

    def test_duplicate_daily_join_fails(self):
        grid = pd.DataFrame({"id": [1], "study_interval": [2022], "day_in_study": [1]})
        with self.assertRaises(pd.errors.MergeError):
            preprocessing.join_daily(grid, pd.concat([grid, grid]))

    def test_coverage_counts_union_and_day_boundary_gaps(self):
        coverage = preprocessing.glucose_day_coverage(np.array([60, 60, 120, 86340]))
        self.assertEqual(coverage["glucose_nominal_observed_seconds"], 420)
        self.assertEqual(coverage["glucose_longest_uncovered_seconds"], 85920)
        full = preprocessing.glucose_day_coverage(np.arange(0, 86400, 300))
        self.assertEqual(full["glucose_nominal_coverage_fraction"], 1)
        self.assertEqual(full["glucose_longest_uncovered_seconds"], 0)
        empty = preprocessing.glucose_day_coverage(np.array([]))
        self.assertEqual(empty["glucose_longest_uncovered_seconds"], 86400)

    def test_output_guards(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            with self.assertRaises(ValueError):
                preprocessing.validate_output_paths(raw, [raw / "bad", root / "reports"])
            with self.assertRaises(ValueError):
                preprocessing.validate_output_paths(raw, [root / "output", root / "output/nested"])
            output = root / "output"
            output.mkdir()
            (output / "existing.csv").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                preprocessing.validate_output_paths(raw, [output, root / "reports"])

    def test_end_to_end_outputs_are_reproducible_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            write_synthetic_exports(raw_dir)
            hashes = {path.name: preprocessing.file_sha256(path) for path in raw_dir.glob("*.csv")}
            runs = []
            for number in [1, 2]:
                args = argparse.Namespace(raw_dir=raw_dir,
                                          processed_dir=root / f"processed_{number}",
                                          report_dir=root / f"reports_{number}")
                with redirect_stdout(io.StringIO()):
                    daily = preprocessing.run_audit(args)
                runs.append(args)
                self.assertEqual(daily["day_in_study"].tolist(), [1, 2, 3])
                self.assertTrue(daily["study_interval"].eq(2022).all())
                self.assertEqual(daily.loc[1, "glucose_records"], 0)
                self.assertTrue(daily["exact_luteal_onset"].isna().all())
                self.assertTrue(daily.loc[2, "uncertain_luteal_entry"])
            for attribute in ["processed_dir", "report_dir"]:
                first, second = [getattr(run, attribute) for run in runs]
                for path in first.glob("*.csv"):
                    self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            for path in raw_dir.glob("*.csv"):
                self.assertEqual(preprocessing.file_sha256(path), hashes[path.name])
            with self.assertRaises(FileExistsError):
                preprocessing.run_audit(runs[0])

    def test_glucose_diagnostics_separate_samples_bins_and_uncovered_time(self):
        result = preprocessing.glucose_day_diagnostics(np.array([60, 60, 120, 86340]))
        self.assertEqual(result["glucose_unique_samples"], 3)
        self.assertEqual(result["glucose_occupied_5min_bins"], 2)
        self.assertEqual(result["glucose_nominal_observed_seconds"], 420)
        self.assertEqual(result["glucose_boundary_uncovered_seconds"], 60)
        self.assertEqual(result["glucose_longest_interior_uncovered_seconds"], 85920)
        self.assertEqual(result["glucose_intervals_over_30min"], 1)

    def test_hrv_overshoot_is_retained_without_clipping(self):
        data = pd.DataFrame({"id": [1] * 4, "study_interval": [2022] * 4,
                             "day_in_study": [1] * 4, "rmssd": [20] * 4,
                             "coverage": [0.79, 0.8, 1.005, np.inf],
                             "audit_structurally_usable": [True, True, True, False]})
        settings = preprocessing.eligibility_settings(argparse.Namespace())
        counts = preprocessing.hrv_quality_counts(data, settings)
        self.assertEqual(counts.hrv_quality_windows.iat[0], 2)
        self.assertEqual(counts.hrv_bounded_quality_windows.iat[0], 1)
        self.assertEqual(data.coverage.iloc[2], 1.005)

    def test_phase_uncertainty_has_distinct_censoring_categories(self):
        labels = pd.DataFrame({"id": [1] * 6, "study_interval": [2022] * 6,
                               "day_in_study": [1, 3, 4, 6, 7, 8],
                               "phase": ["luteal", "luteal", "follicular", "luteal", "follicular", "luteal"],
                               "audit_conflicting_key": [False] * 6})
        daily = preprocessing.add_labels(preprocessing.create_daily_grid({"labels": labels}), labels)
        transitions = preprocessing.phase_transition_review(daily)
        self.assertEqual(transitions.transition_status.tolist(), [
            "left_censored", "possible_continuation", "interval_censored", "exact"])
        self.assertEqual(transitions.earliest_onset_day.fillna(-1).tolist(), [-1, -1, 5, 8])
        self.assertEqual(transitions.latest_onset_day.fillna(-1).tolist(), [1, -1, 6, 8])

    def test_main_sleep_overlap_flags_both_sessions_across_days(self):
        data = pd.DataFrame({"id": [1] * 3, "study_interval": [2022] * 3,
                             "day_in_study": [2, 3, 4],
                             "audit_start_seconds": [86400 + 23 * 3600, 2 * 86400 + 6 * 3600, 3 * 86400],
                             "audit_end_seconds": [2 * 86400 + 7 * 3600, 3 * 86400, 3 * 86400 + 3600]})
        flags = preprocessing.overlapping_session_flags(data, pd.Series(True, index=data.index))
        self.assertEqual(flags.tolist(), [True, True, False])

    def test_main_sleep_selection_excludes_naps_and_multiple_main_days(self):
        data = pd.DataFrame({"id": [1] * 4, "study_interval": [2022] * 4,
                             "day_in_study": [1, 2, 2, 3], "mainsleep": [False, True, True, True],
                             "duration": [3600000] * 4, "audit_start_seconds": [0, 86400, 90000, 172800],
                             "audit_end_seconds": [3600, 90000, 93600, 176400],
                             "audit_structurally_usable": [True] * 4})
        result = preprocessing.review_sleep_sessions(data, preprocessing.eligibility_settings(argparse.Namespace()))
        self.assertEqual(result.review_selected.tolist(), [False, False, False, True])

    def test_temperature_requires_exact_selected_sleep_match(self):
        sleep = pd.DataFrame({"id": [1], "study_interval": [2022],
                              "audit_start_seconds": [86400], "audit_end_seconds": [90000],
                              "review_selected": [True]})
        data = pd.DataFrame({"id": [1, 1], "study_interval": [2022, 2022],
                             "day_in_study": [1, 2], "audit_start_seconds": [86400, 86400],
                             "audit_end_seconds": [90000, 90001], "nightly_temperature": [35, 35],
                             "temperature_samples": [100, 100], "type": ["SKIN", "SKIN"],
                             "audit_structurally_usable": [True, True]})
        result = preprocessing.review_temperature_sessions(data, sleep)
        self.assertEqual(result.review_selected.tolist(), [True, False])

    def test_quality_settings_reject_invalid_thresholds(self):
        for settings in [{"min_glucose_coverage": np.nan}, {"min_hrv_coverage": 1.1},
                         {"min_hrv_windows": 0}, {"max_glucose_gap_hours": -1}]:
            with self.assertRaises(ValueError):
                preprocessing.eligibility_settings(argparse.Namespace(**settings))

    def test_rhr_equal_values_on_distinct_days_are_not_duplicates(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            path = Path(directory) / "resting_heart_rate.csv"
            path.write_text("id,study_interval,day_in_study,value,error\n"
                            "1,2022,1,60,1\n1,2022,2,60,1\n", encoding="utf-8")
            data, summary, _ = preprocessing.audit_table(
                Path(directory), "resting_heart_rate", preprocessing.TABLE_SPECS["resting_heart_rate"])
            self.assertEqual(len(data), 2)
            self.assertEqual(summary["exact_duplicates_removed"], 0)
            self.assertEqual(summary["conflicting_key_rows"], 0)

    def test_eligibility_review_end_to_end_and_unit_exclusion(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            write_synthetic_exports(raw_dir)
            for path in raw_dir.glob("*.csv"):
                original = pd.read_csv(path)
                for column in ["day_in_study", "sleep_start_day_in_study", "sleep_end_day_in_study"]:
                    if column in original:
                        original[column] = original[column].replace({3: 2})
                if path.stem == "glucose":
                    times = [f"{minute // 60:02d}:{minute % 60:02d}:00" for minute in range(0, 1440, 5)]
                    original = original.loc[original.index.repeat(288)].reset_index(drop=True)
                    original["timestamp"] = times * (len(original) // 288)
                if path.stem == "heart_rate_variability_details":
                    original = original.loc[original.index.repeat(6)].reset_index(drop=True)
                    original["timestamp"] = [f"01:{minute:02d}:00" for minute in range(0, 30, 5)] * (len(original) // 6)
                    original["coverage"] = 1.005
                second = original.assign(id=6)
                pd.concat([original, second]).to_csv(path, index=False)
            args = argparse.Namespace(raw_dir=raw_dir, processed_dir=root / "processed",
                                      report_dir=root / "reports", eligibility_review=True)
            with redirect_stdout(io.StringIO()):
                preprocessing.run_audit(args)
            daily = pd.read_csv(args.processed_dir / "quality_review_daily_2022.csv")
            self.assertTrue(daily.loc[daily.id.eq(1), "quality_units_accepted"].all())
            self.assertFalse(daily.loc[daily.id.eq(6), "quality_units_accepted"].any())
            self.assertFalse(daily.loc[daily.id.eq(6), "same_day_quality_overlap"].any())
            self.assertTrue(daily.loc[daily.id.eq(1), "same_day_quality_overlap"].all())
            self.assertTrue(daily.quality_glucose.all())
            self.assertTrue(daily.quality_hrv.all())
            self.assertTrue(daily.study_interval.eq(2022).all())
            self.assertFalse(daily.duplicated(preprocessing.KEY).any())
            transitions = pd.read_csv(args.processed_dir / "phase_transition_review_2022.csv")
            self.assertTrue(transitions.transition_status.eq("exact").all())
            self.assertEqual(int((daily.same_day_quality_overlap & daily.exact_luteal_onset.eq(1)).sum()), 1)
            log = (args.report_dir / "decision_log.md").read_text(encoding="utf-8")
            self.assertIn("all three primary input sets", log)
            self.assertIn("not be used as the final modelling cohort", log)


if __name__ == "__main__":
    unittest.main()
