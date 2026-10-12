"""Focused checks for the descriptive 2022 audit; run with unittest discover."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
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


def write_temporal_exports(raw_dir):
    write_synthetic_exports(raw_dir)
    for path in raw_dir.glob("*.csv"):
        original = pd.read_csv(path).iloc[[0]]
        records = []
        for day in range(1, 7):
            item = original.copy()
            if "day_in_study" in item:
                item["day_in_study"] = day
            if path.stem == "hormones_and_selfreport":
                item["phase"] = "Follicular" if day < 4 else "Luteal"
            if "sleep_end_day_in_study" in item:
                start_day, start_hour = (1, 0) if day == 1 else (day - 1, 23)
                end_hour = 9 if day == 4 else 7
                item["sleep_start_day_in_study"] = start_day
                item["sleep_end_day_in_study"] = day
                item["sleep_start_timestamp"] = f"{start_hour:02d}:00:00"
                item["sleep_end_timestamp"] = f"{end_hour:02d}:00:00"
                if path.stem == "sleep":
                    minutes = ((day - start_day) * 24 + end_hour - start_hour) * 60
                    item["duration"], item["timeinbed"], item["minutesasleep"] = minutes * 60000, minutes, minutes
            if path.stem in ["glucose", "heart_rate_variability_details"]:
                minutes = range(0, 1440, 5) if path.stem == "glucose" else range(60, 90, 5)
                item = item.loc[item.index.repeat(len(minutes))].reset_index(drop=True)
                item["timestamp"] = [f"{minute // 60:02d}:{minute % 60:02d}:00" for minute in minutes]
                if path.stem == "heart_rate_variability_details":
                    item["coverage"] = 1.005
            records.append(item)
        first = pd.concat(records, ignore_index=True)
        second, third = first.assign(id=6), first.assign(id=11)
        if path.stem == "glucose":
            second["glucose_value"] /= preprocessing.MG_DL_TO_MMOL_L
            third["glucose_value"] /= preprocessing.MG_DL_TO_MMOL_L
        pd.concat([first, second, third, first.assign(study_interval=2024)]).to_csv(path, index=False)


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

    def test_eligibility_review_end_to_end_and_confirmed_unit_inclusion(self):
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
                if path.stem == "glucose":
                    second["glucose_value"] /= preprocessing.MG_DL_TO_MMOL_L
                pd.concat([original, second]).to_csv(path, index=False)
            args = argparse.Namespace(raw_dir=raw_dir, processed_dir=root / "processed",
                                      report_dir=root / "reports", eligibility_review=True)
            with redirect_stdout(io.StringIO()):
                preprocessing.run_audit(args)
            daily = pd.read_csv(args.processed_dir / "quality_review_daily_2022.csv")
            self.assertTrue(daily.loc[daily.id.eq(1), "quality_units_accepted"].all())
            self.assertTrue(daily.loc[daily.id.eq(6), "quality_units_accepted"].all())
            self.assertTrue(daily.loc[daily.id.eq(6), "same_day_quality_overlap"].all())
            self.assertTrue(daily.loc[daily.id.eq(1), "same_day_quality_overlap"].all())
            self.assertTrue(daily.quality_glucose.all())
            self.assertTrue(daily.quality_hrv.all())
            self.assertTrue(daily.study_interval.eq(2022).all())
            self.assertFalse(daily.duplicated(preprocessing.KEY).any())
            transitions = pd.read_csv(args.processed_dir / "phase_transition_review_2022.csv")
            self.assertTrue(transitions.transition_status.eq("exact").all())
            self.assertEqual(int((daily.same_day_quality_overlap & daily.exact_luteal_onset.eq(1)).sum()), 2)
            log = (args.report_dir / "decision_log.md").read_text(encoding="utf-8")
            self.assertIn("all three primary input sets", log)
            glucose = pd.read_csv(args.processed_dir / "glucose_2022_audited.csv")
            self.assertTrue(np.allclose(glucose.glucose_value, 1.0))
            self.assertIn("not be used as the final modelling cohort", log)


    def test_hrv_matches_whole_intervals_to_sleep_across_midnight(self):
        start = 86400 + 23 * 3600
        end = 2 * 86400 + 7 * 3600
        sleep = pd.DataFrame({"id": [1], "study_interval": [2022], "day_in_study": [2],
                              "audit_start_seconds": [start], "audit_end_seconds": [end],
                              "review_selected": [True]})
        starts = [start, start + 55 * 60, end - 300, end - 120, start - 120, start]
        data = pd.DataFrame({"id": [1] * 5 + [2], "study_interval": [2022] * 6,
                             "day_in_study": [1, 1, 2, 2, 1, 1], "timestamp": ["00:00:00"] * 6,
                             "rmssd": [20] * 6, "coverage": [1.005] * 6,
                             "audit_end_seconds": np.array(starts) + 300,
                             "audit_structurally_usable": [True] * 6})
        matched = preprocessing.align_hrv_to_sleep(data, sleep, preprocessing.eligibility_settings(argparse.Namespace()))
        self.assertEqual(matched.matched_main_sleep.tolist(), [True, True, True, False, False, False])
        self.assertEqual(matched.matched_sleep_end_day.fillna(-1).tolist(), [2, 2, 2, -1, -1, -1])
        counts = preprocessing.sleep_hrv_counts(matched)
        self.assertEqual(counts.hrv_main_sleep_quality_windows.tolist(), [3])
        self.assertEqual(counts.day_in_study.tolist(), [2])
        self.assertEqual(matched.day_in_study.tolist(), data.day_in_study.tolist())

    def test_calendar_history_cannot_bridge_gaps_or_participants(self):
        daily = pd.DataFrame({"id": [1] * 5 + [2], "study_interval": [2022] * 6,
                              "day_in_study": [1, 2, 4, 5, 6, 3],
                              "phase": [None, "follicular", None, None, "luteal", "luteal"],
                              "aligned_inputs_available": [True] * 6})
        result = preprocessing.add_calendar_history(daily)
        self.assertEqual(result.history_usable_days.tolist(), [1, 2, 2, 2, 3, 1])
        self.assertEqual(result.eligible_prediction.tolist(), [False, False, False, False, True, False])
        self.assertEqual(result.history_first_glucose_day.iloc[4], 3)
        self.assertEqual(result.history_first_aligned_day.iloc[4], 4)

    def test_morning_alignment_uses_prior_day_and_inclusive_cutoff(self):
        quality = pd.DataFrame({"id": [1] * 3, "study_interval": [2022] * 3,
                                "day_in_study": [1, 2, 3], "phase": ["follicular"] * 3,
                                "luteal_status": [0] * 3, "exact_luteal_onset": [None, 0, 0],
                                "uncertain_luteal_entry": [False] * 3, "quality_units_accepted": [True] * 3,
                                "quality_glucose": [False, True, True], "quality_rhr": [True, False, True],
                                "glucose_unique_samples": [10, 288, 144]})
        sleep = quality[preprocessing.KEY].copy()
        sleep["audit_start_seconds"] = sleep.day_in_study * 86400
        sleep["audit_end_seconds"] = sleep.day_in_study * 86400 + np.array([43200, 43201, 43199])
        sleep["review_selected"] = True
        intervals = quality[preprocessing.KEY].copy()
        intervals = intervals.loc[intervals.index.repeat(6)].reset_index(drop=True)
        intervals["matched_sleep_end_day"] = intervals.day_in_study
        intervals["matched_main_sleep"], intervals["usable_in_main_sleep"] = True, True
        result = preprocessing.aligned_morning_days(quality, sleep, sleep, intervals,
                                                   preprocessing.eligibility_settings(argparse.Namespace()), 12)
        self.assertEqual(result.aligned_glucose.tolist(), [False, False, True])
        self.assertEqual(result.aligned_rhr.tolist(), [False, True, False])
        self.assertEqual(result.aligned_main_sleep.tolist(), [True, False, True])
        self.assertEqual(result.aligned_hrv.tolist(), [True, False, True])
        self.assertEqual(result.aligned_temperature.tolist(), [True, False, True])
        self.assertEqual(result.prior_day_glucose_unique_samples.iloc[1:].tolist(), [10, 288])
        self.assertFalse(result.eligible_prediction.any())

    def test_temporal_review_reproducibility_confirmed_units_and_history(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            write_temporal_exports(raw_dir)
            hashes = {path.name: preprocessing.file_sha256(path) for path in raw_dir.glob("*.csv")}
            runs = []
            for number in [1, 2]:
                args = argparse.Namespace(raw_dir=raw_dir, processed_dir=root / f"processed_{number}",
                                          report_dir=root / f"reports_{number}", temporal_review=True)
                with redirect_stdout(io.StringIO()):
                    preprocessing.run_audit(args)
                runs.append(args)
                daily = pd.read_csv(args.processed_dir / "morning_alignment_12_2022.csv")
                eligible = pd.read_csv(args.processed_dir / "eligible_observations_12_2022.csv")
                self.assertTrue(daily.study_interval.eq(2022).all())
                self.assertFalse(daily.duplicated(preprocessing.KEY).any())
                self.assertEqual(eligible.id.tolist(), [1] * 3 + [6] * 3 + [11] * 3)
                self.assertEqual(eligible.day_in_study.tolist(), [4, 5, 6] * 3)
                self.assertTrue(eligible.history_usable_days.eq(3).all())
                self.assertEqual(int(eligible.exact_luteal_onset.eq(1).sum()), 3)
                self.assertFalse((args.processed_dir / "eligible_observations_08_2022.csv").exists())
                self.assertFalse((args.processed_dir / "morning_alignment_08_2022.csv").exists())
                attrition = pd.read_csv(args.report_dir / "temporal_attrition.csv")
                self.assertEqual(attrition.cutoff_hour.unique().tolist(), [12])
                log = (args.report_dir / "temporal_protocol.md").read_text(encoding="utf-8")
                config = json.loads((args.report_dir / "run_config.json").read_text())
                self.assertEqual(config["prediction_cutoff"], "12:00:00")
                self.assertNotIn("sensitivity_cutoff", config["temporal_settings"])
                self.assertEqual(config["temporal_settings"]["prediction_cutoff"], "12:00:00")
                self.assertIn("Primary cutoff: 12:00", log)
                self.assertIn("sync times", config["temporal_settings"]["availability_assumption"])
                self.assertIn("not demonstrated real-time deployment", log)
            for attribute in ["processed_dir", "report_dir"]:
                first, second = [getattr(run, attribute) for run in runs]
                for path in first.glob("*.csv"):
                    self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            self.assertEqual((runs[0].report_dir / "temporal_protocol.md").read_bytes(),
                             (runs[1].report_dir / "temporal_protocol.md").read_bytes())
            for path in raw_dir.glob("*.csv"):
                self.assertEqual(preprocessing.file_sha256(path), hashes[path.name])


    def test_glucose_conversion_preserves_source_values_and_rejects_double_conversion(self):
        data = pd.DataFrame({"id": [1, 6, 11, 6], "study_interval": [2022] * 4,
                             "glucose_value": [5.0, 100.0, 118.0, np.nan]})
        result = preprocessing.standardise_glucose_units(data)
        self.assertTrue(np.allclose(result.glucose_value.iloc[:3], [5.0, 5.55, 6.549]))
        self.assertTrue(result.glucose_source_value.equals(data.glucose_value))
        self.assertEqual(result.glucose_source_unit.tolist(), ["mmol/L", "mg/dL", "mg/dL", "mg/dL"])
        self.assertEqual(result.glucose_unit_converted.tolist(), [False, True, True, True])
        self.assertTrue(result.glucose_unit.eq("mmol/L").all())
        self.assertTrue(pd.isna(result.glucose_value.iloc[3]))
        self.assertEqual(data.glucose_value.iloc[1], 100)
        with self.assertRaises(ValueError):
            preprocessing.standardise_glucose_units(result)

    def test_conversion_preserves_invalid_flags_and_filters_2024_first(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parent / "tests") as directory:
            path = Path(directory) / "glucose.csv"
            path.write_text("id,study_interval,day_in_study,timestamp,glucose_value\n"
                            "6,2022,1,00:00:00,100\n6,2022,1,00:05:00,bad\n"
                            "11,2022,1,00:00:00,0\n11,2022,1,00:05:00,-18\n"
                            "6,2024,1,00:00:00,100\n", encoding="utf-8")
            before = path.read_bytes()
            data, summary, _ = preprocessing.audit_table(
                Path(directory), "glucose", preprocessing.TABLE_SPECS["glucose"])
            self.assertEqual(summary["excluded_other_interval_rows"], 1)
            self.assertEqual(summary["invalid_value_rows"], 3)
            self.assertEqual(int(data.audit_structurally_usable.sum()), 1)
            self.assertEqual(data.glucose_value.iloc[0], 5.55)
            self.assertEqual(data.glucose_source_value.iloc[1], "bad")
            self.assertTrue(data.study_interval.eq(2022).all())
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
