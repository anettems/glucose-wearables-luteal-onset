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


if __name__ == "__main__":
    unittest.main()
