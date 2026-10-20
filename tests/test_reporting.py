"""Reporting checks use synthetic data; no held-out project results are generated."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import primary_evaluation as evaluation
import primary_protocol as protocol
import reporting
from reporting_tables import calibration_tables, descriptive_tables, matched_timing_table
from test_primary_evaluation import freeze_synthetic_protocol, synthetic_observations


def reporting_fixture(root):
    processed, preparation = root / "processed", root / "preparation"
    processed.mkdir()
    preparation.mkdir()
    observations = synthetic_observations()
    observations.to_csv(processed / "observations_2022.csv", index=False)
    observations[protocol.KEY + ["luteal_status", "exact_luteal_onset"]].to_csv(
        processed / "prediction_alignment_14_2022.csv", index=False,
    )
    (preparation / "run_config.json").write_text(json.dumps({"features_ready": True, "prediction_cutoff": "14:00:00"}))
    frozen = freeze_synthetic_protocol(root, processed, preparation)
    return processed, preparation, frozen


class ReportingTests(unittest.TestCase):
    def test_coverage_keeps_excluded_participants_and_uses_calendar_gaps(self):
        observations = synthetic_observations(2)
        alignment = observations[protocol.KEY].copy()
        alignment = pd.concat([alignment, alignment.loc[alignment.id.eq(2)].assign(id=3)], ignore_index=True)
        observations = observations.loc[observations.day_in_study.isin([1, 4, 6])].copy()
        observations.loc[observations.id.eq(1), "sleeping_hr_bpm"] = np.nan
        tables = descriptive_tables(observations, alignment)
        coverage = tables["participant_coverage"].set_index("id")
        self.assertEqual(coverage.loc[1, "eligible_days"], 3)
        self.assertEqual(coverage.loc[1, "ineligible_days_within_eligible_span"], 3)
        self.assertEqual(coverage.loc[1, "eligible_fraction_of_aligned_days"], 0.5)
        self.assertFalse(coverage.loc[3, "in_modelling_cohort"])
        self.assertTrue(np.isnan(coverage.loc[3, "luteal_fraction_of_eligible_days"]))
        missing = tables["feature_missingness"].query("id == 'all' and feature == 'sleeping_hr_bpm'").iloc[0]
        self.assertEqual(missing.missing_fraction, 0.5)
        distribution = tables["feature_distributions"].set_index("feature").loc["sleeping_hr_bpm"]
        self.assertEqual(distribution.observed_days, 3)
        self.assertEqual(tables["outcome_balance"].eligible_days.sum(), 6)

    def test_calibration_includes_boundary_probabilities_and_keeps_empty_bins(self):
        base = synthetic_observations(1).iloc[:4][protocol.KEY + ["luteal_status"]].copy()
        base["probability"] = [0, 0.1, 0.9, 1]
        predictions = pd.concat([base.assign(input_set=name) for name in ["glucose", "wearable", "combined"]])
        tables = calibration_tables(predictions)
        bins = tables["calibration_bins"].query("input_set == 'combined'").set_index("bin")
        self.assertEqual(bins.days.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(bins.loc[9, "mean_probability"], 0.95)
        self.assertTrue(np.isnan(bins.loc[2, "observed_luteal_fraction"]))
        expected_brier = np.mean((base.probability - base.luteal_status) ** 2)
        self.assertTrue(np.allclose(tables["probability_summary"].brier_score, expected_brier))
        predictions.loc[predictions.input_set.eq("combined"), "probability"] = 1.01
        with self.assertRaises(ValueError):
            calibration_tables(predictions)

    def test_timing_keeps_sign_and_undefined_no_match_error(self):
        events = pd.DataFrame({"matched": [True, True, False], "reference_day_in_study": [5, 9, 12],
                               "alert_day_in_study": [3, 10, np.nan], "timing_error_days": [2, 1, np.nan]})
        matched = matched_timing_table(events)
        self.assertEqual(matched.signed_error_days.tolist(), [-2, 1])
        self.assertTrue(np.isnan(matched_timing_table(events.loc[~events.matched]).timing_error_days.mean()))
        events.loc[0, "timing_error_days"] = 1
        with self.assertRaisesRegex(ValueError, "timing errors disagree"):
            matched_timing_table(events)

    def test_no_matches_figure_labels_timing_as_undefined(self):
        import reporting_figures as figures
        rows = [{"input_set": name, "matched_starts": 0, "missed_evaluable_starts": 2, "unmatched_alerts": 1}
                for name in ["glucose", "wearable", "combined"]]
        empty = pd.DataFrame(columns=["input_set", "signed_error_days", "timing_error_days"])
        with patch.object(figures, "save_figure") as save:
            figures.onset_figure({"onset_summary": pd.DataFrame(rows), "matched_timing": empty}, None)
        figure = save.call_args.args[0]
        labels = figure.axes[1].get_legend_handles_labels()[1]
        self.assertTrue(all("n=0, MAE=undefined" in label for label in labels))
        figures.plt.close(figure)

    def test_report_guards_protect_inputs_freeze_evaluation_and_existing_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, preparation, frozen, completed = [root / name for name in ["processed", "preparation", "frozen", "evaluation"]]
            for protected in [processed, preparation, frozen, completed, Path(evaluation.__file__).parent / "data/raw"]:
                with self.assertRaises(ValueError):
                    reporting.create_report_directory(protected / "nested", processed, preparation, frozen, completed)
            output = root / "report"
            output.mkdir()
            (output / "keep.txt").write_text("original")
            with self.assertRaises(FileExistsError):
                reporting.create_report_directory(output, processed, preparation, frozen, completed)
            self.assertEqual((output / "keep.txt").read_text(), "original")

    def test_descriptive_only_does_not_read_evaluation_or_fit_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, preparation, frozen = reporting_fixture(root)
            with patch.object(reporting, "load_completed_evaluation", side_effect=AssertionError("No evaluation reads")), \
                    patch.object(evaluation, "evaluate_participant_folds", side_effect=AssertionError("No fitting")):
                output = reporting.generate_report(processed, preparation, frozen, root / "report")
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "complete")
            self.assertIn("Model outputs pending", (output / "README.md").read_text())
            self.assertFalse((output / "calibration.png").exists())
            manifest = json.loads((output / "report_manifest.json").read_text())
            for name, digest in manifest["outputs"].items():
                self.assertEqual(evaluation.preprocessing.file_sha256(output / name), digest)

    def test_complete_synthetic_evaluation_produces_all_figures_without_refitting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, \
                patch.object(protocol, "BOOTSTRAP_RESAMPLES", 8):
            root = Path(directory)
            processed, preparation, frozen = reporting_fixture(root)
            completed = root / "evaluation"
            evaluation.run_primary_evaluation(processed, preparation, completed, frozen_protocol_dir=frozen)
            with patch.object(evaluation, "evaluate_participant_folds", side_effect=AssertionError("No refitting")):
                output = reporting.generate_report(processed, preparation, frozen, root / "report", completed=completed)
            for name in ["participant_coverage", "outcome_and_missingness", "feature_distributions", "daily_performance",
                         "discrimination_curves", "calibration", "onset_counts_and_timing"]:
                for extension in ["png", "pdf"]:
                    self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
            self.assertEqual(len(pd.read_csv(output / "calibration_bins.csv")), 30)
            data, references, _ = evaluation.load_primary_inputs(processed, preparation)
            folds = pd.read_csv(frozen / "participant_folds.csv")
            def load():
                return reporting.load_completed_evaluation(completed, frozen, data, references, folds)
            saved = pd.read_csv(completed / "model_summary.csv")
            changed = saved.copy()
            changed.loc[0, "estimate"] += 0.1
            changed.to_csv(completed / "model_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load()
            saved.to_csv(completed / "model_summary.csv", index=False)
            saved_events = pd.read_csv(completed / "onset_summary.csv")
            changed_events = saved_events.copy()
            changed_events.loc[0, "unmatched_alerts"] += 1
            changed_events.to_csv(completed / "onset_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load()
            saved_events.to_csv(completed / "onset_summary.csv", index=False)
            predictions = pd.read_csv(completed / "model_predictions.csv")
            changed_predictions = predictions.copy()
            changed_predictions.loc[changed_predictions.id.eq(1), "outer_test_fold"] = 99
            changed_predictions.to_csv(completed / "model_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "folds differ"):
                load()
            predictions.to_csv(completed / "model_predictions.csv", index=False)
            status = completed / "run_status.json"
            status.write_text(json.dumps({"status": "failed"}))
            with self.assertRaisesRegex(ValueError, "completed evaluation"):
                load()


if __name__ == "__main__":
    unittest.main()
