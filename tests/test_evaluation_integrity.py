"""Scientific invariants and failure cases for the primary evaluation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import daily_metrics as metrics
import daily_models as models
import main
import onset_alerts as onset
import participant_folds as folds
import primary_evaluation as evaluation
import primary_protocol as config
from model_inputs import FEATURE_SETS, validate_features
from test_primary_evaluation import freeze_synthetic_protocol, prediction_rows, synthetic_observations, synthetic_references


def paired_predictions():
    rows = []
    for model_index, input_set in enumerate(FEATURE_SETS):
        for person in range(1, 5):
            for day, label in [(1, 0), (2, 1)]:
                rows.append({"id": person, "study_interval": 2022, "day_in_study": day,
                             "luteal_status": label, "probability": (person + day + model_index) / 10,
                             "input_set": input_set, "outer_test_fold": person})
    return pd.DataFrame(rows)


class EvaluationIntegrityTests(unittest.TestCase):
    def test_pairs_reject_missing_keys_changed_labels_and_nonfinite_probabilities(self):
        predictions = paired_predictions()
        cases = [predictions.drop(index=0), pd.concat([predictions, predictions.iloc[[0]]]),
                 predictions.assign(probability=np.nan), predictions.assign(probability=np.inf)]
        changed = predictions.copy()
        changed.loc[0, "luteal_status"] = 1
        cases.append(changed)
        changed = predictions.copy()
        changed.loc[0, "outer_test_fold"] = 2
        cases.append(changed)
        for frame in cases:
            with self.subTest(case=len(frame)), self.assertRaises(ValueError):
                metrics.validate_paired_predictions(frame)

    def test_bootstrap_preserves_whole_participants_and_pairs_every_metric(self):
        predictions = paired_predictions().sample(frac=1, random_state=7)
        with patch.object(config, "BOOTSTRAP_RESAMPLES", 8):
            bootstrap = metrics.bootstrap_metrics(predictions)
            repeat = metrics.bootstrap_metrics(predictions)
        pd.testing.assert_frame_equal(bootstrap, repeat)
        self.assertTrue(any(len(set(json.loads(draw))) < 4 for draw in bootstrap.participant_draw))
        for record in bootstrap.to_dict("records"):
            draw = json.loads(record["participant_draw"])
            self.assertEqual(len(draw), 4)
            for input_set in FEATURE_SETS:
                subset = predictions.loc[predictions.input_set.eq(input_set)]
                sample = pd.concat([subset.loc[subset.id.eq(person)] for person in draw])
                self.assertEqual(len(sample), 8)
                self.assertAlmostEqual(record[f"{input_set}_average_precision"],
                                       metrics.average_precision(sample.luteal_status, sample.probability))
                self.assertAlmostEqual(record[f"{input_set}_roc_auc"],
                                       metrics.roc_auc(sample.luteal_status, sample.probability))
            self.assertAlmostEqual(record["combined_minus_wearable_average_precision"],
                                   record["combined_average_precision"] - record["wearable_average_precision"])

    def test_undefined_bootstrap_draws_are_retained_and_counted(self):
        predictions = paired_predictions().loc[lambda frame: frame.id.le(2)].copy()
        predictions["luteal_status"] = predictions.id - 1
        with patch.object(config, "BOOTSTRAP_RESAMPLES", 32):
            bootstrap = metrics.bootstrap_metrics(predictions)
        self.assertEqual(len(bootstrap), 32)
        self.assertTrue(bootstrap.wearable_roc_auc.isna().any())
        self.assertTrue(bootstrap.wearable_average_precision.isna().any())
        summary, paired = metrics.summary_tables(predictions, bootstrap)
        row = summary.loc[summary.input_set.eq("wearable") & summary.metric.eq("roc_auc")].iloc[0]
        self.assertEqual(row.bootstrap_valid_resamples, bootstrap.wearable_roc_auc.notna().sum())
        self.assertEqual(paired.bootstrap_requested_resamples.iloc[0], 32)

    def test_model_setting_ties_choose_smaller_c_and_use_separate_participants(self):
        development = synthetic_observations(participants=6)
        assignments = pd.DataFrame({"id": range(1, 7), "validation_fold": [1, 2, 3, 1, 2, 3]})
        def predict(training, validation, feature_columns, inverse_penalty_strength):
            self.assertFalse(set(training.id) & set(validation.id))
            return np.full(len(validation), 0.5), {}
        with patch.object(models, "fit_and_predict", side_effect=predict) as fitted:
            selected, tuning, validation = models.tune_logistic_model(
                development, assignments, FEATURE_SETS["combined"],
            )
        self.assertEqual(selected, 0.01)
        self.assertEqual(fitted.call_count, 12)
        self.assertEqual(len(tuning), 12)
        self.assertFalse(validation.duplicated(config.KEY).any())
        self.assertEqual(len(validation), len(development))

    def test_convergence_warning_stops_model_evaluation(self):
        class FailingEstimator:
            def fit(self, features, labels):
                warnings.warn("did not converge", ConvergenceWarning)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            models.fit_model(FailingEstimator(), np.zeros((2, 1)), np.array([0, 1]))

    def test_folds_require_evaluable_events_and_observed_training_features(self):
        observations = synthetic_observations(participants=4)
        assignments = pd.DataFrame({"id": range(1, 5), "test_fold": [1, 1, 2, 2]})
        references = synthetic_references(observations)
        with self.assertRaisesRegex(ValueError, "Fold feasibility"):
            folds.validate_fold_assignment(observations, references.assign(day_in_study=100),
                                           assignments, "test_fold", 2)
        observations.loc[observations.id.ne(1), "sleeping_hr_bpm"] = np.nan
        with self.assertRaisesRegex(ValueError, "Training feasibility"):
            folds.validate_fold_assignment(observations, references, assignments, "test_fold", 2)

    def test_event_matching_is_one_to_one_and_validation_matches_detailed_results(self):
        predictions = prediction_rows([1, 2, 3, 4, 5], [0.9, 0.1, 0.9, 0.1, 0.9])
        references = pd.DataFrame({"id": [1, 1, 1], "study_interval": [2022] * 3,
                                   "day_in_study": [2, 4, 20]})
        generated, reference_rows, alert_rows, detailed = onset.evaluate_alerts(predictions, references, 0.5, 1)
        fast = onset.alert_metrics(predictions, references, 0.5, 1)
        self.assertEqual(fast, detailed)
        self.assertEqual(reference_rows.matched.sum(), 2)
        self.assertEqual(alert_rows.matched.sum(), 2)
        self.assertEqual(len(generated), 3)
        self.assertEqual(detailed["unmatched_alerts"], 1)
        self.assertFalse(reference_rows.loc[reference_rows.reference_day_in_study.eq(20), "evaluable"].iloc[0])
        self.assertEqual(reference_rows.loc[reference_rows.matched, "alert_day_in_study"].tolist(), [1, 3])

    def test_two_day_alerts_do_not_reset_across_gap_and_are_never_backdated(self):
        days = [1, 2, 4, 5, 6, 7]
        probabilities = [0.5, 0.5, 0.9, 0.1, 0.5, 0.5]
        indices = onset.alert_days(days, probabilities, 0.5, 2)
        self.assertEqual([days[index] for index in indices], [2, 7])
        for values in [[0.9, np.nan], [0.9, np.inf]]:
            with self.assertRaises(ValueError):
                onset.alert_days([1, 2], values, 0.5, 1)

    def test_output_guards_protect_raw_inputs_and_existing_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            raw, processed, report = root / "raw", root / "processed", root / "report"
            raw.mkdir()
            for target in [raw, raw / "nested", processed, processed / "nested", report / "nested", root]:
                with self.assertRaises(ValueError):
                    evaluation.create_evaluation_directory(target, processed, report, raw)
            output = root / "results"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                evaluation.create_evaluation_directory(output, processed, report, raw)
            self.assertEqual(marker.read_text(), "original")

    def test_protocol_and_folds_are_persisted_before_predictions(self):
        observations = synthetic_observations()
        alignment = observations[config.KEY + ["luteal_status", "exact_luteal_onset"]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, report, output = root / "processed", root / "report", root / "results"
            processed.mkdir()
            report.mkdir()
            observations.to_csv(processed / "observations_2022.csv", index=False)
            alignment.to_csv(processed / "prediction_alignment_14_2022.csv", index=False)
            (report / "run_config.json").write_text(json.dumps({"features_ready": True, "prediction_cutoff": "14:00:00"}))
            frozen = freeze_synthetic_protocol(root, processed, report)
            def stop_after_check(*args):
                for name in ["evaluation_protocol.json", "participant_folds.csv", "validation_fold_assignments.csv",
                             "fold_feasibility.csv", "provenance.json"]:
                    self.assertTrue((output / name).is_file(), name)
                raise RuntimeError("intentional stop before predictions")
            with patch.object(evaluation, "evaluate_participant_folds", side_effect=stop_after_check):
                with self.assertRaisesRegex(RuntimeError, "intentional stop"):
                    evaluation.run_primary_evaluation(processed, report, output, frozen_protocol_dir=frozen)
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "failed")
            self.assertFalse((output / "model_predictions.csv").exists())
            alignment.loc[0, "luteal_status"] = 1
            alignment.to_csv(processed / "prediction_alignment_14_2022.csv", index=False)
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                evaluation.load_primary_inputs(processed, report)

    def test_main_help_exposes_evaluation_options_and_rejects_nonbinary_outcomes(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit):
            main.main(["--help"])
        self.assertIn("--evaluate-primary", stream.getvalue())
        self.assertIn("--evaluation-dir", stream.getvalue())
        self.assertIn("--freeze-primary", stream.getvalue())
        self.assertIn("--frozen-protocol-dir", stream.getvalue())
        observations = synthetic_observations().assign(luteal_status=2)
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_features(observations)


if __name__ == "__main__":
    unittest.main()
