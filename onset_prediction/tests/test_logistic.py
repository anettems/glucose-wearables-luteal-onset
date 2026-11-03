#### Verify direct onset regression, training-only preprocessing and probability statistics.
#### Synthetic participant cohorts test logical outputs and boundary cases before real fitting.
#### Freeze tests use temporary files and never change the saved project preparation.

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_prediction import protocol
from onset_prediction.evaluation import check_shared_replacements, fit_test_folds
from onset_prediction.freezing import construction_sources, freeze_logistic, load_frozen
from onset_prediction.logistic import (fit_logistic, input_matrix, parameter_rows,
                                       predict_logistic, select_strength, tune_logistic)
from onset_prediction.metrics import (bootstrap_metrics, calibration_bins, interval_record,
                                      participant_draws, probability_metrics, validate_paired)
from preprocessing import file_sha256
from test_feasibility import synthetic_cohort


def model_data(people):
    #### Give every participant eight days with a clear relation between inputs and outcome.
    rows = []
    for person in people:
        for day in range(1, 9):
            row = {"id": person, "study_interval": 2022, "day_in_study": day,
                   protocol.TARGET: int(day >= 5)}
            row.update({name: day + person / 100 for name in protocol.feature_columns("combined")})
            rows.append(row)
    return pd.DataFrame(rows)


class LogisticTests(unittest.TestCase):
    def test_training_only_medians_scaling_and_shared_inputs(self):
        #### Extreme assessment values cannot change training replacements or population scaling.
        training, assessment = model_data([1, 2]), model_data([3])
        missing = "sleeping_hr_bpm__deviation"
        training.loc[[0, 2, 8], missing] = np.nan
        assessment[missing] = np.nan
        fitted = fit_logistic(training, "combined", 0.1)
        expected_median = training[missing].median()
        expected_filled = training[missing].fillna(expected_median)
        rows = parameter_rows(fitted).set_index("feature")
        self.assertAlmostEqual(rows.loc[missing, "training_median"], expected_median)
        self.assertAlmostEqual(rows.loc[missing, "mean"], expected_filled.mean())
        self.assertAlmostEqual(rows.loc[missing, "scale"], expected_filled.std(ddof=0))
        self.assertAlmostEqual(input_matrix(assessment, fitted.columns, fitted.medians)[0, fitted.columns.index(missing)], expected_median)
        wearable = parameter_rows(fit_logistic(training, "wearable", 0.1)).set_index("feature")
        for name in protocol.feature_columns("wearable"):
            self.assertEqual(rows.loc[name, "training_median"], wearable.loc[name, "training_median"])
        before = rows.copy()
        extreme = assessment.copy()
        extreme[protocol.feature_columns("glucose")] = 1e8
        predict_logistic(fitted, extreme)
        pd.testing.assert_frame_equal(before, parameter_rows(fitted).set_index("feature"))

    def test_probability_logic_and_parameter_reconstruction(self):
        #### Direct onset probabilities follow the learned relation and saved coefficients.
        training, assessment = model_data([1, 2]), model_data([3])
        fitted = fit_logistic(training, "combined", 0.1)
        probabilities = predict_logistic(fitted, assessment)
        self.assertGreater(probabilities[assessment[protocol.TARGET].eq(1)].mean(),
                           probabilities[assessment[protocol.TARGET].eq(0)].mean())
        rows = parameter_rows(fitted)
        matrix = input_matrix(assessment, fitted.columns, fitted.medians)
        expected = expit(rows.intercept.iloc[0] + ((matrix - rows["mean"].to_numpy()) /
                         rows.scale.to_numpy()) @ rows.coefficient.to_numpy())
        np.testing.assert_allclose(probabilities, expected, atol=1e-12)
        changed_labels = assessment.assign(**{protocol.TARGET: 1 - assessment[protocol.TARGET]})
        np.testing.assert_array_equal(probabilities, predict_logistic(fitted, changed_labels))
        self.assertFalse(set(protocol.KEY + [protocol.TARGET]) & set(fitted.columns))
        self.assertIsNone(fitted.pipeline.named_steps["logisticregression"].class_weight)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(fitted, path)
            np.testing.assert_array_equal(probabilities, predict_logistic(joblib.load(path), assessment))

    def test_constant_features_have_scale_one(self):
        #### A constant feature is valid and StandardScaler must not divide by zero.
        training = model_data([1, 2])
        training["skin_temperature_c"] = 7.0
        fitted = fit_logistic(training, "wearable", 0.01)
        rows = parameter_rows(fitted).set_index("feature")
        self.assertEqual(rows.loc["skin_temperature_c", "scale"], 1)

    def test_invalid_inputs_overlap_and_convergence_are_rejected(self):
        #### Stop rather than fit unknown outcomes, missing glucose or unconverged models.
        training, assessment = model_data([1, 2]), model_data([3])
        for invalid in (training.assign(**{protocol.TARGET: 0}),
                        training.assign(**{protocol.TARGET: np.nan}),
                        training.assign(glucose_cv_percent=np.nan),
                        training.assign(skin_temperature_c=np.inf)):
            with self.assertRaises(ValueError):
                fit_logistic(invalid, "combined", 0.1)
        fitted = fit_logistic(training, "combined", 0.1)
        with self.assertRaises(ValueError):
            predict_logistic(fitted, training)
        with self.assertRaises(ValueError):
            predict_logistic(fitted, assessment.assign(glucose_cv_percent=np.nan))
        with patch("onset_prediction.logistic.make_pipeline") as factory:
            factory.return_value.fit.side_effect = ConvergenceWarning("not converged")
            with self.assertRaises(RuntimeError):
                fit_logistic(training, "combined", 0.1)

    def test_tuning_scores_mean_ap_and_tie_break(self):
        #### Give each fold equal weight and choose the smaller C when mean AP is tied.
        rows = [{"C": strength, "validation_fold": fold, "average_precision": 0.5, "days": fold}
                for strength in (0.01, 0.1, 1, 10) for fold in (1, 2, 3)]
        scores = pd.DataFrame(rows)
        self.assertEqual(select_strength(scores), 0.01)
        scores.loc[scores.C.eq(0.1), "average_precision"] = [0.9, 0.4, 0.4]
        self.assertEqual(select_strength(scores), 0.1)
        with self.assertRaises(ValueError):
            select_strength(scores.iloc[1:])
        development = model_data(range(1, 7))
        assignments = pd.DataFrame({"id": range(1, 7), "validation_fold": [1, 2, 3, 1, 2, 3]})
        chosen, scores, predictions, parameters = tune_logistic(development, assignments, "wearable")
        self.assertEqual(len(scores), 12)
        self.assertEqual(chosen, select_strength(scores))
        self.assertEqual(len(predictions), 4 * len(development))
        self.assertEqual(len(parameters), 12 * 9)
        for _, frame in predictions.groupby("C"):
            self.assertFalse(frame.duplicated(protocol.KEY).any())
        with self.assertRaises(ValueError):
            tune_logistic(development, assignments.iloc[1:], "wearable")

    def test_shared_medians_checked_even_when_selected_C_differs(self):
        #### Shared replacement checks must not be bypassed by different selected strengths.
        rows = pd.DataFrame({"test_fold": [1, 1], "C": [0.01, 10],
                             "feature": ["skin_temperature_c"] * 2,
                             "replacement_allowed": [True, True], "training_median": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            check_shared_replacements(rows)


class ProbabilityTests(unittest.TestCase):
    def test_known_metric_values_and_undefined_classes(self):
        #### Verify metric meanings against small hand-calculated probability examples.
        predictions = pd.DataFrame({protocol.TARGET: [0, 0, 1, 1], "probability": [0.1, 0.4, 0.35, 0.8]})
        result = probability_metrics(predictions)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["brier_score"], 0.158125)
        result = probability_metrics(predictions.assign(**{protocol.TARGET: 0}))
        self.assertTrue(np.isnan(result["average_precision"]))
        self.assertTrue(np.isnan(result["roc_auc"]))
        for value in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                probability_metrics(predictions.assign(probability=value))

    def test_calibration_boundaries_and_empty_bins(self):
        #### Boundary probabilities enter the intended bin, including probability one.
        bins = calibration_bins(pd.DataFrame({protocol.TARGET: [0, 1, 0, 1],
                                              "probability": [0, 0.1, 0.5, 1]}))
        self.assertEqual(bins.loc[bins.days.gt(0), "bin"].tolist(), [0, 1, 5, 9])
        self.assertEqual(int(bins.days.sum()), 4)
        self.assertTrue(bins.loc[bins.days.eq(0), "observed_frequency"].isna().all())

    def test_bootstrap_pairing_repeated_participants_and_intervals(self):
        #### Selecting one participant twice must retain both copies and the paired differences.
        data = model_data([1, 2])
        parts = []
        for name, probability in (("glucose", 0.2), ("wearable", 0.3), ("combined", 0.3)):
            parts.append(data[protocol.KEY + [protocol.TARGET]].assign(probability=probability, input_set=name))
        predictions = pd.concat(parts, ignore_index=True)
        draws = pd.DataFrame({"replicate": [1, 2], "participant_draw": ["[1, 1]", "[2, 2]"]})
        bootstrap = bootstrap_metrics(predictions, draws)
        np.testing.assert_array_equal(bootstrap.combined_minus_wearable_average_precision, [0, 0])
        expected = probability_metrics(predictions.loc[predictions.input_set.eq("wearable") & predictions.id.eq(1)])
        self.assertEqual(bootstrap.wearable_brier_score.iloc[0], expected["brier_score"])
        repeated = participant_draws([1, 2], resamples=4)
        pd.testing.assert_frame_equal(repeated, participant_draws([2, 1], resamples=4))
        interval = interval_record([0, 1, 2, np.nan])
        self.assertAlmostEqual(interval["ci_lower"], 0.05)
        self.assertAlmostEqual(interval["ci_upper"], 1.95)
        self.assertEqual(interval["valid_resamples"], 3)
        with self.assertRaises(ValueError):
            bootstrap_metrics(predictions.loc[~((predictions.input_set == "combined") & (predictions.id == 1))], draws)

    def test_exactly_one_paired_probability_per_key_and_fold(self):
        #### Missing, duplicate, differently labelled or wrongly folded predictions fail checks.
        data = model_data([1, 2])
        assignments = pd.DataFrame({"id": [1, 2], "test_fold": [1, 2]})
        paired = pd.concat([data[protocol.KEY + [protocol.TARGET]].merge(assignments, on="id").assign(
            probability=0.5, input_set=name, model_family="logistic_regression")
                            for name in ("glucose", "wearable", "combined")], ignore_index=True)
        validate_paired(paired, data, assignments)
        changed_label, changed_fold = paired.copy(), paired.copy()
        changed_label.loc[0, protocol.TARGET] = 1 - changed_label.loc[0, protocol.TARGET]
        changed_fold.loc[0, "test_fold"] = 2
        for broken in (paired.iloc[1:], pd.concat([paired, paired.iloc[[0]]]), changed_label, changed_fold):
            with self.assertRaises(ValueError):
                validate_paired(broken, data, assignments)


class FrozenEvaluationTests(unittest.TestCase):
    def test_feasibility_source_paths_accept_windows_separators(self):
        #### Only entry points may change; construction files stay protected in either format.
        sources = {"onset_prediction\\main.py": "entry", "onset_prediction/README.md": "docs",
                   "onset_prediction\\outcomes.py": "labels", "preprocessing.py": "quality"}
        self.assertEqual(construction_sources(sources), {
            "onset_prediction\\outcomes.py": "labels", "preprocessing.py": "quality"})

    def test_synthetic_saved_folds_complete_all_three_input_sets(self):
        #### The full fitting loop yields fifteen models and one paired prediction per key.
        from onset_prediction.folds import build_folds
        data, links = synthetic_cohort()
        test, validation, _, _ = build_folds(data, links)
        with tempfile.TemporaryDirectory() as folder:
            tables = fit_test_folds(data, test, validation, Path(folder))
            self.assertEqual(len(list(Path(folder).glob("*.joblib"))), 15)
        self.assertEqual(len(tables["test_predictions"]), 3 * len(data))
        self.assertEqual(len(tables["selected_settings"]), 15)
        validate_paired(tables["test_predictions"], data, test)

    def test_freeze_rejects_changed_inputs_and_frozen_files(self):
        #### A frozen evaluation must not silently accept altered model data or settings.
        from onset_prediction.folds import build_folds
        data, links = synthetic_cohort()
        data["prior_day_source_day"] = data.day_in_study - 1
        data["prediction_cutoff_seconds"] = 14 * 3600
        data["main_sleep_end_seconds"] = 8 * 3600
        for name in protocol.record()["feature_sets"]["combined"][:5]:
            data[f"{name}__last_reference_day"] = data.day_in_study - 1
            data[f"{name}__last_trend_day"] = data.day_in_study - 1
        test, validation, _, _ = build_folds(data, links)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, processed, previous = root / "code", root / "processed", root / "previous"
            for path in (code, processed, previous):
                path.mkdir()
            (root / "EVALUATION_PROTOCOL.md").write_text("synthetic protocol", encoding="utf-8")
            marker = code / "marker.py"
            marker.write_text("unchanged", encoding="utf-8")
            inputs = {str(marker): file_sha256(marker)}
            links.to_csv(processed / "reference_links.csv", index=False)
            test.to_csv(previous / "participant_folds.csv", index=False)
            validation.to_csv(previous / "validation_fold_assignments.csv", index=False)
            frozen = root / "results/frozen"
            with patch.object(protocol, "ROOT", code), patch("onset_prediction.freezing.source_hashes", return_value={}), \
                 patch("onset_prediction.freezing.load_feasibility", return_value=(data, inputs)):
                freeze_logistic(processed, previous, frozen)
                loaded = load_frozen(frozen)
                self.assertEqual(len(loaded[0]), len(data))
                self.assertEqual(len(loaded[3]), 2000)
                marker.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_frozen(frozen)
                marker.write_text("unchanged", encoding="utf-8")
                with (frozen / "observations.csv").open("a", encoding="utf-8") as handle:
                    handle.write("changed")
                with self.assertRaises(ValueError):
                    load_frozen(frozen)


if __name__ == "__main__":
    unittest.main()
