"""Focused tests for the locked participant-held-out primary evaluation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import daily_metrics as metrics
import daily_models as models
import onset_alerts as onset
import participant_folds as folds
import primary_evaluation as evaluation
import primary_protocol as config
from model_inputs import FEATURE_SETS


def synthetic_observations(participants=10):
    """Create a small cohort with every outcome class and onset event represented."""
    rows = []
    for participant in range(1, participants + 1):
        for day in range(1, 7):
            luteal = int(day in (3, 4))
            rows.append({
                "id": participant,
                "study_interval": 2022,
                "day_in_study": day,
                "phase": "luteal" if luteal else "follicular",
                "luteal_status": luteal,
                "exact_luteal_onset": int(day == 3),
                "uncertain_luteal_entry": False,
                "prior_day_source_day": day - 1,
                "sleeping_hr_source_day": day,
                "prediction_cutoff_seconds": day * 86400 + 14 * 3600,
                "main_sleep_start_seconds": (day - 1) * 86400 + 23 * 3600,
                "main_sleep_end_seconds": day * 86400 + 7 * 3600,
                "valid_wearable_count": 3,
                "aligned_sleeping_hr": True,
                "aligned_hrv": True,
                "aligned_temperature": True,
                "sleeping_hr_bpm": 64.0 - 3.0 * luteal + participant / 100,
                "glucose_median_mmol_l": 5.0 + 0.3 * luteal + participant / 1000,
                "glucose_cv_percent": 15.0 + 2.0 * luteal + participant / 100,
                "log_median_rmssd_ms": np.log(30.0 + 2.0 * luteal + participant / 100),
                "skin_temperature_c": 33.0 + 0.2 * luteal + participant / 1000,
                "sleeping_hr_bpm_missing": False,
                "log_median_rmssd_ms_missing": False,
                "skin_temperature_c_missing": False,
            })
    return pd.DataFrame(rows)


def synthetic_references(observations):
    starts = observations.loc[observations.exact_luteal_onset.eq(1), config.KEY].copy()
    extra = pd.DataFrame([{"id": 1, "study_interval": 2022, "day_in_study": 7}])
    return pd.concat([starts, extra], ignore_index=True).sort_values(config.KEY).reset_index(drop=True)


def prediction_rows(days, probabilities):
    return pd.DataFrame({
        "id": [1] * len(days),
        "study_interval": [2022] * len(days),
        "day_in_study": days,
        "probability": probabilities,
        "luteal_status": [0] * len(days),
    })


class PrimaryEvaluationTests(unittest.TestCase):
    def test_sklearn_metrics_and_logistic_fit_are_finite(self):
        self.assertEqual(metrics.average_precision([0, 1, 1], [0.1, 0.9, 0.8]), 1.0)
        self.assertEqual(metrics.roc_auc([0, 1, 1], [0.1, 0.9, 0.8]), 1.0)
        self.assertEqual(metrics.roc_auc([0, 1], [0.5, 0.5]), 0.5)
        model = models.create_logistic_model(1.0).fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]),
        )
        coefficients = np.r_[model.intercept_, model.coef_[0]]
        iterations = model.n_iter_[0]
        self.assertTrue(np.isfinite(coefficients).all())
        self.assertGreater(iterations, 0)

    def test_sklearn_c_matches_sum_loss_objective_for_the_prespecified_grid(self):
        features = np.array([[-1.8], [-0.6], [0.2], [0.5], [1.9], [2.5]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        design = np.column_stack([np.ones(len(features)), features])
        for candidate_c in config.LOGISTIC_C_VALUES:
            with self.subTest(C=candidate_c):
                model = models.create_logistic_model(candidate_c).fit(features, labels)
                self.assertEqual(model.C, candidate_c)
                coefficients = np.r_[model.intercept_, model.coef_[0]]
                probabilities = 1 / (1 + np.exp(-(design @ coefficients)))
                gradient = design.T @ (probabilities - labels)
                gradient[1:] += coefficients[1:] / candidate_c
                self.assertLess(np.max(np.abs(gradient)), 1e-6)

    def test_pipeline_scaling_uses_training_participants_only(self):
        observations = synthetic_observations(participants=4)
        training = observations.loc[observations.id.le(3)].copy()
        evaluation = observations.loc[observations.id.eq(4)].copy()
        feature_columns = FEATURE_SETS["combined"]
        evaluation.loc[:, feature_columns] = evaluation.loc[:, feature_columns] + 1000

        _, parameters = models.fit_and_predict(
            training, evaluation, feature_columns, inverse_penalty_strength=1.0,
        )

        expected_means = training[feature_columns].mean().to_dict()
        for feature, expected in expected_means.items():
            self.assertAlmostEqual(parameters["feature_means"][feature], expected)

    def test_fold_assignments_are_reproducible_and_participant_disjoint(self):
        observations = synthetic_observations()
        references = synthetic_references(observations)
        first = folds.build_fold_assignments(observations, references)
        second = folds.build_fold_assignments(observations, references)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        outer, inner, feasibility = first
        self.assertEqual(set(outer.outer_test_fold), {1, 2, 3, 4, 5})
        self.assertEqual(len(inner), 4 * len(observations.id.unique()))
        self.assertTrue((feasibility.luteal_days > 0).all())
        self.assertTrue((feasibility.non_luteal_days > 0).all())
        self.assertTrue((feasibility.reference_starts > 0).all())
        for outer_fold in range(1, 6):
            test_ids = set(outer.loc[outer.outer_test_fold.eq(outer_fold), "id"])
            validation_ids = set(inner.loc[inner.outer_test_fold.eq(outer_fold), "id"])
            self.assertFalse(test_ids & validation_ids)

    def test_alerts_break_pending_sequences_but_reset_only_below_threshold(self):
        pending = prediction_rows([1, 3, 4], [0.9, 0.9, 0.9])
        alerts = onset.generate_alerts(pending, threshold=0.5, consecutive_days=2)
        self.assertEqual(alerts.alert_day_in_study.tolist(), [4])

        active = prediction_rows([1, 2, 4, 5, 6], [0.9, 0.9, 0.9, 0.1, 0.9])
        alerts = onset.generate_alerts(active, threshold=0.5, consecutive_days=1)
        self.assertEqual(alerts.alert_day_in_study.tolist(), [1, 6])

    def test_full_alignment_start_is_evaluable_and_tie_prefers_earlier_alert(self):
        predictions = prediction_rows([3, 4, 5], [0.9, 0.1, 0.9])
        references = pd.DataFrame([
            {"id": 1, "study_interval": 2022, "day_in_study": 4},
            {"id": 1, "study_interval": 2022, "day_in_study": 7},
        ])
        _, reference_events, _, metrics = onset.evaluate_alerts(
            predictions, references, threshold=0.5, consecutive_days=1,
        )
        first = reference_events.loc[reference_events.reference_day_in_study.eq(4)].iloc[0]
        second = reference_events.loc[reference_events.reference_day_in_study.eq(7)].iloc[0]
        self.assertTrue(first.evaluable)
        self.assertEqual(first.alert_day_in_study, 3)
        self.assertTrue(second.evaluable)
        self.assertEqual(metrics["evaluable_reference_starts"], 2)

    def test_alert_tie_breaker_prefers_higher_threshold_and_two_days(self):
        base = prediction_rows([1, 2, 3], [0.0, 0.0, 0.0])
        base["validation_fold"] = 1
        predictions = base.copy()
        for fold in [2, 3]:
            copy = base.copy()
            copy["id"] = fold
            copy["validation_fold"] = fold
            predictions = pd.concat([predictions, copy], ignore_index=True)
        references = pd.DataFrame([
            {"id": 1, "study_interval": 2022, "day_in_study": 2},
            {"id": 2, "study_interval": 2022, "day_in_study": 2},
            {"id": 3, "study_interval": 2022, "day_in_study": 2},
        ])
        threshold, days, _ = onset.select_alert_rule(predictions, references)
        self.assertEqual(threshold, 0.9)
        self.assertEqual(days, 2)

    def test_primary_evaluation_saves_paired_outputs_and_training_only_parameters(self):
        observations = synthetic_observations()
        references = synthetic_references(observations)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, report, output = root / "processed", root / "report", root / "results"
            processed.mkdir()
            report.mkdir()
            observations.to_csv(processed / "observations_2022.csv", index=False)
            alignment = observations[config.KEY + ["luteal_status", "exact_luteal_onset"]].copy()
            extra = references.loc[references.day_in_study.eq(7)].assign(luteal_status=1, exact_luteal_onset=1)
            alignment = pd.concat([alignment, extra], ignore_index=True)
            alignment.to_csv(processed / "prediction_alignment_14_2022.csv", index=False)
            (report / "run_config.json").write_text(json.dumps({
                "features_ready": True,
                "prediction_cutoff": "14:00:00",
            }), encoding="utf-8")

            with patch.object(config, "BOOTSTRAP_RESAMPLES", 12):
                result = main.run_primary_evaluation(processed, report, output)

            self.assertEqual(result["participants"], 10)
            predictions = pd.read_csv(output / "model_predictions.csv")
            self.assertEqual(len(predictions), len(observations) * len(FEATURE_SETS))
            self.assertFalse(predictions.duplicated(["input_set"] + config.KEY).any())
            for input_set in FEATURE_SETS:
                actual = predictions.loc[predictions.input_set.eq(input_set), config.KEY].sort_values(config.KEY)
                expected = observations[config.KEY].sort_values(config.KEY)
                pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
            self.assertTrue(predictions.probability.between(0, 1).all())
            self.assertEqual(len(pd.read_csv(output / "bootstrap_metrics.csv")), 12)
            self.assertTrue((output / "evaluation_protocol.json").exists())
            protocol = json.loads((output / "evaluation_protocol.json").read_text())
            self.assertEqual(protocol["software_versions"]["scikit-learn"], evaluation.software_versions()["scikit-learn"])
            self.assertIn("LogisticRegression", protocol["model_implementation"])
            self.assertEqual(protocol["version"], config.PRIMARY_PROTOCOL_VERSION)
            self.assertEqual(protocol["c_convention"], "scikit_learn_standard")
            self.assertEqual(protocol["sklearn_c_conversion"], "identity: sklearn_C = protocol_C")
            self.assertTrue((output / "validation_predictions.csv").exists())

            parameters = pd.read_csv(output / "fitted_parameters.csv")
            self.assertTrue(parameters.parameter_category.eq("feature_means").any())
            medians = parameters.loc[parameters.parameter_category.eq("wearable_medians")]
            self.assertTrue(medians.groupby(["outer_test_fold", "parameter"]).value.nunique().eq(1).all())
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "complete")
            self.assertTrue((output / "onset_summary.csv").is_file())
            manifest = json.loads((output / "provenance.json").read_text())
            self.assertIn("daily_models.py", manifest["sources"])
            settings = pd.read_csv(output / "selected_settings.csv")
            self.assertEqual(len(settings), 15)
            self.assertTrue(set(settings.inverse_penalty_strength).issubset(config.LOGISTIC_C_VALUES))
            fitted_c = parameters.loc[parameters.parameter_category.eq("sklearn_C")].merge(
                settings, on=["outer_test_fold", "input_set"], validate="one_to_one",
            )
            np.testing.assert_array_equal(fitted_c.value, fitted_c.inverse_penalty_strength)


if __name__ == "__main__":
    unittest.main()
