#### Test fixed forest settings, training-only replacement and participant-disjoint tuning.
#### Small synthetic cohorts verify probability averaging, leaf constraints and reproducibility.
#### Temporary snapshots test exact sharing with logistic regression and freeze integrity.

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_prediction import protocol
from onset_prediction.evaluation import check_shared_replacements, evaluate_one, probability_tables
from onset_prediction.folds import build_folds
from onset_prediction.forest import (create_forest, fit_forest, parameter_rows, predict_forest,
                                     select_leaf_size, tune_forest)
from onset_prediction.forest_freezing import SHARED_FILES, freeze_forest, load_forest_frozen
from onset_prediction.freezing import freeze_logistic
from onset_prediction.logistic import fit_logistic, input_matrix
from onset_prediction.metrics import bootstrap_metrics, participant_draws, summary_tables, validate_paired
from preprocessing import file_sha256
from test_feasibility import synthetic_cohort
from test_logistic import model_data


class ForestTests(unittest.TestCase):
    def test_fixed_settings_leaf_constraints_and_determinism(self):
        #### Trees respect depth/leaf limits and repeat exactly with the agreed seed.
        training, assessing = model_data([1, 2, 3, 4]), model_data([5])
        fitted = fit_forest(training, "combined", 5)
        params = fitted.estimator.get_params()
        self.assertEqual(params["n_estimators"], 100)
        self.assertEqual(params["max_depth"], 5)
        self.assertEqual(params["max_features"], "sqrt")
        self.assertEqual(params["random_state"], 20261020)
        self.assertEqual(params["n_jobs"], 1)
        self.assertTrue(params["bootstrap"])
        self.assertIsNone(params["class_weight"])
        self.assertIsNone(params["max_samples"])
        for tree in fitted.estimator.estimators_:
            self.assertLessEqual(tree.tree_.max_depth, 5)
            leaves = tree.tree_.children_left == -1
            self.assertTrue((tree.tree_.n_node_samples[leaves] >= 5).all())
            self.assertEqual(tree.tree_.weighted_n_node_samples[0], len(training))
        repeat = fit_forest(training, "combined", 5)
        np.testing.assert_array_equal(predict_forest(fitted, assessing), predict_forest(repeat, assessing))
        with self.assertRaises(ValueError):
            create_forest(2)

    def test_training_only_medians_match_logistic_and_other_inputs(self):
        #### Missing wearable replacement is shared and cannot use held-out values or labels.
        training, assessing = model_data([1, 2, 3, 4]), model_data([5])
        feature = "skin_temperature_c__deviation"
        training.loc[[0, 3, 7], feature] = np.nan
        assessing[feature] = np.nan
        fitted = fit_forest(training, "combined", 1)
        wearable = fit_forest(training, "wearable", 10)
        logistic = fit_logistic(training, "combined", 0.1)
        self.assertEqual(fitted.medians[feature], training[feature].median())
        pd.testing.assert_series_equal(fitted.medians, logistic.medians)
        for name in protocol.feature_columns("wearable"):
            self.assertEqual(fitted.medians[name], wearable.medians[name])
        before = fitted.medians.copy()
        predict_forest(fitted, assessing.assign(**{feature: 1e9}))
        pd.testing.assert_series_equal(fitted.medians, before)
        probabilities = predict_forest(fitted, assessing)
        np.testing.assert_array_equal(probabilities, predict_forest(
            fitted, assessing.assign(**{protocol.TARGET: 1 - assessing[protocol.TARGET]})))
        records = pd.concat([parameter_rows(fitted), parameter_rows(wearable)], ignore_index=True)
        check_shared_replacements(records.assign(test_fold=1))

    def test_probabilities_average_trees_on_unscaled_inputs(self):
        #### Forest probabilities equal the average of its individual tree probabilities.
        training, assessing = model_data([1, 2, 3, 4]), model_data([5])
        fitted = fit_forest(training, "glucose", 1)
        matrix = input_matrix(assessing, fitted.columns, fitted.medians)
        np.testing.assert_array_equal(matrix, assessing[fitted.columns].to_numpy())
        manual = np.mean([tree.predict_proba(matrix)[:, 1] for tree in fitted.estimator.estimators_], axis=0)
        probabilities = predict_forest(fitted, assessing)
        np.testing.assert_allclose(probabilities, manual, rtol=0, atol=1e-14)
        self.assertGreater(probabilities[assessing[protocol.TARGET].eq(1)].mean(),
                           probabilities[assessing[protocol.TARGET].eq(0)].mean())
        self.assertFalse(set(protocol.KEY + [protocol.TARGET]) & set(fitted.columns))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forest.joblib"
            joblib.dump(fitted, path)
            np.testing.assert_array_equal(probabilities, predict_forest(joblib.load(path), assessing))

    def test_invalid_inputs_and_participant_overlap(self):
        #### Refuse missing glucose, absent wearable training values and invalid outcome classes.
        training, assessing = model_data([1, 2, 3, 4]), model_data([5])
        for invalid in (training.assign(**{protocol.TARGET: 0}),
                        training.assign(**{protocol.TARGET: np.nan}),
                        training.assign(glucose_cv_percent=np.nan),
                        training.assign(skin_temperature_c=np.nan),
                        training.assign(skin_temperature_c=np.inf)):
            with self.assertRaises(ValueError):
                fit_forest(invalid, "combined", 1)
        fitted = fit_forest(training, "combined", 1)
        with self.assertRaises(ValueError):
            predict_forest(fitted, training)
        with self.assertRaises(ValueError):
            predict_forest(fitted, assessing.assign(glucose_cv_percent=np.nan))

    def test_leaf_selection_and_candidate_validation_predictions(self):
        #### Exact ties favour ten and each candidate covers every development key once.
        scores = pd.DataFrame([{"min_samples_leaf": leaf, "validation_fold": fold,
                                "average_precision": 0.5, "days": fold}
                               for leaf in (1, 5, 10) for fold in (1, 2, 3)])
        self.assertEqual(select_leaf_size(scores), 10)
        scores.loc[scores.min_samples_leaf.eq(5), "average_precision"] = [0.9, 0.4, 0.4]
        self.assertEqual(select_leaf_size(scores), 5)
        with self.assertRaises(ValueError):
            select_leaf_size(scores.iloc[1:])
        development = model_data(range(1, 7))
        assignments = pd.DataFrame({"id": range(1, 7), "validation_fold": [1, 2, 3, 1, 2, 3]})
        chosen, scores, predictions, parameters = tune_forest(development, assignments, "wearable")
        self.assertEqual(chosen, select_leaf_size(scores))
        self.assertEqual(len(scores), 9)
        self.assertEqual(len(parameters), 9 * 9)
        self.assertEqual(len(predictions), 3 * len(development))
        for _, frame in predictions.groupby("min_samples_leaf"):
            self.assertFalse(frame.duplicated(protocol.KEY).any())
        with self.assertRaises(ValueError):
            tune_forest(development, assignments.iloc[1:], "wearable")

    def test_shared_evaluation_and_metrics_use_forest_family(self):
        #### The common evaluator saves the selected forest and correctly labels its metrics.
        development, testing = model_data(range(1, 7)), model_data([7, 8])
        assignments = pd.DataFrame({"id": range(1, 7), "validation_fold": [1, 2, 3, 1, 2, 3]})
        with tempfile.TemporaryDirectory() as folder:
            tables = evaluate_one(development, testing, assignments, "glucose", 1,
                                  Path(folder), "random_forest")
            saved = joblib.load(Path(folder) / "test_fold_1_glucose.joblib")
            np.testing.assert_array_equal(tables["test_predictions"].probability, predict_forest(saved, testing))
        self.assertEqual(len(tables["selected_settings"]), 1)
        self.assertIn("min_samples_leaf", tables["selected_settings"])
        paired = pd.concat([tables["test_predictions"].assign(input_set=name)
                            for name in ("glucose", "wearable", "combined")], ignore_index=True)
        test_assignments = pd.DataFrame({"id": [7, 8], "test_fold": [1, 1]})
        validate_paired(paired, testing, test_assignments, "random_forest")
        with self.assertRaises(ValueError):
            validate_paired(paired, testing, test_assignments)
        draws = participant_draws(testing.id.unique(), resamples=4)
        bootstrap = bootstrap_metrics(paired, draws)
        summary, difference = summary_tables(paired, bootstrap, "random_forest")
        self.assertEqual(set(summary.model_family), {"random_forest"})
        self.assertEqual(difference.estimate.iloc[0], 0)
        self.assertEqual(set(probability_tables(paired, "random_forest")["calibration_bins"].model_family), {"random_forest"})


class ForestFreezeTests(unittest.TestCase):
    def test_forest_snapshot_shares_exact_logistic_inputs_and_protects_them(self):
        #### Copy the original cohort, folds and draws exactly, then reject changed shared files.
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
            (root / "EVALUATION_PROTOCOL.md").write_text("synthetic shared design", encoding="utf-8")
            marker = code / "marker.py"
            marker.write_text("unchanged", encoding="utf-8")
            inputs = {str(marker): file_sha256(marker)}
            links.to_csv(processed / "reference_links.csv", index=False)
            test.to_csv(previous / "participant_folds.csv", index=False)
            validation.to_csv(previous / "validation_fold_assignments.csv", index=False)
            logistic, forest = root / "results/logistic", root / "results/forest"
            with patch.object(protocol, "ROOT", code), \
                 patch("onset_prediction.freezing.source_hashes", return_value={}), \
                 patch("onset_prediction.forest_freezing.source_hashes", return_value={}), \
                 patch("onset_prediction.freezing.load_feasibility", return_value=(data, inputs.copy())), \
                 patch("onset_prediction.forest_freezing.load_feasibility", return_value=(data, inputs.copy())):
                freeze_logistic(processed, previous, logistic)
                freeze_forest(processed, previous, logistic, forest)
                loaded = load_forest_frozen(forest)
                self.assertEqual(len(loaded[0]), len(data))
                self.assertEqual(len(loaded[3]), 2000)
                for name in SHARED_FILES:
                    self.assertEqual(file_sha256(logistic / name), file_sha256(forest / name))
                with (logistic / "bootstrap_participants.csv").open("a", encoding="utf-8") as handle:
                    handle.write("changed")
                with self.assertRaises(ValueError):
                    load_forest_frozen(forest)


if __name__ == "__main__":
    unittest.main()
