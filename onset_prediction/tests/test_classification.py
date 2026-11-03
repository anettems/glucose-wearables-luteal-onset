#### Test daily classification metrics and validation-only probability threshold selection.
#### Synthetic outcomes check threshold equality, ties, participant separation and paired draws.

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_prediction import protocol
from onset_prediction.classification import (bootstrap_classification, classification_interval,
                                             classify_family, confusion_counts, count_metrics,
                                             select_threshold, summarise, validate_validation)
from onset_prediction.metrics import INPUT_SETS


def synthetic_saved():
    #### Give five participants separate test folds and deterministic perfect probabilities.
    observations = pd.DataFrame([{"id": person, "study_interval": 2022, "day_in_study": day,
                                  protocol.TARGET: day % 2} for person in range(1, 6)
                                 for day in range(1, 5)])
    assignments = pd.DataFrame({"id": range(1, 6), "test_fold": range(1, 6)})
    test, validation, settings, validation_folds = [], [], [], []
    for fold in range(1, 6):
        development = [person for person in range(1, 6) if person != fold]
        mapping = {person: index % 3 + 1 for index, person in enumerate(development)}
        validation_folds.extend({"id": person, "test_fold": fold, "validation_fold": value}
                                for person, value in mapping.items())
        for inputs in INPUT_SETS:
            settings.append({"model_family": "logistic_regression", "input_set": inputs,
                             "test_fold": fold, "C": 0.1})
            for row in observations.to_dict("records"):
                common = {**row, "model_family": "logistic_regression", "input_set": inputs,
                          "test_fold": fold, "probability": 0.2 + 0.6 * row[protocol.TARGET]}
                if row["id"] == fold:
                    test.append(common)
                else:
                    validation.append({**common, "C": 0.1, "validation_fold": mapping[row["id"]]})
                    #### A second, unselected setting must not influence the chosen threshold.
                    validation.append({**common, "C": 10, "probability": 0.5,
                                       "validation_fold": mapping[row["id"]]})
    return (observations, assignments, pd.DataFrame(validation_folds),
            (pd.DataFrame(test), pd.DataFrame(validation), pd.DataFrame(settings)))


class ClassificationTests(unittest.TestCase):
    def test_confusion_metrics_match_independent_definitions(self):
        #### Check all four confusion counts and each requested metric against scikit-learn.
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        predicted = np.array([0, 0, 1, 0, 1, 1, 1])
        counts = confusion_counts(labels, predicted)
        np.testing.assert_array_equal(counts, [2, 1, 1, 3])
        scores = count_metrics(counts)
        self.assertAlmostEqual(scores["accuracy"], accuracy_score(labels, predicted))
        self.assertAlmostEqual(scores["balanced_accuracy"], balanced_accuracy_score(labels, predicted))
        self.assertAlmostEqual(scores["sensitivity"], recall_score(labels, predicted))
        self.assertAlmostEqual(scores["precision"], precision_score(labels, predicted))
        self.assertAlmostEqual(scores["specificity"], 2 / 3)
        self.assertTrue(np.isnan(count_metrics([3, 0, 4, 0])["precision"]))
        self.assertEqual(count_metrics([3, 0, 4, 0])["balanced_accuracy"], 0.5)
        self.assertTrue(np.isnan(count_metrics([3, 0, 0, 0])["sensitivity"]))
        self.assertEqual(classification_interval([np.nan] * 3)["valid_resamples"], 0)

    def test_threshold_ties_equality_and_invalid_validation(self):
        #### The higher tied threshold wins, and equality still gives a positive prediction.
        data = pd.DataFrame([{"id": fold, "study_interval": 2022, "day_in_study": day,
                              "validation_fold": fold, protocol.TARGET: day % 2,
                              "probability": 0.2 + 0.6 * (day % 2)}
                             for fold in range(1, 4) for day in range(1, 5)])
        threshold, score, curve = select_threshold(data)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(score, 1)
        self.assertEqual(len(curve), 303)
        self.assertEqual(int((data.probability >= threshold).sum()), 6)
        constant = data.assign(probability=0.5)
        self.assertEqual(select_threshold(constant)[0], 1)
        with self.assertRaises(ValueError):
            select_threshold(pd.concat([data, data.iloc[:1]]))
        with self.assertRaises(ValueError):
            select_threshold(data[data.validation_fold.ne(3)])

    def test_selected_setting_and_threshold_ignore_test_outcomes(self):
        #### Test-label changes cannot alter thresholds, but they do alter held-out metrics.
        observations, assignments, folds, saved = synthetic_saved()
        result, thresholds, _ = classify_family("logistic_regression", "C", saved,
                                                observations, assignments, folds)
        self.assertEqual(len(result), 60)
        self.assertTrue(thresholds.threshold.eq(0.8).all())
        self.assertTrue(summarise(result)[0].accuracy.eq(1).all())
        changed = saved[0].copy()
        changed.loc[changed.test_fold.eq(1), protocol.TARGET] = 1 - changed.loc[
            changed.test_fold.eq(1), protocol.TARGET]
        altered_observations = observations.copy()
        altered_observations.loc[altered_observations.id.eq(1), protocol.TARGET] = 1 - altered_observations.loc[
            altered_observations.id.eq(1), protocol.TARGET]
        #### Isolate fold one: its validation cohort is unchanged by test-label alterations.
        data = saved[1].query("test_fold == 1 and input_set == 'combined' and C == 0.1")
        validate_validation(data, altered_observations, assignments, folds, 1)
        self.assertEqual(select_threshold(data)[0], thresholds.iloc[0].threshold)
        bad = data.copy()
        bad.loc[bad.index[0], "id"] = 1
        with self.assertRaises(ValueError):
            validate_validation(bad, observations, assignments, folds, 1)

    def test_bootstrap_repeats_whole_participants_and_pairs_models(self):
        #### Repeated participant selections retain every day and use the same draws per model.
        observations, assignments, folds, saved = synthetic_saved()
        result, _, _ = classify_family("logistic_regression", "C", saved, observations, assignments, folds)
        forest = result.assign(model_family="random_forest")
        forest.loc[forest.id.eq(1), "predicted_positive"] = 0
        predictions = pd.concat([result, forest], ignore_index=True)
        selected = [1, 1, 1, 2, 3]
        draws = pd.DataFrame({"replicate": [1], "participant_draw": [json.dumps(selected)]})
        boot = bootstrap_classification(predictions, draws)
        for row in boot.itertuples(index=False):
            data = predictions.loc[predictions.model_family.eq(row.model_family)
                                   & predictions.input_set.eq(row.input_set)]
            repeated = pd.concat([data[data.id.eq(person)] for person in selected])
            expected = count_metrics(confusion_counts(repeated[protocol.TARGET], repeated.predicted_positive))
            self.assertEqual(row.accuracy, expected["accuracy"])
            self.assertEqual(row.precision, expected["precision"])


if __name__ == "__main__":
    unittest.main()
