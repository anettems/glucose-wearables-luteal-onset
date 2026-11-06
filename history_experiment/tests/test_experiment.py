#### Check the calendar contrast, eligibility boundaries, training isolation and paired scoring.
#### Synthetic inputs give known expected values without using study outcomes to set the rules.

import json
import unittest

import numpy as np
import pandas as pd

from history_experiment.analysis import VARIANTS, bootstrap, fit_variant, pair_predictions, paired_metrics
from history_experiment.features import FEATURE, build_history
from model_inputs import GLUCOSE_FEATURES, WEARABLE_FEATURES
from onset_prediction.logistic import predict_logistic, select_strength
from onset_prediction.protocol import TARGET, feature_columns


def timeline():
    #### Give the oldest, middle and recent weeks distinct glucose levels and include gaps.
    days = [day for day in range(1, 25) if day not in (4, 11, 19)]
    data = pd.DataFrame({'id': 1, 'study_interval': 2022, 'day_in_study': days,
                         'prior_day_source_day': np.array(days) - 1})
    for column in list(GLUCOSE_FEATURES) + list(WEARABLE_FEATURES):
        data[column] = np.select([data.day_in_study.le(7), data.day_in_study.le(14),
                                  data.day_in_study.le(21)], [4., 8., 6.], default=90.)
    return data


class ExperimentTests(unittest.TestCase):
    #### Exercise the decisions specific to the new feature and matched comparison.

    def test_calendar_and_timing(self):
        #### Day 22 uses 1–7 and 15–21, excluding current and later aligned glucose.
        data = timeline()
        other = data.assign(id=2, glucose_median_mmol_l=np.nan)
        windows = build_history(pd.concat([data, other], ignore_index=True))
        rows = windows.loc[windows.day_in_study.eq(22)].set_index('id')
        self.assertEqual(rows.loc[1, FEATURE], 2.)
        self.assertEqual(rows.loc[1, 'history_days'], 18)
        self.assertEqual(rows.loc[1, 'oldest_week_days'], 6)
        self.assertEqual(rows.loc[1, 'recent_week_days'], 6)
        self.assertTrue(pd.isna(rows.loc[2, FEATURE]))

    def test_eligibility_requirements(self):
        #### Failing either end week or the overall count leaves the feature unavailable.
        for missing in ([1, 2, 3], [15, 16, 17], [1, 8, 9, 10]):
            data = timeline().loc[lambda frame: ~frame.day_in_study.isin(missing)]
            row = build_history(data).loc[lambda frame: frame.day_in_study.eq(22)].iloc[0]
            self.assertTrue(pd.isna(row[FEATURE]))

    def test_training_only_and_no_glucose_replacement(self):
        #### Verify the added predictor uses the training scale and fails when absent.
        train = pd.DataFrame({'id': [1, 1, 2, 2], 'study_interval': 2022,
                              'day_in_study': [1, 2, 1, 2], TARGET: [0, 1, 0, 1]})
        for column in feature_columns('combined'):
            train[column] = [1., 2., 3., 4.]
        train[FEATURE] = [0., 2., 4., 6.]
        fit = fit_variant(train, VARIANTS[1], 1.)
        assessment = train.iloc[:1].assign(id=3, **{FEATURE: 300.})
        self.assertEqual(len(predict_logistic(fit, assessment)), 1)
        self.assertEqual(fit.pipeline.named_steps['standardscaler'].mean_[-1], 3.)
        with self.assertRaises(ValueError):
            predict_logistic(fit, assessment.assign(**{FEATURE: np.nan}))
        with self.assertRaises(ValueError):
            predict_logistic(fit, train)

    def test_paired_keys_and_participant_bootstrap(self):
        #### Verify a missing prediction fails and each draw keeps all days from one person.
        data = pd.DataFrame({'id': [1, 1, 2, 2], 'study_interval': 2022,
                             'day_in_study': [1, 2, 1, 2], TARGET: [0, 1, 0, 1]})
        test = pd.DataFrame({'id': [1, 2], 'test_fold': [1, 2]})
        keys = data.merge(test, on='id')
        predictions = pd.concat([keys.assign(variant=variant, probability=scores)
                                 for variant, scores in zip(VARIANTS, ([.2, .8, .2, .8], [.8, .2, .2, .8]))])
        paired = pair_predictions(predictions, data, test)
        with self.assertRaises(AssertionError):
            pair_predictions(predictions.iloc[:-1], data, test)
        draws = pd.DataFrame({'replicate': [1, 2], 'participant_draw': [json.dumps([1, 1]), '[2, 2]']})
        sampled = bootstrap(paired, draws)
        self.assertEqual(sampled.difference_average_precision.tolist(), [-.5, 0.])
        self.assertLess(paired_metrics(paired)['difference']['average_precision'], 0.)

    def test_tuning_tie(self):
        #### Use the existing smaller-penalty-parameter tie rule with all three validation folds.
        scores = pd.DataFrame([(c, fold, .3) for c in (.01, .1, 1., 10.) for fold in (1, 2, 3)],
                              columns=['C', 'validation_fold', 'average_precision'])
        self.assertEqual(select_strength(scores), .01)
