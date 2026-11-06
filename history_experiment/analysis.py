#### Fit and compare two combined-input logistic models on the same held-out days.
#### Use saved participant folds and training-only preprocessing. Estimate paired uncertainty
#### by resampling whole participants after fitting all models.

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from history_experiment.features import FEATURE
from onset_prediction import protocol
from onset_prediction.logistic import (FittedLogistic, create_classifier, fit_logistic,
    input_matrix, parameter_rows, predict_logistic, select_strength)
from onset_prediction.metrics import METRICS, interval_record, probability_metrics
from onset_prediction.outcomes import validate_keys


VARIANTS = ('baseline', 'plus_21d_glucose')
IDENTITY = protocol.KEY + [protocol.TARGET, 'test_fold']


def fit_variant(training, variant, strength):
    #### Reuse the baseline fitter and apply the same settings to the extra observed predictor.
    if variant == 'baseline':
        return fit_logistic(training, 'combined', strength)
    if variant != VARIANTS[1]:
        raise ValueError('Unknown history experiment variant')
    validate_keys(training)
    labels = training[protocol.TARGET].to_numpy()
    if set(labels) != {0, 1} or not np.isfinite(training[FEATURE]).all():
        raise ValueError('Expected both classes and observed extra glucose values')

    columns = protocol.feature_columns('combined') + [FEATURE]
    medians = training[columns].median()
    if not np.isfinite(medians).all():
        raise ValueError('A predictor has no observed training median')
    matrix = input_matrix(training, columns, medians.drop(FEATURE))
    pipeline = make_pipeline(StandardScaler(), create_classifier(strength))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(matrix, labels)
    return FittedLogistic(pipeline, columns, medians.drop(FEATURE),
                          tuple(sorted(training.id.unique())), len(training), 'combined')


def fitted_parameters(fitted, training):
    #### Record the extra predictor's training median for audit purposes only.
    if FEATURE not in fitted.columns:
        return parameter_rows(fitted)
    from dataclasses import replace

    medians = fitted.medians.copy()
    medians[FEATURE] = training[FEATURE].median()
    return parameter_rows(replace(fitted, medians=medians))


def tune(development, assignments, variant):
    #### Score each setting on three held-out validation groups with equal fold weight.
    scores, predictions, parameters = [], [], []
    for fold, people in assignments.groupby('validation_fold'):
        mask = development.id.isin(people.id)
        training, assessment = development.loc[~mask], development.loc[mask]
        for strength in protocol.record()['models']['logistic_regression']['C']:
            fitted = fit_variant(training, variant, strength)
            part = assessment[protocol.KEY + [protocol.TARGET]].assign(
                probability=predict_logistic(fitted, assessment), C=strength, validation_fold=fold)
            predictions.append(part)
            parameters.append(fitted_parameters(fitted, training).assign(validation_fold=fold))
            scores.append({'C': strength, 'validation_fold': fold, **probability_metrics(part)})
    scores = pd.DataFrame(scores)
    return select_strength(scores), scores, pd.concat(predictions), pd.concat(parameters)


def fit_folds(data, test, validation):
    #### Tune independently, refit on development participants and predict the held-out fold.
    outputs = {name: [] for name in ('predictions', 'validation_predictions', 'tuning_scores',
               'selected_settings', 'fitted_parameters', 'validation_parameters')}
    for fold, people in test.groupby('test_fold'):
        mask = data.id.isin(people.id)
        development, assessment = data.loc[~mask], data.loc[mask]
        assignments = validation.loc[validation.test_fold.eq(fold)]
        for variant in VARIANTS:
            chosen, scores, predictions, parameters = tune(development, assignments, variant)
            fitted = fit_variant(development, variant, chosen)
            tables = dict(predictions=assessment[protocol.KEY + [protocol.TARGET]].assign(
                probability=predict_logistic(fitted, assessment)), validation_predictions=predictions,
                tuning_scores=scores, selected_settings=pd.DataFrame([{'C': chosen,
                    'mean_validation_AP': scores.loc[scores.C.eq(chosen), 'average_precision'].mean()}]),
                fitted_parameters=fitted_parameters(fitted, development), validation_parameters=parameters)
            for name, table in tables.items():
                outputs[name].append(table.assign(variant=variant, test_fold=fold))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in outputs.items()}


def pair_predictions(predictions, data, test):
    #### Require one held-out probability per variant and key, with identical labels and folds.
    if set(predictions.variant) != set(VARIANTS):
        raise ValueError('Missing or unexpected model variant')
    expected = data.merge(test, on='id', validate='many_to_one')[IDENTITY]
    paired = expected.sort_values(protocol.KEY).reset_index(drop=True)
    for variant in VARIANTS:
        rows = predictions.loc[predictions.variant.eq(variant)].sort_values(protocol.KEY)
        pd.testing.assert_frame_equal(rows[IDENTITY].reset_index(drop=True), paired[IDENTITY])
        probability_metrics(rows)
        paired[variant] = rows.probability.to_numpy()
    return paired


def paired_metrics(paired):
    #### Calculate both sets of scores and subtract baseline scores on the same days.
    scores = {variant: probability_metrics(paired.assign(probability=paired[variant]))
              for variant in VARIANTS}
    scores['difference'] = {metric: scores[VARIANTS[1]][metric] - scores[VARIANTS[0]][metric]
                            for metric in METRICS}
    return scores


def bootstrap(paired, draws):
    #### Keep all days of each sampled participant, including repeated selections.
    positions = paired.groupby('id').indices
    rows = []
    for draw in draws.itertuples(index=False):
        people = json.loads(draw.participant_draw)
        if len(people) != len(positions) or not set(people).issubset(positions):
            raise ValueError('Invalid participant bootstrap draw')
        sample = paired.iloc[np.concatenate([positions[person] for person in people])]
        scores = paired_metrics(sample)
        rows.append({'replicate': draw.replicate, **{
            f'{variant}_{metric}': values[metric] for variant, values in scores.items()
            for metric in METRICS}})
    return pd.DataFrame(rows)


def intervals(paired, replicates):
    #### Summarise pooled test estimates and paired percentile intervals.
    return pd.DataFrame([
        {'variant': variant, 'metric': metric, 'estimate': values[metric],
         **interval_record(replicates[f'{variant}_{metric}'])}
        for variant, values in paired_metrics(paired).items() for metric in METRICS])
