#### Fit logistic regression directly to the three-day onset outcome using saved folds.
#### Replace missing wearable predictors and standardise inputs using training participants
#### only. Compare the four agreed C values by mean validation average precision.
#### Return probabilities, fitted models and parameter records; never create alerts.

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from onset_prediction import protocol
from onset_prediction.folds import validate_assignment
from onset_prediction.metrics import probability_metrics, validate_scores
from onset_prediction.outcomes import validate_keys


@dataclass
class FittedLogistic:
    #### Retain the complete fitted transformation and model with its training identifiers.
    pipeline: object
    columns: list
    medians: pd.Series
    training_ids: tuple
    training_days: int
    input_set: str


def input_matrix(data, columns, medians):
    #### Replace only wearable predictors with supplied training medians, never glucose inputs.
    matrix = data[columns].copy()
    glucose = protocol.feature_columns("glucose")
    if matrix[[name for name in columns if name in glucose]].isna().any().any():
        raise ValueError("Glucose inputs must be observed and are not replaced")
    wearable = [name for name in columns if name not in glucose]
    matrix[wearable] = matrix[wearable].fillna(medians)
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("A model input or training replacement value is not finite")
    return matrix.to_numpy(dtype=float)


def create_classifier(strength):
    #### Express the agreed L2 penalty using the installed scikit-learn l1_ratio interface.
    settings = protocol.record()["models"]["logistic_regression"]
    return LogisticRegression(C=strength, l1_ratio=0, solver=settings["solver"],
                              fit_intercept=True, class_weight=None, max_iter=settings["max_iter"],
                              tol=settings["tol"], random_state=protocol.SEED)


def fit_logistic(training, input_set, strength):
    #### Fit replacement, population-standard-deviation scaling and regression on training only.
    validate_keys(training)
    labels = training[protocol.TARGET].to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError("Logistic training requires both binary outcome classes")
    columns = protocol.feature_columns(input_set)
    medians = training[columns].median()
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        raise ValueError("A predictor has no finite observed training median")
    matrix = input_matrix(training, columns, medians)
    pipeline = make_pipeline(StandardScaler(), create_classifier(strength))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(matrix, labels.astype(int))
        except ConvergenceWarning as error:
            raise RuntimeError("Logistic regression failed to converge") from error
    return FittedLogistic(pipeline, columns, medians, tuple(sorted(training.id.unique())),
                          len(training), input_set)


def predict_logistic(fitted, assessment):
    #### Apply the fitted steps to separate participants and return class-one probabilities.
    validate_keys(assessment)
    if set(fitted.training_ids) & set(assessment.id):
        raise ValueError("Training and assessment participants overlap")
    matrix = input_matrix(assessment, fitted.columns, fitted.medians)
    classifier = fitted.pipeline.named_steps["logisticregression"]
    positive = int(np.flatnonzero(classifier.classes_ == 1)[0])
    probabilities = fitted.pipeline.predict_proba(matrix)[:, positive]
    validate_scores(np.zeros(len(probabilities), dtype=int), probabilities)
    return probabilities


def parameter_rows(fitted):
    #### Save medians, scaling and coefficients in predictor order for reproducibility checks.
    scaler = fitted.pipeline.named_steps["standardscaler"]
    classifier = fitted.pipeline.named_steps["logisticregression"]
    rows = []
    for index, name in enumerate(fitted.columns):
        rows.append({"feature": name, "training_median": float(fitted.medians[name]),
                     "replacement_allowed": name in protocol.feature_columns("wearable"),
                     "mean": float(scaler.mean_[index]), "scale": float(scaler.scale_[index]),
                     "coefficient": float(classifier.coef_[0, index]),
                     "intercept": float(classifier.intercept_[0]),
                     "iterations": int(classifier.n_iter_[0]), "C": float(classifier.C),
                     "training_days": fitted.training_days,
                     "training_participants": len(fitted.training_ids)})
    return pd.DataFrame(rows)


def select_strength(scores):
    #### Select by the equally weighted mean fold AP, resolving exact ties with smaller C.
    expected = protocol.record()["models"]["logistic_regression"]["C"]
    if scores.duplicated(["C", "validation_fold"]).any() or set(scores.C) != set(expected):
        raise ValueError("Missing or duplicated candidate validation scores")
    for _, group in scores.groupby("C"):
        if set(group.validation_fold) != set(range(1, protocol.VALIDATION_FOLDS + 1)):
            raise ValueError("Every candidate needs every validation fold")
    if not np.isfinite(scores.average_precision).all():
        raise ValueError("Undefined validation average precision")
    means = scores.groupby("C", as_index=False).average_precision.mean()
    return float(means.sort_values(["average_precision", "C"], ascending=[False, True]).iloc[0].C)


def tune_logistic(development, assignments, input_set):
    #### Fit every candidate on two validation folds and assess the third, without test inputs.
    validate_assignment(development, assignments, "validation_fold", protocol.VALIDATION_FOLDS)
    scores, predictions, parameters = [], [], []
    for strength in protocol.record()["models"]["logistic_regression"]["C"]:
        for fold in range(1, protocol.VALIDATION_FOLDS + 1):
            ids = assignments.loc[assignments.validation_fold.eq(fold), "id"]
            training = development.loc[~development.id.isin(ids)]
            assessing = development.loc[development.id.isin(ids)]
            fitted = fit_logistic(training, input_set, strength)
            probabilities = predict_logistic(fitted, assessing)
            part = assessing[protocol.KEY + [protocol.TARGET]].assign(
                probability=probabilities, C=strength, validation_fold=fold)
            predictions.append(part)
            scores.append({"C": strength, "validation_fold": fold, **probability_metrics(part)})
            parameters.append(parameter_rows(fitted).assign(validation_fold=fold))
    scores = pd.DataFrame(scores)
    return (select_strength(scores), scores, pd.concat(predictions, ignore_index=True),
            pd.concat(parameters, ignore_index=True))
