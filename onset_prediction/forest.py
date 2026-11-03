#### Fit random forests directly to the three-day onset outcome using saved participant folds.
#### Reuse training-only wearable replacement, with no feature scaling or class weighting.
#### Tune only minimum leaf size by mean validation AP; exact ties favour larger leaves.

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from onset_prediction import protocol
from onset_prediction.folds import validate_assignment
from onset_prediction.logistic import input_matrix
from onset_prediction.metrics import probability_metrics, validate_scores
from onset_prediction.outcomes import validate_keys


@dataclass
class FittedForest:
    #### Retain the forest and training medians with the identifiers used for fitting.
    estimator: object
    columns: list
    medians: pd.Series
    training_ids: tuple
    training_days: int
    input_set: str


def create_forest(leaf_size):
    #### Fix every forest setting except the agreed minimum observations per leaf.
    settings = protocol.record()["models"]["random_forest"]
    if leaf_size not in settings["min_samples_leaf"]:
        raise ValueError("Minimum leaf size must be one of 1, 5 and 10")
    return RandomForestClassifier(
        n_estimators=settings["n_estimators"], max_depth=settings["max_depth"],
        max_features=settings["max_features"], min_samples_leaf=leaf_size,
        criterion=settings["criterion"], min_samples_split=settings["min_samples_split"],
        bootstrap=settings["bootstrap"], max_samples=settings["max_samples"],
        class_weight=None, random_state=protocol.SEED, n_jobs=1,
        min_weight_fraction_leaf=0, max_leaf_nodes=None, min_impurity_decrease=0,
        oob_score=False, verbose=0, warm_start=False, ccp_alpha=0, monotonic_cst=None)


def fit_forest(training, input_set, leaf_size):
    #### Fit training medians and the fixed-depth forest, without standardising the inputs.
    validate_keys(training)
    labels = training[protocol.TARGET].to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError("Random-forest training requires both binary outcome classes")
    columns = protocol.feature_columns(input_set)
    medians = training[columns].median()
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        raise ValueError("A predictor has no finite observed training median")
    matrix = input_matrix(training, columns, medians)
    estimator = create_forest(leaf_size)
    estimator.fit(matrix, labels.astype(int))
    return FittedForest(estimator, columns, medians, tuple(sorted(training.id.unique())),
                        len(training), input_set)


def predict_forest(fitted, assessment):
    #### Return class-one probabilities for separate participants using the training medians.
    validate_keys(assessment)
    if set(fitted.training_ids) & set(assessment.id):
        raise ValueError("Training and assessment participants overlap")
    matrix = input_matrix(assessment, fitted.columns, fitted.medians)
    positive = int(np.flatnonzero(fitted.estimator.classes_ == 1)[0])
    probabilities = fitted.estimator.predict_proba(matrix)[:, positive]
    validate_scores(np.zeros(len(probabilities), dtype=int), probabilities)
    return probabilities


def parameter_rows(fitted):
    #### Record fitted replacement values and forest constraints alongside each predictor.
    rows = []
    for name in fitted.columns:
        rows.append({"feature": name, "training_median": float(fitted.medians[name]),
                     "replacement_allowed": name in protocol.feature_columns("wearable"),
                     "min_samples_leaf": fitted.estimator.min_samples_leaf,
                     "n_estimators": len(fitted.estimator.estimators_),
                     "max_depth": fitted.estimator.max_depth,
                     "maximum_fitted_depth": max(tree.tree_.max_depth for tree in fitted.estimator.estimators_),
                     "training_days": fitted.training_days,
                     "training_participants": len(fitted.training_ids)})
    return pd.DataFrame(rows)


def select_leaf_size(scores):
    #### Choose highest mean fold AP, resolving exact ties with the larger minimum leaf size.
    candidates = protocol.record()["models"]["random_forest"]["min_samples_leaf"]
    if scores.duplicated(["min_samples_leaf", "validation_fold"]).any() or set(scores.min_samples_leaf) != set(candidates):
        raise ValueError("Missing or duplicated leaf-size validation scores")
    for _, group in scores.groupby("min_samples_leaf"):
        if set(group.validation_fold) != set(range(1, protocol.VALIDATION_FOLDS + 1)):
            raise ValueError("Every candidate needs every validation fold")
    if not np.isfinite(scores.average_precision).all():
        raise ValueError("Undefined validation average precision")
    means = scores.groupby("min_samples_leaf", as_index=False).average_precision.mean()
    selected = means.sort_values(["average_precision", "min_samples_leaf"], ascending=[False, False])
    return int(selected.iloc[0].min_samples_leaf)


def tune_forest(development, assignments, input_set):
    #### Assess each leaf size on the same three participant-disjoint validation folds.
    validate_assignment(development, assignments, "validation_fold", protocol.VALIDATION_FOLDS)
    scores, predictions, parameters = [], [], []
    for leaf_size in protocol.record()["models"]["random_forest"]["min_samples_leaf"]:
        for fold in range(1, protocol.VALIDATION_FOLDS + 1):
            ids = assignments.loc[assignments.validation_fold.eq(fold), "id"]
            training = development.loc[~development.id.isin(ids)]
            assessing = development.loc[development.id.isin(ids)]
            fitted = fit_forest(training, input_set, leaf_size)
            probabilities = predict_forest(fitted, assessing)
            part = assessing[protocol.KEY + [protocol.TARGET]].assign(
                probability=probabilities, min_samples_leaf=leaf_size, validation_fold=fold)
            predictions.append(part)
            scores.append({"min_samples_leaf": leaf_size, "validation_fold": fold, **probability_metrics(part)})
            parameters.append(parameter_rows(fitted).assign(validation_fold=fold))
    scores = pd.DataFrame(scores)
    return (select_leaf_size(scores), scores, pd.concat(predictions, ignore_index=True),
            pd.concat(parameters, ignore_index=True))
