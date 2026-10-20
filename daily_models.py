"""Training-only preprocessing and validation-selected logistic regression."""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import primary_protocol as protocol
from daily_metrics import daily_metrics
from model_inputs import replace_missing_wearables


def create_logistic_model(inverse_penalty_strength, training_observations):
    """Build the fixed L2 estimator using the protocol's mean-loss scale."""
    # sklearn scales its regularisation against the summed sample loss; the
    # protocol defines C against the mean loss, so divide by training size.
    return LogisticRegression(
        C=inverse_penalty_strength / training_observations,
        solver="lbfgs",
        l1_ratio=0,
        fit_intercept=True,
        class_weight=None,
        max_iter=protocol.LOGISTIC_MAX_ITERATIONS,
        tol=protocol.LOGISTIC_TOLERANCE,
        random_state=protocol.PRIMARY_SEED,
    )


def fit_model(model, features, labels):
    """Stop the run when optimisation has not converged."""
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Logistic-regression training requires both outcome classes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return model.fit(features, labels)
        except ConvergenceWarning as error:
            raise RuntimeError("Logistic regression did not converge; no valid model was fitted") from error


def fit_and_predict(training, evaluation, feature_columns, inverse_penalty_strength):
    """Fit split-specific replacement, scaling and logistic regression, then predict."""
    training_filled, evaluation_filled, medians = replace_missing_wearables(training, evaluation)
    training_features = training_filled[feature_columns].to_numpy(dtype=float)
    evaluation_features = evaluation_filled[feature_columns].to_numpy(dtype=float)
    model = make_pipeline(
        StandardScaler(),
        create_logistic_model(inverse_penalty_strength, len(training_features)),
    )
    fit_model(model, training_features, training_filled.luteal_status.to_numpy(dtype=int))
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    probabilities = model.predict_proba(evaluation_features)[:, 1]
    parameters = {
        "wearable_medians": medians.to_dict(),
        "feature_means": dict(zip(feature_columns, scaler.mean_)),
        "feature_scales": dict(zip(feature_columns, scaler.scale_)),
        "coefficients": dict(zip(
            ["intercept"] + list(feature_columns),
            [classifier.intercept_[0]] + classifier.coef_[0].tolist(),
        )),
        "iterations": int(classifier.n_iter_[0]),
        "sklearn_C": float(classifier.C),
        "training_observations": len(training_features),
    }
    return probabilities, parameters


def tune_logistic_model(development, validation_assignments, feature_columns):
    """Choose C from participant-disjoint validation predictions."""
    if validation_assignments.id.duplicated().any():
        raise ValueError("Each development participant must have one validation-fold assignment")
    assignment = validation_assignments.set_index("id").validation_fold
    if set(development.id) != set(assignment.index):
        raise ValueError("Validation assignments do not cover the development participants")
    if set(assignment) != set(range(1, protocol.INNER_FOLDS + 1)):
        raise ValueError("Expected every prespecified validation fold")

    tuning_rows, candidate_predictions = [], {}
    for inverse_penalty_strength in protocol.LOGISTIC_C_VALUES:
        validation_parts = []
        for validation_fold in range(1, protocol.INNER_FOLDS + 1):
            validation_ids = assignment.loc[assignment.eq(validation_fold)].index
            validation = development.loc[development.id.isin(validation_ids)].copy()
            training = development.loc[~development.id.isin(validation_ids)].copy()
            probabilities, _ = fit_and_predict(
                training, validation, feature_columns, inverse_penalty_strength,
            )
            part = validation[protocol.KEY + ["luteal_status"]].copy()
            part["probability"] = probabilities
            part["validation_fold"] = validation_fold
            validation_parts.append(part)
            tuning_rows.append({
                "inverse_penalty_strength": inverse_penalty_strength,
                "validation_fold": validation_fold,
                **daily_metrics(part),
            })
        candidate_predictions[inverse_penalty_strength] = pd.concat(validation_parts, ignore_index=True)

    tuning = pd.DataFrame(tuning_rows)
    means = tuning.groupby("inverse_penalty_strength", as_index=False).average_precision.mean()
    selected = means.sort_values(
        ["average_precision", "inverse_penalty_strength"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0].inverse_penalty_strength
    return float(selected), tuning, candidate_predictions[float(selected)]
