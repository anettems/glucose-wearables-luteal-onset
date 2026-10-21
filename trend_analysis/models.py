"""Logistic regression with training-only replacement of raw and derived predictors."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from daily_metrics import daily_metrics
from daily_models import create_logistic_model, fit_model
import primary_protocol as config


def fit_and_predict(training, assessment, columns, strength):
    if set(training.id) & set(assessment.id):
        raise ValueError("Training and assessment participants overlap")
    if training.empty or assessment.empty:
        raise ValueError("Expected non-empty training and assessment cohorts")
    medians = training[columns].median()
    if medians.isna().any():
        raise ValueError("A predictor has no observed training value")
    train = training[columns].fillna(medians).to_numpy(dtype=float)
    test = assessment[columns].fillna(medians).to_numpy(dtype=float)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Non-finite model input")
    model = make_pipeline(StandardScaler(), create_logistic_model(strength))
    fit_model(model, train, training.luteal_status.to_numpy(dtype=int))
    scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    parameters = {"replacement_medians": medians.to_dict(), "feature_means": dict(zip(columns, scaler.mean_)),
                  "feature_scales": dict(zip(columns, scaler.scale_)),
                  "coefficients": dict(zip(["intercept"] + columns, [classifier.intercept_[0], *classifier.coef_[0]])),
                  "iterations": int(classifier.n_iter_[0]), "sklearn_C": float(classifier.C),
                  "training_observations": len(training)}
    return model.predict_proba(test)[:, 1], parameters


def tune_model(development, assignments, columns):
    if assignments.id.duplicated().any() or set(assignments.id) != set(development.id):
        raise ValueError("Validation assignments do not match development participants")
    if set(assignments.validation_fold) != set(range(1, config.INNER_FOLDS + 1)):
        raise ValueError("Missing validation folds")
    scores, predictions = [], {}
    for strength in config.LOGISTIC_C_VALUES:
        parts = []
        for fold in range(1, config.INNER_FOLDS + 1):
            ids = assignments.loc[assignments.validation_fold.eq(fold), "id"]
            training = development.loc[~development.id.isin(ids)]
            validation = development.loc[development.id.isin(ids)]
            probabilities, _ = fit_and_predict(training, validation, columns, strength)
            part = validation[config.KEY + ["luteal_status"]].assign(probability=probabilities, validation_fold=fold)
            parts.append(part)
            scores.append({"inverse_penalty_strength": strength, "validation_fold": fold, **daily_metrics(part)})
        predictions[strength] = pd.concat(parts, ignore_index=True)
    scores = pd.DataFrame(scores)
    means = scores.groupby("inverse_penalty_strength", as_index=False).average_precision.mean()
    selected = float(means.sort_values(["average_precision", "inverse_penalty_strength"], ascending=[False, True]).iloc[0].inverse_penalty_strength)
    return selected, scores, predictions[selected]
