#### Fit and tune logistic regression for the primary daily luteal-phase comparison.
#### Take paired participant-day tables, selected predictor columns and saved validation
#### assignments as inputs. Replace missing wearable values with training medians, then
#### fit scaling and the model using training participants only. Return luteal probabilities
#### and fitted parameters so each prediction can be checked.
#### Compare C = 0.01, 0.1, 1 and 10 using mean validation average precision, a measure of
#### how well probabilities identify luteal days. Prefer the smaller C when scores tie.
#### Participant folds and held-out testing are managed elsewhere; this file does not
#### construct features, choose alert rules or write results.

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


def create_logistic_model(inverse_penalty_strength):
    #### Create the fixed model with the supplied C, which controls the penalty strength.
    #### The L2 penalty discourages large coefficients; a smaller C gives a stronger penalty.
    return LogisticRegression(
        C=inverse_penalty_strength,
        solver="lbfgs",
        l1_ratio=0,
        fit_intercept=True,
        class_weight=None,
        max_iter=protocol.LOGISTIC_MAX_ITERATIONS,
        tol=protocol.LOGISTIC_TOLERANCE,
        random_state=protocol.PRIMARY_SEED,
    )


def fit_model(model, features, labels):
    #### Fit a model only when both outcome classes are present and fitting converges.
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Logistic-regression training requires both outcome classes")

    #### Stop on a convergence warning rather than use an unfinished model fit.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return model.fit(features, labels)
        except ConvergenceWarning as error:
            raise RuntimeError("Logistic regression did not converge; no valid model was fitted") from error


def fit_and_predict(training, evaluation, feature_columns, inverse_penalty_strength):
    #### Fit all learned steps on training participants, then predict separate participants.
    #### Use the same training medians to replace missing wearables in both tables.
    training_filled, evaluation_filled, medians = replace_missing_wearables(training, evaluation)

    #### Select only the requested predictors; labels and observation keys stay out of X.
    training_features = training_filled[feature_columns].to_numpy(dtype=float)
    evaluation_features = evaluation_filled[feature_columns].to_numpy(dtype=float)

    #### Fit scaling and logistic regression together so scaling uses training values only.
    #### Scaling subtracts each feature's training mean and divides by its training spread.
    model = make_pipeline(
        StandardScaler(),
        create_logistic_model(inverse_penalty_strength),
    )
    fit_model(model, training_features, training_filled.luteal_status.to_numpy(dtype=int))

    #### Apply the fitted steps to evaluation values and return the probability of class 1
    #### (luteal), rather than a classification based on a fixed probability threshold.
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    probabilities = model.predict_proba(evaluation_features)[:, 1]

    #### Record replacement values, scaling and model coefficients for reproducibility.
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
    #### Choose C using validation participants within the development set, before testing.
    #### Require one saved validation assignment for every development participant.
    if validation_assignments.id.duplicated().any():
        raise ValueError("Each development participant must have one validation-fold assignment")
    assignment = validation_assignments.set_index("id").validation_fold
    if set(development.id) != set(assignment.index):
        raise ValueError("Validation assignments do not cover the development participants")
    if set(assignment) != set(range(1, protocol.INNER_FOLDS + 1)):
        raise ValueError("Expected every prespecified validation fold")

    #### Retain each candidate's fold scores and predictions for later assessment.
    tuning_rows, candidate_predictions = [], {}

    #### Compare the four fixed C values using the same three validation folds.
    for inverse_penalty_strength in protocol.LOGISTIC_C_VALUES:
        validation_parts = []

        #### Train on two folds and predict the third, keeping each participant's days
        #### together. Repeat until every development participant has validation predictions.
        for validation_fold in range(1, protocol.INNER_FOLDS + 1):
            validation_ids = assignment.loc[assignment.eq(validation_fold)].index
            validation = development.loc[development.id.isin(validation_ids)].copy()
            training = development.loc[~development.id.isin(validation_ids)].copy()

            #### Refit replacement, scaling and the model for this training split only.
            probabilities, _ = fit_and_predict(
                training, validation, feature_columns, inverse_penalty_strength,
            )

            #### Keep keys and labels with predictions so validation scores can be checked.
            part = validation[protocol.KEY + ["luteal_status"]].copy()
            part["probability"] = probabilities
            part["validation_fold"] = validation_fold
            validation_parts.append(part)
            tuning_rows.append({
                "inverse_penalty_strength": inverse_penalty_strength,
                "validation_fold": validation_fold,
                **daily_metrics(part),
            })

        #### Combine the three validation folds for this candidate C.
        candidate_predictions[inverse_penalty_strength] = pd.concat(validation_parts, ignore_index=True)

    #### Choose the highest mean validation average precision; ties favour the smaller C.
    tuning = pd.DataFrame(tuning_rows)
    means = tuning.groupby("inverse_penalty_strength", as_index=False).average_precision.mean()
    selected = means.sort_values(
        ["average_precision", "inverse_penalty_strength"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0].inverse_penalty_strength
    return float(selected), tuning, candidate_predictions[float(selected)]
