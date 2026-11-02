#### Fit and tune logistic regression for the personal-history analysis.
#### Use the selected current measurements, personal deviations and trends
#### to estimate the probability that a participant-day is labelled luteal.
#### Replace missing inputs with training medians, then standardise the inputs
#### and fit logistic regression using training participants only. Apply the
#### same fitted steps to separate validation or test participants.
#### Compare the four fixed values of C, the inverse penalty strength, using
#### mean validation average precision. Choose the smaller C when scores tie.
#### Return probabilities and fitted parameters so the analysis can be checked.
#### Feature construction and the choice of test participants happen elsewhere.

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from daily_metrics import daily_metrics
from daily_models import create_logistic_model, fit_model
import primary_protocol as config


def fit_and_predict(training, assessment, columns, strength):
    #### Fit replacement, scaling and logistic regression on training participants, then predict
    #### separate people.
    if set(training.id) & set(assessment.id):
        raise ValueError("Training and assessment participants overlap")
    if training.empty or assessment.empty:
        raise ValueError("Expected non-empty training and assessment cohorts")

    #### Estimate one replacement median per selected predictor from training data only.
    medians = training[columns].median()
    if medians.isna().any():
        raise ValueError("A predictor has no observed training value")

    #### Apply the same training medians to both tables before checking for invalid values.
    train = training[columns].fillna(medians).to_numpy(dtype=float)
    test = assessment[columns].fillna(medians).to_numpy(dtype=float)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Non-finite model input")

    #### Fit scaling and logistic regression together on training participants only.
    model = make_pipeline(StandardScaler(), create_logistic_model(strength))
    fit_model(model, train, training.luteal_status.to_numpy(dtype=int))

    #### Save replacement, scaling and model parameters for reproducibility checks.
    scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    parameters = {"replacement_medians": medians.to_dict(), "feature_means": dict(zip(columns, scaler.mean_)),
                  "feature_scales": dict(zip(columns, scaler.scale_)),
                  "coefficients": dict(zip(["intercept"] + columns, [classifier.intercept_[0], *classifier.coef_[0]])),
                  "iterations": int(classifier.n_iter_[0]), "sklearn_C": float(classifier.C),
                  "training_observations": len(training)}

    return model.predict_proba(test)[:, 1], parameters


def tune_model(development, assignments, columns):
    #### Choose C using participant-disjoint validation folds within the development set.
    if assignments.id.duplicated().any() or set(assignments.id) != set(development.id):
        raise ValueError("Validation assignments do not match development participants")
    if set(assignments.validation_fold) != set(range(1, config.INNER_FOLDS + 1)):
        raise ValueError("Missing validation folds")

    #### Keep validation scores and predictions for every candidate C.
    scores, predictions = [], {}

    #### Compare each prespecified C using the same validation participants.
    for strength in config.LOGISTIC_C_VALUES:
        parts = []

        #### Hold out one validation fold and train on the remaining development participants.
        for fold in range(1, config.INNER_FOLDS + 1):
            ids = assignments.loc[assignments.validation_fold.eq(fold), "id"]
            training = development.loc[~development.id.isin(ids)]
            validation = development.loc[development.id.isin(ids)]

            #### Fit all learned steps on this training split before predicting the validation
            #### fold.
            probabilities, _ = fit_and_predict(training, validation, columns, strength)
            part = validation[config.KEY + ["luteal_status"]].assign(probability=probabilities, validation_fold=fold)
            parts.append(part)
            scores.append({"inverse_penalty_strength": strength, "validation_fold": fold, **daily_metrics(part)})

        #### Combine the three validation folds for this candidate setting.
        predictions[strength] = pd.concat(parts, ignore_index=True)

    #### Select the highest mean validation AP, resolving ties with the smaller C.
    scores = pd.DataFrame(scores)
    means = scores.groupby("inverse_penalty_strength", as_index=False).average_precision.mean()
    selected = float(means.sort_values(["average_precision", "inverse_penalty_strength"], ascending=[False, True]).iloc[0].inverse_penalty_strength)

    return selected, scores, predictions[selected]
