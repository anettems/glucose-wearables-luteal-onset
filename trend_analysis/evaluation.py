#### Run the paired model comparison for the exploratory history analysis.
#### Compare glucose-only, wearable-only and combined inputs for each of three
#### feature variants: current values, values plus deviations, and all of these
#### plus trends. Use identical eligible days and saved participant folds.
#### Within each test round, choose C and the onset-alert rule using validation
#### participants. Refit the model on all development participants before
#### predicting the separate test participants. Test outcomes do not choose settings.
#### Collect predictions, selected settings and fitted parameters, then check
#### that every model predicts the same held-out observation keys.
#### Build onset and probability-calibration tables from those saved predictions.
#### File saving and bootstrap comparisons are handled by the workflow and statistics modules.

import pandas as pd

from daily_metrics import daily_metrics, validate_paired_predictions
from model_inputs import FEATURE_SETS
from onset_alerts import evaluate_test_alerts, select_alert_rule
from primary_evaluation import onset_summary, parameter_rows
import primary_protocol as config
from reporting_tables import calibration_tables
from trend_analysis.models import fit_and_predict, tune_model
from trend_analysis.protocol import VARIANTS, feature_columns
from trend_analysis.statistics import validate_all_predictions


def evaluate_input(development, testing, references, assignments, input_set, variant, fold):
    #### Select settings using validation data, then fit and test one input set and feature
    #### variant.
    #### Choose the predictors and tune C using development participants only.
    columns = feature_columns(input_set, variant)
    strength, tuning, validation = tune_model(development, assignments, columns)

    #### Choose the alert rule from validation probabilities before generating test predictions.
    threshold, consecutive, alert_tuning = select_alert_rule(validation, references)

    #### Refit on all development participants and predict the separate test fold.
    probabilities, parameters = fit_and_predict(development, testing, columns, strength)

    #### Label every output with its input set, test fold and feature variant.
    tags = {"input_set": input_set, "outer_test_fold": fold, "variant": variant}

    return {
        "model_predictions": testing[config.KEY + ["luteal_status"]].assign(probability=probabilities, **tags),
        "tuning_results": tuning.assign(**tags),
        "validation_predictions": validation.assign(inverse_penalty_strength=strength, **tags),
        "alert_tuning_results": alert_tuning.assign(**tags),
        "selected_settings": pd.DataFrame([{**tags, "inverse_penalty_strength": strength,
                                             "threshold": threshold, "consecutive_days": consecutive}]),
        "fitted_parameters": parameter_rows(fold, input_set, parameters).assign(variant=variant),
    }


def evaluate_variants(observations, references, test_folds, validation_folds):
    #### Evaluate every model on the same saved test participants and observation keys.
    parts = {}

    #### Compare the three feature variants on the same eligible observations.
    for variant in VARIANTS:
        #### Reserve each saved test fold in turn and use the others for development.
        for fold in range(1, config.OUTER_FOLDS + 1):
            ids = test_folds.loc[test_folds.outer_test_fold.eq(fold), "id"]
            testing = observations.loc[observations.id.isin(ids)]
            development = observations.loc[~observations.id.isin(ids)]

            #### Provide only development participants' reference starts for alert-rule
            #### selection.
            starts = references.loc[references.id.isin(development.id)]
            assignments = validation_folds.loc[validation_folds.outer_test_fold.eq(fold), ["id", "validation_fold"]]

            #### Fit glucose-only, wearable-only and combined models on these same participant
            #### groups.
            for input_set in FEATURE_SETS:
                print(f"Fit: {variant}, test fold {fold}, {input_set}", flush=True)
                result = evaluate_input(development, testing, starts, assignments, input_set, variant, fold)

                #### Collect each output table for later pairing checks and reporting.
                for name, frame in result.items():
                    parts.setdefault(name, []).append(frame)

    #### Combine outputs and verify that all nine models predict identical held-out keys.
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    validate_all_predictions(tables["model_predictions"], observations)

    return tables


def event_and_calibration_tables(tables, observations):
    #### Assess onset alerts and probability calibration separately for each feature variant.
    event_parts = {}

    #### Keep event and calibration summaries separate for each feature variant.
    for variant in VARIANTS:
        predictions = tables["model_predictions"].loc[tables["model_predictions"].variant.eq(variant)]
        settings = tables["selected_settings"].loc[tables["selected_settings"].variant.eq(variant)].drop(columns="variant")
        validate_paired_predictions(predictions, observations)

        #### Apply the chosen alert rules and summarise agreement with supplied reference
        #### starts.
        generated, events, alerts, metrics = evaluate_test_alerts(predictions, tables["references"], settings)
        variant_tables = {"onset_alerts": generated, "onset_reference_events": events,
                          "onset_alert_events": alerts, "onset_fold_metrics": metrics,
                          "onset_summary": onset_summary(metrics, events), **calibration_tables(predictions)}

        #### Collect fold-level daily and event measures as a stability check.
        rows = []

        #### Join daily and onset measures for each model and test fold.
        for (name, fold), group in predictions.groupby(["input_set", "outer_test_fold"]):
            event = metrics.loc[metrics.input_set.eq(name) & metrics.outer_test_fold.eq(fold)].iloc[0].to_dict()
            rows.append({**daily_metrics(group), **event})

        variant_tables["model_fold_metrics"] = pd.DataFrame(rows)

        #### Label every generated table before combining results across variants.
        for name, frame in variant_tables.items():
            event_parts.setdefault(name, []).append(frame.assign(variant=variant))

    return {name: pd.concat(frames, ignore_index=True) for name, frames in event_parts.items()}
