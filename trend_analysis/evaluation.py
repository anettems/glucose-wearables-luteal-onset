"""Evaluate the fixed exploratory variants on the same participant folds."""

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
    columns = feature_columns(input_set, variant)
    strength, tuning, validation = tune_model(development, assignments, columns)
    threshold, consecutive, alert_tuning = select_alert_rule(validation, references)
    probabilities, parameters = fit_and_predict(development, testing, columns, strength)
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
    parts = {}
    for variant in VARIANTS:
        for fold in range(1, config.OUTER_FOLDS + 1):
            ids = test_folds.loc[test_folds.outer_test_fold.eq(fold), "id"]
            testing = observations.loc[observations.id.isin(ids)]
            development = observations.loc[~observations.id.isin(ids)]
            starts = references.loc[references.id.isin(development.id)]
            assignments = validation_folds.loc[validation_folds.outer_test_fold.eq(fold), ["id", "validation_fold"]]
            for input_set in FEATURE_SETS:
                print(f"Fit: {variant}, test fold {fold}, {input_set}", flush=True)
                result = evaluate_input(development, testing, starts, assignments, input_set, variant, fold)
                for name, frame in result.items():
                    parts.setdefault(name, []).append(frame)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    validate_all_predictions(tables["model_predictions"], observations)
    return tables


def event_and_calibration_tables(tables, observations):
    event_parts = {}
    for variant in VARIANTS:
        predictions = tables["model_predictions"].loc[tables["model_predictions"].variant.eq(variant)]
        settings = tables["selected_settings"].loc[tables["selected_settings"].variant.eq(variant)].drop(columns="variant")
        validate_paired_predictions(predictions, observations)
        generated, events, alerts, metrics = evaluate_test_alerts(predictions, tables["references"], settings)
        variant_tables = {"onset_alerts": generated, "onset_reference_events": events,
                          "onset_alert_events": alerts, "onset_fold_metrics": metrics,
                          "onset_summary": onset_summary(metrics, events), **calibration_tables(predictions)}
        rows = []
        for (name, fold), group in predictions.groupby(["input_set", "outer_test_fold"]):
            event = metrics.loc[metrics.input_set.eq(name) & metrics.outer_test_fold.eq(fold)].iloc[0].to_dict()
            rows.append({**daily_metrics(group), **event})
        variant_tables["model_fold_metrics"] = pd.DataFrame(rows)
        for name, frame in variant_tables.items():
            event_parts.setdefault(name, []).append(frame.assign(variant=variant))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in event_parts.items()}
