"""Reuse participant-separated fitting, onset evaluation and paired bootstraps."""

import numpy as np
import pandas as pd

import primary_protocol as config
from daily_metrics import bootstrap_metrics, daily_metrics, interval_record, summary_tables, validate_paired_predictions
from model_inputs import FEATURE_SETS
from onset_alerts import evaluate_test_alerts, select_alert_rule
from primary_evaluation import onset_summary, parameter_rows
from reporting_tables import calibration_tables
from trend_analysis.models import fit_and_predict, tune_model
from within_day_analysis import protocol


def evaluate_input(development, testing, references, assignments, input_set, variant, fold):
    columns = protocol.feature_columns(input_set, variant)
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


def validate_predictions(predictions, observations):
    if set(predictions.variant) != set(protocol.VARIANTS):
        raise ValueError("Expected baseline and within-day predictions")
    expected = None
    for variant in protocol.VARIANTS:
        subset = predictions.loc[predictions.variant.eq(variant)]
        validate_paired_predictions(subset, observations)
        actual = subset.loc[subset.input_set.eq("wearable"), config.KEY + ["outer_test_fold"]]
        actual = actual.sort_values(config.KEY).reset_index(drop=True)
        if expected is not None and not actual.equals(expected):
            raise ValueError("Feature variants use different test folds")
        expected = actual
    wearable = predictions.loc[predictions.input_set.eq("wearable")]
    first = wearable.loc[wearable.variant.eq("baseline")].sort_values(config.KEY).probability.to_numpy()
    second = wearable.loc[wearable.variant.eq("within_day")].sort_values(config.KEY).probability.to_numpy()
    if not np.array_equal(first, second):
        raise ValueError("Unchanged wearable inputs produced different probabilities")


def evaluate_models(observations, references, test_folds, validation_folds):
    parts = {}
    for fold in range(1, config.OUTER_FOLDS + 1):
        ids = test_folds.loc[test_folds.outer_test_fold.eq(fold), "id"]
        testing = observations.loc[observations.id.isin(ids)]
        development = observations.loc[~observations.id.isin(ids)]
        starts = references.loc[references.id.isin(development.id)]
        assignments = validation_folds.loc[validation_folds.outer_test_fold.eq(fold), ["id", "validation_fold"]]
        for input_set in FEATURE_SETS:
            cached = None
            for variant in protocol.VARIANTS:
                print(f"Fit: {variant}, test fold {fold}, {input_set}", flush=True)
                if cached is not None and input_set == "wearable":
                    result = {name: frame.assign(variant=variant) for name, frame in cached.items()}
                else:
                    result = evaluate_input(development, testing, starts, assignments, input_set, variant, fold)
                cached = result
                for name, frame in result.items():
                    parts.setdefault(name, []).append(frame)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    validate_predictions(tables["model_predictions"], observations)
    return tables


def event_tables(tables):
    parts = {}
    for variant in protocol.VARIANTS:
        predictions = tables["model_predictions"].loc[tables["model_predictions"].variant.eq(variant)]
        settings = tables["selected_settings"].loc[tables["selected_settings"].variant.eq(variant)]
        generated, events, alerts, metrics = evaluate_test_alerts(predictions, tables["references"], settings)
        rows = []
        for (name, fold), group in predictions.groupby(["input_set", "outer_test_fold"]):
            event = metrics.loc[metrics.input_set.eq(name) & metrics.outer_test_fold.eq(fold)].iloc[0].to_dict()
            rows.append({**daily_metrics(group), **event})
        frames = {"onset_alerts": generated, "onset_reference_events": events, "onset_alert_events": alerts,
                  "onset_fold_metrics": metrics, "onset_summary": onset_summary(metrics, events),
                  "model_fold_metrics": pd.DataFrame(rows), **calibration_tables(predictions)}
        for name, frame in frames.items():
            parts.setdefault(name, []).append(frame.assign(variant=variant))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def feature_effects(summary, bootstrap):
    indexed = summary.set_index(["variant", "input_set", "metric"]).estimate
    first = bootstrap.loc[bootstrap.variant.eq("baseline")].sort_values("replicate").reset_index(drop=True)
    second = bootstrap.loc[bootstrap.variant.eq("within_day")].sort_values("replicate").reset_index(drop=True)
    if not first[["replicate", "participant_draw"]].equals(second[["replicate", "participant_draw"]]):
        raise ValueError("Feature comparisons require identical participant bootstrap draws")
    rows = []
    for name in FEATURE_SETS:
        for metric in ["average_precision", "roc_auc"]:
            differences = second[f"{name}_{metric}"] - first[f"{name}_{metric}"]
            estimate = indexed["within_day", name, metric] - indexed["baseline", name, metric]
            rows.append({"comparison": "within_day_minus_baseline", "input_set": name, "metric": metric,
                         "estimate": float(estimate), **interval_record(differences)})
    return pd.DataFrame(rows)


def bootstrap_tables(predictions, observations):
    validate_predictions(predictions, observations)
    draws, summaries, paired = [], [], []
    for variant in protocol.VARIANTS:
        print(f"Bootstrap: {variant}, {config.BOOTSTRAP_RESAMPLES} participant draws", flush=True)
        subset = predictions.loc[predictions.variant.eq(variant)]
        bootstrap = bootstrap_metrics(subset)
        summary, effect = summary_tables(subset, bootstrap)
        draws.append(bootstrap.assign(variant=variant))
        summaries.append(summary.assign(variant=variant))
        paired.append(effect.assign(variant=variant))
    bootstrap, summary = pd.concat(draws, ignore_index=True), pd.concat(summaries, ignore_index=True)
    return {"bootstrap_metrics": bootstrap, "model_summary": summary,
            "paired_comparison": pd.concat(paired, ignore_index=True),
            "feature_comparisons": feature_effects(summary, bootstrap)}
