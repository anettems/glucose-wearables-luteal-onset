"""Paired daily performance estimates and participant bootstrap intervals."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import primary_protocol as protocol
from model_inputs import FEATURE_SETS


def validate_scores(labels, probabilities):
    """Reject invalid outcomes or probabilities before computing a metric."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Expected non-empty aligned outcome and probability vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Daily outcomes must be binary 0/1 values")
    if not (np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Predicted probabilities must be finite and between zero and one")
    return labels.astype(int), probabilities


def average_precision(labels, probabilities):
    """Calculate non-interpolated AP; no-positive samples are undefined."""
    labels, probabilities = validate_scores(labels, probabilities)
    if labels.sum() == 0:
        raise ValueError("Average precision requires at least one positive outcome")
    return float(average_precision_score(labels, probabilities))


def roc_auc(labels, probabilities):
    """Calculate ROC-AUC only when both outcome classes are present."""
    labels, probabilities = validate_scores(labels, probabilities)
    if labels.min() == labels.max():
        raise ValueError("ROC-AUC requires both outcome classes")
    return float(roc_auc_score(labels, probabilities))


def daily_metrics(predictions):
    labels, probabilities = validate_scores(predictions.luteal_status, predictions.probability)
    return {
        "observations": len(labels),
        "luteal_days": int(labels.sum()),
        "luteal_prevalence": float(labels.mean()),
        "average_precision": average_precision(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
    }


def validate_paired_predictions(predictions, observations=None):
    """Require identical keys, outcomes and testing folds for all three models."""
    if set(predictions.input_set) != set(FEATURE_SETS):
        raise ValueError("Expected exactly the three prespecified input sets")
    wearable = predictions.loc[predictions.input_set.eq("wearable")]
    if observations is None:
        observations = wearable
    columns = protocol.KEY + ["luteal_status"]
    expected = observations[columns].sort_values(protocol.KEY).reset_index(drop=True)
    if expected.empty or expected.duplicated(protocol.KEY).any() or expected.isna().any().any():
        raise ValueError("Expected unique, non-missing observation keys and outcomes")
    for input_set in FEATURE_SETS:
        subset = predictions.loc[predictions.input_set.eq(input_set)]
        actual = subset[columns].sort_values(protocol.KEY).reset_index(drop=True)
        if subset.duplicated(protocol.KEY).any() or not actual.equals(expected):
            raise ValueError("Input sets must have one prediction per key and identical outcomes")
        validate_scores(subset.luteal_status, subset.probability)
    if "outer_test_fold" in predictions:
        expected_folds = wearable[protocol.KEY + ["outer_test_fold"]].sort_values(protocol.KEY)
        if predictions.outer_test_fold.isna().any():
            raise ValueError("Every held-out prediction needs a test fold")
        for _, subset in predictions.groupby("input_set"):
            actual = subset[protocol.KEY + ["outer_test_fold"]].sort_values(protocol.KEY)
            if not actual.reset_index(drop=True).equals(expected_folds.reset_index(drop=True)):
                raise ValueError("Input sets have different held-out test folds")
        if predictions.groupby("id").outer_test_fold.nunique().gt(1).any():
            raise ValueError("A participant appears in more than one held-out test fold")


def bootstrap_metrics(predictions):
    """Use one participant draw for every model, preserving repeated selections."""
    validate_paired_predictions(predictions)
    grouped = {
        input_set: dict(tuple(frame.groupby("id", sort=True)))
        for input_set, frame in predictions.groupby("input_set", sort=True)
    }
    participants = sorted(grouped["wearable"])
    rng = np.random.default_rng(protocol.BOOTSTRAP_SEED)
    rows = []
    for replicate in range(protocol.BOOTSTRAP_RESAMPLES):
        selected = rng.choice(participants, size=len(participants), replace=True)
        row = {"replicate": replicate + 1, "participant_draw": json.dumps(selected.tolist())}
        for input_set in FEATURE_SETS:
            sample = pd.concat([grouped[input_set][person] for person in selected], ignore_index=True)
            labels = sample.luteal_status.to_numpy(dtype=int)
            row[f"{input_set}_average_precision"] = (
                average_precision(labels, sample.probability) if labels.any() else np.nan
            )
            row[f"{input_set}_roc_auc"] = (
                roc_auc(labels, sample.probability) if len(np.unique(labels)) == 2 else np.nan
            )
        row["combined_minus_wearable_average_precision"] = (
            row["combined_average_precision"] - row["wearable_average_precision"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def interval(values):
    """Use finite replicates and report their count separately in each summary."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Fewer than two defined bootstrap replicates; cannot estimate an interval")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def interval_record(values):
    lower, upper = interval(values)
    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_valid_resamples": int(np.isfinite(values).sum()),
        "bootstrap_requested_resamples": len(values),
    }


def summary_tables(predictions, bootstrap):
    """Report pooled estimates; confidence intervals condition on fitted models."""
    validate_paired_predictions(predictions)
    rows, metrics = [], {}
    for input_set in FEATURE_SETS:
        metrics[input_set] = daily_metrics(predictions.loc[predictions.input_set.eq(input_set)])
        for metric in ["average_precision", "roc_auc"]:
            rows.append({
                "input_set": input_set, "metric": metric, "estimate": metrics[input_set][metric],
                **interval_record(bootstrap[f"{input_set}_{metric}"]),
            })
        rows.append({
            "input_set": input_set, "metric": "luteal_prevalence",
            "estimate": metrics[input_set]["luteal_prevalence"],
            "ci_lower": np.nan, "ci_upper": np.nan,
            "bootstrap_valid_resamples": np.nan, "bootstrap_requested_resamples": len(bootstrap),
        })
    paired = pd.DataFrame([{
        "comparison": "combined_minus_wearable", "metric": "average_precision_difference",
        "estimate": metrics["combined"]["average_precision"] - metrics["wearable"]["average_precision"],
        **interval_record(bootstrap["combined_minus_wearable_average_precision"]),
    }])
    return pd.DataFrame(rows), paired
