#### Assess paired three-day onset probabilities with ranking, calibration and Brier scores.
#### Bootstrap whole participants with the same saved draws for every input set.
#### No threshold-dependent classification or event-alert metrics are calculated.

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from onset_prediction import protocol


INPUT_SETS = ("glucose", "wearable", "combined")
METRICS = ("average_precision", "roc_auc", "brier_score")


def validate_scores(labels, probabilities):
    #### Reject invalid or misaligned outcomes and probabilities before calculating metrics.
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Expected non-empty aligned outcome and probability vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary outcomes")
    if not (np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Expected finite probabilities between zero and one")
    return labels.astype(int), probabilities


def probability_metrics(predictions):
    #### Calculate non-interpolated AP, ROC-AUC and squared probability error.
    labels, probabilities = validate_scores(predictions[protocol.TARGET], predictions.probability)
    return {"days": len(labels), "positive_days": int(labels.sum()), "prevalence": float(labels.mean()),
            "average_precision": float(average_precision_score(labels, probabilities)) if labels.any() else np.nan,
            "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan,
            "brier_score": float(np.mean(np.square(probabilities - labels)))}


def validate_paired(predictions, observations, assignments, model_family="logistic_regression"):
    #### Require exactly one prediction per saved key, outcome and test fold for all input sets.
    if set(predictions.input_set) != set(INPUT_SETS):
        raise ValueError("Expected exactly the three agreed input sets")
    if model_family not in ("logistic_regression", "random_forest") or set(predictions.model_family) != {model_family}:
        raise ValueError("Unexpected model family in paired evaluation")
    columns = protocol.KEY + [protocol.TARGET, "test_fold"]
    expected = observations[protocol.KEY + [protocol.TARGET]].merge(
        assignments, on="id", how="left", validate="many_to_one")[columns]
    expected = expected.sort_values(protocol.KEY).reset_index(drop=True)
    if expected.empty or expected.isna().any().any() or expected.duplicated(protocol.KEY).any():
        raise ValueError("Invalid expected prediction keys, outcomes or folds")
    for name in INPUT_SETS:
        subset = predictions.loc[predictions.input_set.eq(name)]
        actual = subset[columns].sort_values(protocol.KEY).reset_index(drop=True)
        if subset.duplicated(protocol.KEY).any() or not actual.equals(expected):
            raise ValueError("Missing, duplicated or unpaired held-out prediction")
        validate_scores(subset[protocol.TARGET], subset.probability)
    if predictions.groupby("id").test_fold.nunique().gt(1).any():
        raise ValueError("A participant appears in multiple test folds")


def calibration_bins(predictions):
    #### Retain all ten fixed bins, with empty summaries missing and probability one in the last.
    labels, probabilities = validate_scores(predictions[protocol.TARGET], predictions.probability)
    bins = np.minimum(np.floor(probabilities * 10).astype(int), 9)
    rows = []
    for number in range(10):
        selected = bins == number
        rows.append({"bin": number, "lower": number / 10, "upper": (number + 1) / 10,
                     "days": int(selected.sum()), "positive_days": int(labels[selected].sum()),
                     "mean_probability": float(probabilities[selected].mean()) if selected.any() else np.nan,
                     "observed_frequency": float(labels[selected].mean()) if selected.any() else np.nan})
    return pd.DataFrame(rows)


def participant_draws(participants, resamples=2000):
    #### Generate reproducible participant draws that can also be reused by random forest later.
    participants = sorted(participants)
    rng = np.random.default_rng(protocol.SEED + 1)
    return pd.DataFrame([{"replicate": number + 1,
                          "participant_draw": json.dumps(rng.choice(participants, len(participants), replace=True).tolist())}
                         for number in range(resamples)])


def bootstrap_metrics(predictions, draws):
    #### Retain every selected participant's days, including repeated selections, for each model.
    groups = {name: frame.sort_values(protocol.KEY).reset_index(drop=True)
              for name, frame in predictions.groupby("input_set", sort=False)}
    reference = groups["wearable"]
    positions = {person: np.flatnonzero(reference.id.to_numpy() == person) for person in sorted(reference.id.unique())}
    for name in INPUT_SETS:
        if not groups[name][protocol.KEY + [protocol.TARGET]].equals(reference[protocol.KEY + [protocol.TARGET]]):
            raise ValueError("Bootstrap inputs are not paired")
    rows = []
    for draw in draws.itertuples(index=False):
        selected = json.loads(draw.participant_draw)
        if len(selected) != len(positions) or not set(selected).issubset(positions):
            raise ValueError("Invalid saved participant bootstrap draw")
        indices = np.concatenate([positions[person] for person in selected])
        row = {"replicate": draw.replicate}
        for name in INPUT_SETS:
            metrics = probability_metrics(groups[name].iloc[indices])
            row.update({f"{name}_{metric}": metrics[metric] for metric in METRICS})
        row["combined_minus_wearable_average_precision"] = row["combined_average_precision"] - row["wearable_average_precision"]
        rows.append(row)
    return pd.DataFrame(rows)


def interval_record(values):
    #### Calculate percentile intervals from finite draws and report their contributing counts.
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        raise ValueError("Too few defined bootstrap replicates for a confidence interval")
    lower, upper = np.quantile(finite, [0.025, 0.975], method="linear")
    return {"ci_lower": float(lower), "ci_upper": float(upper),
            "valid_resamples": len(finite), "requested_resamples": len(values)}


def summary_tables(predictions, bootstrap, model_family="logistic_regression"):
    #### Summarise pooled model estimates and the paired glucose contribution.
    rows, estimates = [], {}
    for name in INPUT_SETS:
        estimates[name] = probability_metrics(predictions.loc[predictions.input_set.eq(name)])
        for metric in METRICS:
            rows.append({"model_family": model_family, "input_set": name, "metric": metric,
                         "estimate": estimates[name][metric], "prevalence": estimates[name]["prevalence"],
                         **interval_record(bootstrap[f"{name}_{metric}"])})
    difference = estimates["combined"]["average_precision"] - estimates["wearable"]["average_precision"]
    paired = {"model_family": model_family, "comparison": "combined_minus_wearable",
              "metric": "average_precision_difference", "estimate": difference,
              **interval_record(bootstrap["combined_minus_wearable_average_precision"])}
    return pd.DataFrame(rows), pd.DataFrame([paired])
