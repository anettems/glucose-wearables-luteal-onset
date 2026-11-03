#### Assess saved onset probabilities as daily yes/no predictions, without refitting models.
#### Select thresholds using only the chosen model's participant-disjoint validation scores.
#### Save pooled accuracy, balanced accuracy, sensitivity, precision and specificity, with
#### the original paired participant bootstrap draws. These are day-level, not event metrics.

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onset_prediction import protocol
from onset_prediction.freezing import (check_results_destination, read_json, verify_hashes,
                                      versions)
from onset_prediction.metrics import INPUT_SETS, interval_record, validate_paired, validate_scores
from onset_prediction.workflow import source_hashes, write_json
from preprocessing import file_sha256


THRESHOLDS = np.arange(101) / 100
METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "precision", "specificity")
COUNTS = ("tn", "fp", "fn", "tp")
FAMILIES = {"logistic_regression": ("onset_logistic_2022_v1", "C"),
            "random_forest": ("onset_forest_2022_v1", "min_samples_leaf")}


def count_metrics(counts):
    #### Convert confusion counts into metrics, retaining undefined ratios as missing.
    tn, fp, fn, tp = np.asarray(counts, dtype=float)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {"accuracy": (tp + tn) / (tn + fp + fn + tp),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "specificity": specificity}


def confusion_counts(labels, predicted):
    #### Count true negatives, false positives, false negatives and true positives in order.
    labels, predicted = validate_scores(labels, predicted)
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Expected binary predictions")
    return np.bincount(2 * labels + predicted.astype(int), minlength=4)


def select_threshold(validation):
    #### Maximise equal-weight mean fold balanced accuracy; exact ties favour higher thresholds.
    if set(validation.validation_fold) != {1, 2, 3} or validation.duplicated(protocol.KEY).any():
        raise ValueError("Expected one validation prediction per development key and three folds")
    rows = []
    for fold, data in validation.groupby("validation_fold"):
        labels, probabilities = validate_scores(data[protocol.TARGET], data.probability)
        if set(labels) != {0, 1}:
            raise ValueError("Every validation fold must contain both outcome classes")
        for threshold in THRESHOLDS:
            counts = confusion_counts(labels, probabilities >= threshold)
            rows.append({"threshold": threshold, "validation_fold": fold,
                         **count_metrics(counts)})
    scores = pd.DataFrame(rows)
    means = scores.groupby("threshold").balanced_accuracy.mean()
    threshold = float(means[means.eq(means.max())].index.max())
    return threshold, float(means.loc[threshold]), scores


def validate_validation(data, observations, assignments, validation_folds, fold):
    #### Reject missing keys, wrong labels, fold errors or test participants in threshold selection.
    expected = observations.loc[~observations.id.isin(assignments.loc[
        assignments.test_fold.eq(fold), "id"]), protocol.KEY + [protocol.TARGET]]
    expected = expected.merge(validation_folds.loc[validation_folds.test_fold.eq(fold),
                                                  ["id", "validation_fold"]],
                              on="id", validate="many_to_one")
    columns = protocol.KEY + [protocol.TARGET, "validation_fold"]
    actual = data[columns].sort_values(protocol.KEY).reset_index(drop=True)
    expected = expected[columns].sort_values(protocol.KEY).reset_index(drop=True)
    if actual.isna().any().any() or not actual.equals(expected):
        raise ValueError("Validation predictions differ from the saved development cohort/folds")


def classify_family(family, setting, saved, observations, assignments, validation_folds):
    #### Apply each validation-selected threshold only to its corresponding held-out test fold.
    selected_rows, score_rows, predictions = [], [], []
    test, validation, settings = saved
    validate_paired(test, observations, assignments, family)
    if settings.duplicated(["input_set", "test_fold"]).any() or len(settings) != 15:
        raise ValueError("Expected 15 unique selected model settings")
    for inputs in INPUT_SETS:
        for fold in range(1, 6):
            chosen = settings.loc[settings.input_set.eq(inputs) & settings.test_fold.eq(fold)]
            if len(chosen) != 1 or chosen.model_family.iloc[0] != family:
                raise ValueError("Missing or incorrect selected model setting")
            value = chosen[setting].iloc[0]
            data = validation.loc[validation.input_set.eq(inputs) & validation.test_fold.eq(fold)
                                  & validation[setting].eq(value)]
            if set(data.model_family) != {family}:
                raise ValueError("Incorrect validation model family")
            validate_validation(data, observations, assignments, validation_folds, fold)
            threshold, score, curve = select_threshold(data)
            identity = {"model_family": family, "input_set": inputs, "test_fold": fold}
            selected_rows.append({**identity, "model_setting": setting, "setting_value": value,
                                  "threshold": threshold, "mean_validation_balanced_accuracy": score})
            score_rows.append(curve.assign(**identity))
            held_out = test.loc[test.input_set.eq(inputs) & test.test_fold.eq(fold)].copy()
            held_out["threshold"] = threshold
            held_out["predicted_positive"] = held_out.probability.ge(threshold).astype(int)
            predictions.append(held_out)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(selected_rows), pd.concat(score_rows)


def summarise(predictions):
    #### Report pooled held-out metrics and fold stability with their complete confusion counts.
    pooled, folds = [], []
    for (family, inputs), data in predictions.groupby(["model_family", "input_set"]):
        counts = confusion_counts(data[protocol.TARGET], data.predicted_positive)
        identity = {"model_family": family, "input_set": inputs}
        pooled.append({**identity, "days": len(data), "participants": data.id.nunique(),
                       **dict(zip(COUNTS, counts)),
                       **count_metrics(counts)})
        for fold, subset in data.groupby("test_fold"):
            counts = confusion_counts(subset[protocol.TARGET], subset.predicted_positive)
            folds.append({**identity, "test_fold": fold, "days": len(subset),
                          **dict(zip(COUNTS, counts)), **count_metrics(counts)})
    return pd.DataFrame(pooled), pd.DataFrame(folds)


def bootstrap_classification(predictions, draws):
    #### Reuse whole-participant draws across all six models, keeping thresholds fixed.
    people = sorted(predictions.id.unique())
    weights = []
    for row in draws.itertuples(index=False):
        selected = json.loads(row.participant_draw)
        if len(selected) != len(people) or not set(selected).issubset(people):
            raise ValueError("Invalid saved participant draw")
        weights.append([selected.count(person) for person in people])
    weights = np.asarray(weights)
    rows = []
    for (family, inputs), data in predictions.groupby(["model_family", "input_set"]):
        counts = np.array([confusion_counts(group[protocol.TARGET], group.predicted_positive)
                           for _, group in data.groupby("id", sort=True)])
        for replicate, totals in zip(draws.replicate, weights @ counts):
            rows.append({"replicate": replicate, "model_family": family, "input_set": inputs,
                         **count_metrics(totals)})
    return pd.DataFrame(rows)


def load_saved(root, family):
    #### Verify immutable saved predictions and frozen inputs without refitting historical models.
    directory = root / FAMILIES[family][0]
    frozen, evaluation = directory / "frozen", directory / "evaluation"
    if read_json(evaluation / "run_status.json").get("status") != "complete":
        raise ValueError("A completed probability evaluation is required")
    verify_hashes(read_json(evaluation / "output_manifest.json")["files"], evaluation)
    verify_hashes(read_json(frozen / "freeze_manifest.json")["files"], frozen)
    saved = tuple(pd.read_csv(evaluation / name) for name in (
        "test_predictions.csv", "validation_predictions.csv", "selected_settings.csv"))
    return frozen, evaluation, saved


def classification_interval(values):
    #### Keep undefined precision or single-class bootstrap metrics missing without stopping.
    finite = np.isfinite(values).sum()
    if finite < 2:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "valid_resamples": int(finite),
                "requested_resamples": len(values)}
    return interval_record(values)


def write_report(destination, pooled):
    #### Explain daily metrics and the all-negative baseline alongside the numerical results.
    first = pooled.iloc[0]
    baseline = (first.tn + first.fp) / first.days
    lines = ["# Daily yes/no assessment of three-day onset probabilities", "",
             "Thresholds maximise mean balanced accuracy over three validation folds, using",
             "only the model setting previously selected by validation average precision.",
             "Candidates: 0.00 to 1.00 in steps of 0.01; exact ties favour the higher threshold.",
             "A probability equal to the threshold counts as positive. No models are refitted.", "",
             f"Each configuration covers {int(first.days)} held-out days from {int(first.participants)} participants.",
             f"Always predicting no onset gives {baseline:.2%} accuracy, 50% balanced accuracy,",
             "0% sensitivity and undefined precision (there are no positive predictions).", "",
             "| Model | Inputs | Accuracy | Balanced accuracy | Sensitivity | Precision |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {row.model_family} | {row.input_set} | {row.accuracy:.1%} | "
                     f"{row.balanced_accuracy:.1%} | {row.sensitivity:.1%} | {row.precision:.1%} |")
    lines += ["", "Sensitivity counts positive prediction days, not distinct onsets.",
              "Precision is the fraction of positive daily predictions followed by onset within three days.",
              "Balanced accuracy is the mean of sensitivity and specificity.",
              "See pooled_metrics.csv for confusion counts, fold_metrics.csv for stability,",
              "and metric_intervals.csv for 95% percentile intervals from the 2,000 original",
              "paired participant draws. Intervals retain fitted models and selected thresholds.",
              "Undefined metrics are excluded from their interval and contributing counts are saved."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_classification(results_root, destination):
    #### Record the assessment rule before selection, then save new reproducible result files.
    results_root, destination = Path(results_root).resolve(), Path(destination).resolve()
    check_results_destination(destination, [results_root / name for name, _ in FAMILIES.values()])
    loaded = {family: load_saved(results_root, family) for family in FAMILIES}
    reference = loaded["logistic_regression"][0]
    shared = ("observations.csv", "participant_folds.csv", "validation_fold_assignments.csv",
              "bootstrap_participants.csv")
    for frozen, _, _ in loaded.values():
        for name in shared:
            if file_sha256(frozen / name) != file_sha256(reference / name):
                raise ValueError("Model families do not share identical frozen cohorts/folds/draws")
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {str(path): file_sha256(path) for frozen, evaluation, _ in loaded.values()
              for path in [*(frozen / name for name in shared),
                           *(evaluation / name for name in ("test_predictions.csv",
                             "validation_predictions.csv", "selected_settings.csv", "output_manifest.json"))]}
    write_json(destination / "assessment_protocol.json", {
        "version": "onset-daily-classification-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_grid": THRESHOLDS.tolist(), "selection": "mean_validation_balanced_accuracy",
        "tie_break": "higher_threshold", "positive_rule": "probability >= threshold",
        "model_setting": "previously_selected_by_validation_AP", "bootstrap_resamples": 2000,
        "bootstrap_scope": "conditional_on_fitted_models_and_selected_thresholds",
        "inputs": inputs, "sources": source_hashes(), "versions": versions()})
    observations, assignments, validation_folds, draws = [pd.read_csv(reference / name) for name in shared]
    if len(draws) != 2000 or draws.replicate.duplicated().any():
        raise ValueError("Expected 2,000 distinct saved participant draws")
    classified = [classify_family(family, FAMILIES[family][1], saved, observations,
                                 assignments, validation_folds) for family, (_, _, saved) in loaded.items()]
    predictions, thresholds, scores = [pd.concat(items, ignore_index=True) for items in zip(*classified)]
    pooled, folds = summarise(predictions)
    bootstrap = bootstrap_classification(predictions, draws)
    intervals = [{"model_family": row.model_family, "input_set": row.input_set, "metric": metric,
                  "estimate": getattr(row, metric), **classification_interval(bootstrap.loc[
                      bootstrap.model_family.eq(row.model_family) & bootstrap.input_set.eq(row.input_set), metric])}
                 for row in pooled.itertuples(index=False) for metric in METRICS]
    tables = {"test_classifications": predictions, "selected_thresholds": thresholds,
              "threshold_validation_scores": scores, "pooled_metrics": pooled, "fold_metrics": folds,
              "bootstrap_metrics": bootstrap, "metric_intervals": pd.DataFrame(intervals)}
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    write_report(destination, pooled)
    write_json(destination / "output_manifest.json", {"files": {
        path.name: file_sha256(path) for path in destination.iterdir() if path.is_file()}})
    return pooled


def main():
    #### Run the additional daily assessment against existing saved probability evaluations.
    parser = argparse.ArgumentParser(description="Assess saved onset probabilities as daily yes/no predictions.")
    parser.add_argument("--results-root", type=Path, default=protocol.ROOT.parent / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_classification_2022_v1")
    args = parser.parse_args()
    print(run_classification(args.results_root, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
