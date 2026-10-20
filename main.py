"""Prepare the 2022 inputs or run the locked primary model evaluation.

The default command rebuilds the one-day 14:00 analysis inputs. Use
--evaluate-primary to run only the pre-specified participant-held-out
logistic-regression evaluation from existing prepared inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import preprocessing
from model_inputs import FEATURE_SETS, prepare_datasets, replace_missing_wearables, validate_features


PRIMARY_SEED = 20261020
PRIMARY_PROTOCOL_VERSION = "2026-10-20-primary-v2"
OUTER_FOLDS = 5
INNER_FOLDS = 3
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_MAX_ITERATIONS = 1000
LOGISTIC_TOLERANCE = 1e-10
ALERT_THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
ALERT_CONSECUTIVE_DAYS = (1, 2)
ALERT_MATCH_WINDOW_DAYS = 3
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = PRIMARY_SEED + 1
MAX_FOLD_SWAP_PASSES = 20
KEY = preprocessing.KEY


def parse_main_arguments(argv=None):
    """Parse primary-evaluation options without changing preprocessing options."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluate-primary", action="store_true")
    parser.add_argument("--evaluation-dir", type=Path)
    options, remaining = parser.parse_known_args(argv)
    arguments = preprocessing.parse_arguments(remaining, prepare=True)
    if options.evaluation_dir is None:
        workspace = Path(__file__).resolve().parents[1]
        options.evaluation_dir = workspace / "results" / "primary_evaluation_2022"
    return options, arguments


def main(argv=None):
    options, args = parse_main_arguments(argv)
    if options.evaluate_primary:
        result = run_primary_evaluation(args.processed_dir, args.report_dir, options.evaluation_dir)
        print(f"Primary evaluation complete: {result['participants']} participants, "
              f"{result['days']} held-out days.")
        print(f"Results: {Path(options.evaluation_dir).resolve()}")
        return result

    args.temporal_review = True
    preprocessing.run_audit(args)
    data = prepare_datasets(args.processed_dir, args.report_dir)
    print(f"Features ready: {data.id.nunique()} participants, {len(data)} days, "
          f"{int(data.exact_luteal_onset.eq(1).sum())} luteal starts.")
    print(f"Dataset: {args.processed_dir / 'observations_2022.csv'}")
    print("Wearable replacement must be fitted separately on each training split.")
    return None


def create_evaluation_directory(path, processed_dir, report_dir):
    """Create an empty output directory without overwriting prior model results."""
    path = Path(path).resolve()
    processed_dir, report_dir = Path(processed_dir).resolve(), Path(report_dir).resolve()
    if path in {processed_dir, report_dir} or path in processed_dir.parents or path in report_dir.parents:
        raise ValueError("Evaluation output must not overlap prepared inputs or their report directory")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_primary_inputs(processed_dir, report_dir):
    """Load fixed features and full phase-label references for primary evaluation."""
    processed_dir, report_dir = Path(processed_dir), Path(report_dir)
    configuration = json.loads((report_dir / "run_config.json").read_text(encoding="utf-8"))
    if not configuration.get("features_ready") or configuration.get("prediction_cutoff") != "14:00:00":
        raise ValueError("Prepared inputs do not match the fixed 14:00 feature protocol")

    observations = pd.read_csv(processed_dir / "observations_2022.csv")
    validate_features(observations)
    references = pd.read_csv(processed_dir / "prediction_alignment_14_2022.csv")
    if references.duplicated(KEY).any() or not references.study_interval.eq(2022).all():
        raise ValueError("Expected unique 2022 reference-label keys")
    if "exact_luteal_onset" not in references:
        raise ValueError("Reference labels do not contain exact_luteal_onset")
    references = references.loc[
        references.id.isin(observations.id.unique()) & references.exact_luteal_onset.eq(1),
        KEY,
    ].sort_values(KEY).reset_index(drop=True)
    if references.empty:
        raise ValueError("No reference luteal starts are available for onset evaluation")
    return observations.sort_values(KEY).reset_index(drop=True), references, configuration


def participant_profile(observations, references):
    """Summarise the outcome and onset burden used to balance participant folds."""
    grouped = observations.groupby("id", sort=True)
    profile = grouped.agg(
        eligible_days=("luteal_status", "size"),
        luteal_days=("luteal_status", "sum"),
    ).reset_index()
    profile["non_luteal_days"] = profile["eligible_days"] - profile["luteal_days"]
    starts = references.groupby("id", sort=True).size().rename("reference_starts")
    profile = profile.join(starts, on="id").fillna({"reference_starts": 0})
    profile["reference_starts"] = profile["reference_starts"].astype(int)
    return profile


def allocation_score(assignments, metrics, fold_count):
    """Score a participant assignment by imbalance in prespecified quantities."""
    capacities = np.array([np.sum(assignments == fold) for fold in range(fold_count)], dtype=float)
    targets = capacities / capacities.sum()
    totals = np.zeros((fold_count, metrics.shape[1]), dtype=float)
    np.add.at(totals, assignments, metrics)
    expected = targets[:, None] * metrics.sum(axis=0)[None, :]
    scales = np.maximum(expected, 1.0)
    imbalance = np.square((totals - expected) / scales).sum()
    invalid = (totals[:, 0] <= 0) | (totals[:, 1] <= 0) | (totals[:, 0] <= totals[:, 1])
    missing_starts = totals[:, 2] <= 0
    return float(imbalance + 1_000_000 * invalid.sum() + 1_000_000 * missing_starts.sum())


def improve_assignment(assignments, metrics, fold_count):
    """Use deterministic improving participant swaps after a seeded random start."""
    best = assignments.copy()
    best_score = allocation_score(best, metrics, fold_count)
    for _ in range(MAX_FOLD_SWAP_PASSES):
        candidate_best, candidate_score = best, best_score
        for first in range(len(best)):
            for second in range(first + 1, len(best)):
                if best[first] == best[second]:
                    continue
                candidate = best.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                score = allocation_score(candidate, metrics, fold_count)
                if score + 1e-12 < candidate_score:
                    candidate_best, candidate_score = candidate, score
        if candidate_score + 1e-12 >= best_score:
            break
        best, best_score = candidate_best, candidate_score
    return best, best_score


def assign_balanced_folds(profile, fold_count, seed):
    """Produce a seed-controlled, capacity-balanced participant fold allocation."""
    if profile.id.nunique() < fold_count:
        raise ValueError(f"Need at least {fold_count} participants for participant-disjoint folds")
    rng = np.random.default_rng(seed)
    profile = profile.reset_index(drop=True)
    participant_count = len(profile)
    capacities = np.full(fold_count, participant_count // fold_count, dtype=int)
    capacities[:participant_count % fold_count] += 1
    metrics = profile[["eligible_days", "luteal_days", "reference_starts"]].to_numpy(dtype=float)
    targets = capacities[:, None] / participant_count * metrics.sum(axis=0)[None, :]
    scales = np.maximum(targets, 1.0)
    order = profile.assign(_tie=rng.random(participant_count)).sort_values(
        ["reference_starts", "luteal_days", "eligible_days", "_tie"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).index.to_numpy()
    assignment = np.full(participant_count, -1, dtype=int)
    counts = np.zeros(fold_count, dtype=int)
    totals = np.zeros((fold_count, metrics.shape[1]), dtype=float)
    for participant in order:
        candidates = np.flatnonzero(counts < capacities)
        candidate_order = rng.permutation(candidates)
        scores = []
        for fold in candidate_order:
            projected = totals.copy()
            projected[fold] += metrics[participant]
            scores.append(np.square((projected - targets) / scales).sum())
        chosen = candidate_order[int(np.argmin(scores))]
        assignment[participant] = chosen
        counts[chosen] += 1
        totals[chosen] += metrics[participant]
    best_assignment, _ = improve_assignment(assignment, metrics, fold_count)
    result = profile[["id"]].copy()
    result["fold"] = best_assignment + 1
    return result.sort_values("id").reset_index(drop=True)


def fold_counts(observations, references, assignments, fold_column):
    """Return planned feasibility counts for one participant-fold assignment."""
    labelled = observations.merge(assignments[["id", fold_column]], on="id", validate="many_to_one")
    starts = references.merge(assignments[["id", fold_column]], on="id", validate="many_to_one")
    summary = labelled.groupby(fold_column, sort=True).agg(
        participants=("id", "nunique"),
        eligible_days=("luteal_status", "size"),
        luteal_days=("luteal_status", "sum"),
    )
    summary["non_luteal_days"] = summary["eligible_days"] - summary["luteal_days"]
    summary["reference_starts"] = starts.groupby(fold_column).size().reindex(summary.index, fill_value=0)
    return summary.reset_index()


def validate_fold_assignment(observations, references, assignments, fold_column, fold_count):
    """Reject leakage, missing classes, or folds without an onset reference event."""
    if assignments.id.duplicated().any() or set(assignments.id) != set(observations.id):
        raise ValueError("Fold assignment must contain each eligible participant exactly once")
    expected = set(range(1, fold_count + 1))
    observed = set(assignments[fold_column])
    if observed != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, found {sorted(observed)}")
    counts = fold_counts(observations, references, assignments, fold_column)
    invalid = counts.loc[
        counts.luteal_days.le(0)
        | counts.non_luteal_days.le(0)
        | counts.reference_starts.le(0),
    ]
    if not invalid.empty:
        raise ValueError(f"Fold feasibility failed for {fold_column}: {invalid.to_dict('records')}")
    return counts


def build_fold_assignments(observations, references):
    """Create and validate all saved outer and nested participant allocations."""
    profile = participant_profile(observations, references)
    outer = assign_balanced_folds(profile, OUTER_FOLDS, PRIMARY_SEED).rename(columns={"fold": "outer_test_fold"})
    feasibility = validate_fold_assignment(
        observations, references, outer, "outer_test_fold", OUTER_FOLDS,
    ).assign(stage="outer")

    inner_parts = []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        development_ids = outer.loc[outer.outer_test_fold.ne(outer_fold), "id"]
        development = observations.loc[observations.id.isin(development_ids)]
        development_references = references.loc[references.id.isin(development_ids)]
        development_profile = participant_profile(development, development_references)
        inner = assign_balanced_folds(
            development_profile, INNER_FOLDS, PRIMARY_SEED + outer_fold,
        ).rename(columns={"fold": "validation_fold"})
        counts = validate_fold_assignment(
            development, development_references, inner, "validation_fold", INNER_FOLDS,
        )
        counts["stage"] = "validation"
        counts["outer_test_fold"] = outer_fold
        feasibility = pd.concat([feasibility, counts], ignore_index=True)
        inner["outer_test_fold"] = outer_fold
        inner_parts.append(inner)
    inner = pd.concat(inner_parts, ignore_index=True)
    return outer, inner, feasibility


def sigmoid(values):
    """Numerically stable logistic transformation."""
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


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
        max_iter=LOGISTIC_MAX_ITERATIONS,
        tol=LOGISTIC_TOLERANCE,
        random_state=PRIMARY_SEED,
    )


def fit_l2_logistic(features, labels, inverse_penalty_strength):
    """Fit the fixed estimator and return its coefficients for focused checks."""
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Logistic-regression training data must contain both outcome classes")
    features = np.asarray(features, dtype=float)
    model = create_logistic_model(inverse_penalty_strength, len(labels))
    model.fit(features, labels)
    coefficients = np.r_[model.intercept_, model.coef_[0]]
    return coefficients, int(model.n_iter_[0])


def fit_and_predict(training, evaluation, feature_columns, inverse_penalty_strength):
    """Fit split-specific replacement, scaling and logistic regression, then predict."""
    training_filled, evaluation_filled, medians = replace_missing_wearables(training, evaluation)
    training_features = training_filled[feature_columns].to_numpy(dtype=float)
    evaluation_features = evaluation_filled[feature_columns].to_numpy(dtype=float)
    model = make_pipeline(
        StandardScaler(),
        create_logistic_model(inverse_penalty_strength, len(training_features)),
    )
    model.fit(training_features, training_filled.luteal_status.to_numpy(dtype=int))
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
    }
    return probabilities, parameters


def average_precision(labels, probabilities):
    """Calculate non-interpolated average precision with scikit-learn."""
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        raise ValueError("Average precision requires at least one positive outcome")
    return float(average_precision_score(labels, probabilities))


def roc_auc(labels, probabilities):
    """Calculate ROC-AUC with scikit-learn."""
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        raise ValueError("ROC-AUC requires both outcome classes")
    return float(roc_auc_score(labels, probabilities))


def daily_metrics(predictions):
    """Summarise daily held-out probabilities without using participant identifiers as inputs."""
    labels = predictions.luteal_status.to_numpy(dtype=int)
    probabilities = predictions.probability.to_numpy(dtype=float)
    return {
        "observations": len(predictions),
        "luteal_days": int(labels.sum()),
        "luteal_prevalence": float(labels.mean()),
        "average_precision": average_precision(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
    }


def tune_logistic_model(development, validation_assignments, feature_columns):
    """Choose C from participant-disjoint validation predictions."""
    if validation_assignments.id.duplicated().any():
        raise ValueError("Each development participant must have one validation-fold assignment")
    assignment = validation_assignments.set_index("id").validation_fold
    if not set(development.id).issubset(set(assignment.index)):
        raise ValueError("Validation assignments do not cover the development participants")

    tuning_rows, candidate_predictions = [], {}
    for inverse_penalty_strength in LOGISTIC_C_VALUES:
        validation_parts = []
        for validation_fold in range(1, INNER_FOLDS + 1):
            validation_ids = assignment.loc[assignment.eq(validation_fold)].index
            validation = development.loc[development.id.isin(validation_ids)].copy()
            training = development.loc[~development.id.isin(validation_ids)].copy()
            probabilities, _ = fit_and_predict(
                training, validation, feature_columns, inverse_penalty_strength,
            )
            part = validation[KEY + ["luteal_status"]].copy()
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


def alert_days(days, probabilities, threshold, consecutive_days):
    """Return alert-day indices without bridging missing prediction days."""
    alerts = []
    active, streak, previous_day = False, 0, None
    for index, (day, probability) in enumerate(zip(days, probabilities, strict=True)):
            is_consecutive = previous_day is not None and day == previous_day + 1
            positive = probability >= threshold
            if active:
                if probability < threshold:
                    active = False
                previous_day = day
                continue
            if not positive:
                streak = 0
            elif is_consecutive:
                streak += 1
            else:
                streak = 1
            if streak >= consecutive_days:
                alerts.append(index)
                active, streak = True, 0
            previous_day = day
    return alerts


def generate_alerts(predictions, threshold, consecutive_days):
    """Generate alerts without bridging missing prediction days or backdating."""
    alerts = []
    for _, group in predictions.sort_values(KEY).groupby(["id", "study_interval"], sort=False):
        days = group.day_in_study.to_numpy(dtype=int)
        probabilities = group.probability.to_numpy(dtype=float)
        for index in alert_days(days, probabilities, threshold, consecutive_days):
            alerts.append({
                "id": group.id.iloc[0],
                "study_interval": group.study_interval.iloc[0],
                "alert_day_in_study": days[index],
                "alert_probability": probabilities[index],
                "threshold": threshold,
                "consecutive_days": consecutive_days,
            })
    return pd.DataFrame(alerts, columns=[
        "id", "study_interval", "alert_day_in_study", "alert_probability",
        "threshold", "consecutive_days",
    ])


def alert_metrics(predictions, references, threshold, consecutive_days):
    """Calculate event metrics without constructing detailed event tables."""
    reference_groups = {
        (participant, interval): frame.day_in_study.to_numpy(dtype=int)
        for (participant, interval), frame in references.groupby(["id", "study_interval"], sort=False)
    }
    matched_total = missed_total = unmatched_total = evaluable_total = 0
    for key, group in predictions.sort_values(KEY).groupby(["id", "study_interval"], sort=False):
        days = group.day_in_study.to_numpy(dtype=int)
        probabilities = group.probability.to_numpy(dtype=float)
        alert_indices = alert_days(days, probabilities, threshold, consecutive_days)
        alert_values = days[alert_indices]
        reference_values = reference_groups.get(key, np.array([], dtype=int))
        evaluable = np.array([
            np.any(np.abs(days - reference) <= ALERT_MATCH_WINDOW_DAYS)
            for reference in reference_values
        ], dtype=bool)
        candidates = []
        for reference_index, reference in enumerate(reference_values):
            if not evaluable[reference_index]:
                continue
            for alert_index, alert in enumerate(alert_values):
                error = abs(int(alert) - int(reference))
                if error <= ALERT_MATCH_WINDOW_DAYS:
                    candidates.append((error, int(alert), int(reference), alert_index, reference_index))
        used_alerts, used_references = set(), set()
        for _, _, _, alert_index, reference_index in sorted(candidates):
            if alert_index not in used_alerts and reference_index not in used_references:
                used_alerts.add(alert_index)
                used_references.add(reference_index)
        matched_total += len(used_references)
        evaluable_total += int(evaluable.sum())
        missed_total += int(evaluable.sum()) - len(used_references)
        unmatched_total += len(alert_values) - len(used_alerts)
    denominator = 2 * matched_total + missed_total + unmatched_total
    return {
        "evaluable_reference_starts": evaluable_total,
        "matched_starts": matched_total,
        "missed_evaluable_starts": missed_total,
        "unmatched_alerts": unmatched_total,
        "event_f1": float(2 * matched_total / denominator) if denominator else 0.0,
        "timing_mae_days": np.nan,
    }


def evaluate_alerts(predictions, references, threshold, consecutive_days):
    """Match alert and reference events one-to-one within the fixed timing window."""
    alerts = generate_alerts(predictions, threshold, consecutive_days)
    participant_ids = predictions.id.unique()
    references = references.loc[references.id.isin(participant_ids)].copy()
    reference_rows, alert_rows, candidates = [], [], []

    for reference in references.itertuples(index=False):
        participant_predictions = predictions.loc[
            (predictions.id == reference.id)
            & (predictions.study_interval == reference.study_interval),
            "day_in_study",
        ]
        evaluable = participant_predictions.sub(reference.day_in_study).abs().le(ALERT_MATCH_WINDOW_DAYS).any()
        reference_rows.append({
            "id": reference.id,
            "study_interval": reference.study_interval,
            "reference_day_in_study": reference.day_in_study,
            "evaluable": bool(evaluable),
            "matched": False,
            "alert_day_in_study": np.nan,
            "timing_error_days": np.nan,
        })
    for _, alert in alerts.iterrows():
        alert_rows.append({
            **alert.to_dict(),
            "matched": False,
            "reference_day_in_study": np.nan,
            "timing_error_days": np.nan,
        })

    reference_results = pd.DataFrame(reference_rows, columns=[
        "id", "study_interval", "reference_day_in_study", "evaluable", "matched",
        "alert_day_in_study", "timing_error_days",
    ])
    alert_results = pd.DataFrame(alert_rows, columns=[
        "id", "study_interval", "alert_day_in_study", "alert_probability",
        "threshold", "consecutive_days", "matched", "reference_day_in_study",
        "timing_error_days",
    ])
    if not alerts.empty and not reference_results.empty:
        for reference_index, reference in reference_results.loc[reference_results.evaluable].iterrows():
            possible = alerts.loc[
                (alerts.id == reference.id)
                & (alerts.study_interval == reference.study_interval)
                & alerts.alert_day_in_study.sub(reference.reference_day_in_study).abs().le(ALERT_MATCH_WINDOW_DAYS),
            ]
            for alert_index, alert in possible.iterrows():
                candidates.append((
                    abs(int(alert.alert_day_in_study) - int(reference.reference_day_in_study)),
                    int(alert.alert_day_in_study),
                    int(reference.reference_day_in_study),
                    alert_index,
                    reference_index,
                ))
    for error, _, _, alert_index, reference_index in sorted(candidates):
        if alert_results.at[alert_index, "matched"] or reference_results.at[reference_index, "matched"]:
            continue
        alert_results.at[alert_index, "matched"] = True
        alert_results.at[alert_index, "reference_day_in_study"] = reference_results.at[
            reference_index, "reference_day_in_study"
        ]
        alert_results.at[alert_index, "timing_error_days"] = error
        reference_results.at[reference_index, "matched"] = True
        reference_results.at[reference_index, "alert_day_in_study"] = alert_results.at[
            alert_index, "alert_day_in_study"
        ]
        reference_results.at[reference_index, "timing_error_days"] = error

    matched = int(reference_results.matched.sum())
    missed = int((reference_results.evaluable & ~reference_results.matched).sum())
    unmatched_alerts = int((~alert_results.matched).sum())
    denominator = 2 * matched + missed + unmatched_alerts
    metrics = {
        "evaluable_reference_starts": int(reference_results.evaluable.sum()),
        "matched_starts": matched,
        "missed_evaluable_starts": missed,
        "unmatched_alerts": unmatched_alerts,
        "event_f1": float(2 * matched / denominator) if denominator else 0.0,
        "timing_mae_days": float(reference_results.loc[
            reference_results.matched, "timing_error_days"
        ].mean()) if matched else np.nan,
    }
    return alerts, reference_results, alert_results, metrics


def select_alert_rule(validation_predictions, references):
    """Choose the fixed-grid alert rule using mean validation event F1."""
    rows = []
    for threshold in ALERT_THRESHOLDS:
        for consecutive_days in ALERT_CONSECUTIVE_DAYS:
            scores = []
            for validation_fold in range(1, INNER_FOLDS + 1):
                predicted = validation_predictions.loc[
                    validation_predictions.validation_fold.eq(validation_fold)
                ]
                metrics = alert_metrics(
                    predicted, references, threshold, consecutive_days,
                )
                scores.append(metrics["event_f1"])
                rows.append({
                    "threshold": threshold,
                    "consecutive_days": consecutive_days,
                    "validation_fold": validation_fold,
                    **metrics,
                })
            rows.append({
                "threshold": threshold,
                "consecutive_days": consecutive_days,
                "validation_fold": "mean",
                "event_f1": float(np.mean(scores)),
                "evaluable_reference_starts": np.nan,
                "matched_starts": np.nan,
                "missed_evaluable_starts": np.nan,
                "unmatched_alerts": np.nan,
                "timing_mae_days": np.nan,
            })
    scores = pd.DataFrame(rows)
    means = scores.loc[scores.validation_fold.eq("mean")]
    selected = means.sort_values(
        ["event_f1", "threshold", "consecutive_days"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    return float(selected.threshold), int(selected.consecutive_days), scores


def parameter_rows(outer_fold, input_set, parameters):
    """Convert split-fitted parameters to auditable long-form records."""
    rows = []
    for category, values in parameters.items():
        if isinstance(values, dict):
            rows.extend({
                "outer_test_fold": outer_fold,
                "input_set": input_set,
                "parameter_category": category,
                "parameter": name,
                "value": value,
            } for name, value in values.items())
        else:
            rows.append({
                "outer_test_fold": outer_fold,
                "input_set": input_set,
                "parameter_category": category,
                "parameter": category,
                "value": values,
            })
    return rows


def validate_paired_predictions(predictions, observations):
    """Ensure every input set has exactly one held-out prediction for every key."""
    expected = observations[KEY].sort_values(KEY).reset_index(drop=True)
    for input_set in FEATURE_SETS:
        subset = predictions.loc[predictions.input_set.eq(input_set)]
        if subset.duplicated(KEY).any() or len(subset) != len(expected):
            raise ValueError("Expected one held-out prediction per key and input set")
        actual = subset[KEY].sort_values(KEY).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError("Input sets do not have identical held-out observation keys")
        if subset.probability.lt(0).any() or subset.probability.gt(1).any():
            raise ValueError("Predicted probabilities must lie between zero and one")


def bootstrap_metrics(predictions):
    """Calculate paired participant-bootstrap daily metrics and intervals."""
    grouped = {
        input_set: {
            participant: frame
            for participant, frame in predictions.loc[
                predictions.input_set.eq(input_set)
            ].groupby("id", sort=True)
        }
        for input_set in FEATURE_SETS
    }
    participants = sorted(grouped["wearable"])
    if any(sorted(grouped[input_set]) != participants for input_set in FEATURE_SETS):
        raise ValueError("Input sets do not contain identical held-out participants")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for replicate in range(BOOTSTRAP_RESAMPLES):
        selected = rng.choice(participants, size=len(participants), replace=True)
        row = {"replicate": replicate + 1}
        for input_set in FEATURE_SETS:
            sample = pd.concat([grouped[input_set][participant] for participant in selected], ignore_index=True)
            row[f"{input_set}_average_precision"] = average_precision(
                sample.luteal_status, sample.probability,
            )
            row[f"{input_set}_roc_auc"] = roc_auc(sample.luteal_status, sample.probability)
        row["combined_minus_wearable_average_precision"] = (
            row["combined_average_precision"] - row["wearable_average_precision"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def interval(values):
    """Return the fixed 95% percentile confidence interval."""
    lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(lower), float(upper)


def summary_tables(predictions, bootstrap):
    """Create primary pooled daily metrics and the paired primary comparison."""
    rows = []
    for input_set in FEATURE_SETS:
        metrics = daily_metrics(predictions.loc[predictions.input_set.eq(input_set)])
        ap_lower, ap_upper = interval(bootstrap[f"{input_set}_average_precision"])
        auc_lower, auc_upper = interval(bootstrap[f"{input_set}_roc_auc"])
        rows.extend([
            {"input_set": input_set, "metric": "average_precision",
             "estimate": metrics["average_precision"], "ci_lower": ap_lower, "ci_upper": ap_upper},
            {"input_set": input_set, "metric": "roc_auc",
             "estimate": metrics["roc_auc"], "ci_lower": auc_lower, "ci_upper": auc_upper},
            {"input_set": input_set, "metric": "luteal_prevalence",
             "estimate": metrics["luteal_prevalence"], "ci_lower": np.nan, "ci_upper": np.nan},
        ])
    difference = (
        daily_metrics(predictions.loc[predictions.input_set.eq("combined")])["average_precision"]
        - daily_metrics(predictions.loc[predictions.input_set.eq("wearable")])["average_precision"]
    )
    lower, upper = interval(bootstrap["combined_minus_wearable_average_precision"])
    paired = pd.DataFrame([{
        "comparison": "combined_minus_wearable",
        "metric": "average_precision_difference",
        "estimate": difference,
        "ci_lower": lower,
        "ci_upper": upper,
    }])
    return pd.DataFrame(rows), paired


def evaluate_test_alerts(predictions, references, settings):
    """Evaluate selected alert rules separately within their outer test folds."""
    alerts, reference_results, alert_results, metric_rows = [], [], [], []
    for setting in settings.itertuples(index=False):
        test_predictions = predictions.loc[
            (predictions.input_set == setting.input_set)
            & (predictions.outer_test_fold == setting.outer_test_fold)
        ]
        generated, references_out, alerts_out, metrics = evaluate_alerts(
            test_predictions, references, setting.threshold, setting.consecutive_days,
        )
        for frame, target in [
            (generated, alerts),
            (references_out, reference_results),
            (alerts_out, alert_results),
        ]:
            frame = frame.copy()
            frame["input_set"] = setting.input_set
            frame["outer_test_fold"] = setting.outer_test_fold
            target.append(frame)
        metric_rows.append({
            "input_set": setting.input_set,
            "outer_test_fold": setting.outer_test_fold,
            "threshold": setting.threshold,
            "consecutive_days": setting.consecutive_days,
            **metrics,
        })
    return (
        pd.concat(alerts, ignore_index=True),
        pd.concat(reference_results, ignore_index=True),
        pd.concat(alert_results, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def run_primary_evaluation(processed_dir, report_dir, evaluation_dir):
    """Run the locked logistic-regression evaluation and save reproducible outputs."""
    processed_dir, report_dir = Path(processed_dir).resolve(), Path(report_dir).resolve()
    output_dir = create_evaluation_directory(evaluation_dir, processed_dir, report_dir)
    observations, references, preparation_config = load_primary_inputs(processed_dir, report_dir)
    protocol = {
        "version": PRIMARY_PROTOCOL_VERSION,
        "study_interval": 2022,
        "seed": PRIMARY_SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "logistic_inverse_penalty_strengths": LOGISTIC_C_VALUES,
        "model_implementation": "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)",
        "model_solver": "lbfgs",
        "sklearn_c_conversion": "protocol_C / number_of_training_observations",
        "model_max_iterations": LOGISTIC_MAX_ITERATIONS,
        "model_tolerance": LOGISTIC_TOLERANCE,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "primary_metric": "pooled_out_of_fold_average_precision",
        "secondary_daily_metric": "pooled_out_of_fold_roc_auc",
        "alert_thresholds": ALERT_THRESHOLDS,
        "alert_consecutive_days": ALERT_CONSECUTIVE_DAYS,
        "alert_selection_metric": "mean_validation_event_f1",
        "alert_match_window_days": ALERT_MATCH_WINDOW_DAYS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_interval": "95_percentile_participant_resampling",
        "fold_allocation": (
            "seed-controlled greedy capacity-balanced allocation followed by "
            f"at most {MAX_FOLD_SWAP_PASSES} improving participant-swap passes"
        ),
    }
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8",
    )

    outer, inner, feasibility = build_fold_assignments(observations, references)
    outer.to_csv(output_dir / "participant_folds.csv", index=False)
    inner.to_csv(output_dir / "validation_fold_assignments.csv", index=False)
    feasibility.to_csv(output_dir / "fold_feasibility.csv", index=False)

    prediction_parts, tuning_parts, alert_tuning_parts, validation_parts = [], [], [], []
    setting_rows, fitted_parameter_rows = [], []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        testing_ids = outer.loc[outer.outer_test_fold.eq(outer_fold), "id"]
        development = observations.loc[~observations.id.isin(testing_ids)].copy()
        testing = observations.loc[observations.id.isin(testing_ids)].copy()
        validation_assignments = inner.loc[
            inner.outer_test_fold.eq(outer_fold), ["id", "validation_fold"]
        ]
        development_references = references.loc[references.id.isin(development.id.unique())]

        for input_set, feature_columns in FEATURE_SETS.items():
            selected_c, tuning, validation_predictions = tune_logistic_model(
                development, validation_assignments, feature_columns,
            )
            threshold, consecutive_days, alert_tuning = select_alert_rule(
                validation_predictions, development_references,
            )
            probabilities, parameters = fit_and_predict(
                development, testing, feature_columns, selected_c,
            )
            prediction = testing[KEY + ["luteal_status"]].copy()
            prediction["probability"] = probabilities
            prediction["input_set"] = input_set
            prediction["outer_test_fold"] = outer_fold
            prediction_parts.append(prediction)

            tuning["input_set"] = input_set
            tuning["outer_test_fold"] = outer_fold
            tuning_parts.append(tuning)
            alert_tuning["input_set"] = input_set
            alert_tuning["outer_test_fold"] = outer_fold
            alert_tuning_parts.append(alert_tuning)
            validation_predictions["input_set"] = input_set
            validation_predictions["outer_test_fold"] = outer_fold
            validation_predictions["inverse_penalty_strength"] = selected_c
            validation_parts.append(validation_predictions)
            setting_rows.append({
                "input_set": input_set,
                "outer_test_fold": outer_fold,
                "inverse_penalty_strength": selected_c,
                "threshold": threshold,
                "consecutive_days": consecutive_days,
            })
            fitted_parameter_rows.extend(parameter_rows(outer_fold, input_set, parameters))

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["input_set"] + KEY,
    ).reset_index(drop=True)
    validate_paired_predictions(predictions, observations)
    settings = pd.DataFrame(setting_rows).sort_values(["input_set", "outer_test_fold"])
    alerts, reference_events, alert_events, event_metrics = evaluate_test_alerts(
        predictions, references, settings,
    )

    fold_metric_rows = []
    for input_set in FEATURE_SETS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            daily = daily_metrics(predictions.loc[
                (predictions.input_set == input_set)
                & (predictions.outer_test_fold == outer_fold)
            ])
            event = event_metrics.loc[
                (event_metrics.input_set == input_set)
                & (event_metrics.outer_test_fold == outer_fold)
            ].iloc[0].to_dict()
            fold_metric_rows.append({**daily, **event})
    fold_metrics = pd.DataFrame(fold_metric_rows)

    bootstrap = bootstrap_metrics(predictions)
    summary, paired = summary_tables(predictions, bootstrap)
    predictions.to_csv(output_dir / "model_predictions.csv", index=False)
    pd.concat(tuning_parts, ignore_index=True).to_csv(output_dir / "tuning_results.csv", index=False)
    pd.concat(validation_parts, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False,
    )
    pd.concat(alert_tuning_parts, ignore_index=True).to_csv(
        output_dir / "alert_tuning_results.csv", index=False,
    )
    settings.to_csv(output_dir / "selected_settings.csv", index=False)
    pd.DataFrame(fitted_parameter_rows).to_csv(output_dir / "fitted_parameters.csv", index=False)
    alerts.to_csv(output_dir / "onset_alerts.csv", index=False)
    reference_events.to_csv(output_dir / "onset_reference_events.csv", index=False)
    alert_events.to_csv(output_dir / "onset_alert_events.csv", index=False)
    event_metrics.to_csv(output_dir / "onset_fold_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "model_fold_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_metrics.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)

    run_configuration = {
        "protocol_version": PRIMARY_PROTOCOL_VERSION,
        "prepared_input_config": preparation_config,
        "processed_dir": str(processed_dir),
        "report_dir": str(report_dir),
        "evaluation_dir": str(output_dir),
        "participants": int(observations.id.nunique()),
        "eligible_days": int(len(observations)),
        "eligible_observation_starts": int(observations.exact_luteal_onset.eq(1).sum()),
        "reference_starts_in_full_alignment": int(len(references)),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "model_ready": True,
        "entry_point_sha256": preprocessing.file_sha256(Path(__file__)),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_configuration, indent=2) + "\n", encoding="utf-8",
    )
    return {
        "participants": int(observations.id.nunique()),
        "days": int(len(observations)),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
