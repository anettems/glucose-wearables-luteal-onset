"""Calendar-based alerts with one shared event-matching implementation."""

import numpy as np
import pandas as pd

import primary_protocol as protocol


ALERT_COLUMNS = ["id", "study_interval", "alert_day_in_study", "alert_probability",
                 "threshold", "consecutive_days"]
REFERENCE_COLUMNS = ["id", "study_interval", "reference_day_in_study", "evaluable", "matched",
                     "alert_day_in_study", "timing_error_days"]
MATCH_COLUMNS = ["matched", "reference_day_in_study", "timing_error_days"]


def validate_event_keys(data):
    if data.duplicated(protocol.KEY).any() or data[protocol.KEY].isna().any().any():
        raise ValueError("Event evaluation requires unique, non-missing observation keys")
    for column in protocol.KEY:
        values = data[column].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values > 0) & (values % 1 == 0)).all():
            raise ValueError("Event keys must be positive integers")
    if not data.study_interval.eq(2022).all():
        raise ValueError("Event evaluation is restricted to 2022")


def alert_days(days, probabilities, threshold, consecutive_days):
    """An absent day breaks pending confirmation; only a low score resets an episode."""
    days = np.asarray(days, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if days.shape != probabilities.shape or days.ndim != 1:
        raise ValueError("Expected aligned one-dimensional days and probabilities")
    if not (np.isfinite(days) & (days % 1 == 0)).all() or (np.diff(days) <= 0).any():
        raise ValueError("Alert days must be unique, increasing calendar days")
    if not (np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Alert probabilities must be finite and between zero and one")
    if consecutive_days not in (1, 2) or not 0 <= threshold <= 1:
        raise ValueError("Unsupported alert rule")
    alerts, active, streak, previous_day = [], False, 0, None
    for index, (day, probability) in enumerate(zip(days, probabilities, strict=True)):
        consecutive = previous_day is not None and day == previous_day + 1
        if active:
            active = probability >= threshold
        else:
            streak = (streak + 1 if consecutive else 1) if probability >= threshold else 0
            if streak >= consecutive_days:
                alerts.append(index)
                active, streak = True, 0
        previous_day = day
    return alerts


def generate_alerts(predictions, threshold, consecutive_days):
    validate_event_keys(predictions)
    rows = []
    for (participant, interval), group in predictions.sort_values(protocol.KEY).groupby(
        ["id", "study_interval"], sort=False,
    ):
        for index in alert_days(group.day_in_study, group.probability, threshold, consecutive_days):
            row = group.iloc[index]
            rows.append({
                "id": participant, "study_interval": interval,
                "alert_day_in_study": int(row.day_in_study), "alert_probability": float(row.probability),
                "threshold": threshold, "consecutive_days": consecutive_days,
            })
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def reference_evaluability(predictions, references):
    """Assess windows using calendar distance to at least one eligible prediction day."""
    days = {key: group.day_in_study.to_numpy(dtype=int) for key, group in
            predictions.groupby(["id", "study_interval"], sort=False)}
    flags = [np.any(np.abs(days.get((row.id, row.study_interval), np.array([])) - row.day_in_study)
                    <= protocol.ALERT_MATCH_WINDOW_DAYS) for row in references.itertuples(index=False)]
    return references.assign(evaluable=np.asarray(flags, dtype=bool))


def match_events(days, alert_values, reference_values):
    """Greedily choose the closest pair, then earlier alert; each event is used once."""
    evaluable = np.array([np.any(np.abs(days - day) <= protocol.ALERT_MATCH_WINDOW_DAYS)
                          for day in reference_values], dtype=bool)
    candidates = []
    for reference_index, reference in enumerate(reference_values):
        if evaluable[reference_index]:
            for alert_index, alert in enumerate(alert_values):
                error = abs(int(alert) - int(reference))
                if error <= protocol.ALERT_MATCH_WINDOW_DAYS:
                    candidates.append((error, int(alert), int(reference), alert_index, reference_index))
    used_alerts, used_references, matches = set(), set(), []
    for error, _, _, alert_index, reference_index in sorted(candidates):
        if alert_index not in used_alerts and reference_index not in used_references:
            used_alerts.add(alert_index)
            used_references.add(reference_index)
            matches.append((alert_index, reference_index, error))
    return evaluable, matches


def iter_event_matches(predictions, references, threshold, consecutive_days):
    """Share alert generation and matching between validation counts and detailed tables."""
    validate_event_keys(predictions)
    validate_event_keys(references)
    reference_groups = {key: group.day_in_study.to_numpy(dtype=int) for key, group in
                        references.sort_values(protocol.KEY).groupby(["id", "study_interval"])}
    for key, group in predictions.sort_values(protocol.KEY).groupby(["id", "study_interval"], sort=False):
        days = group.day_in_study.to_numpy(dtype=int)
        probabilities = group.probability.to_numpy(dtype=float)
        indices = alert_days(days, probabilities, threshold, consecutive_days)
        reference_values = reference_groups.get(key, np.array([], dtype=int))
        evaluable, matches = match_events(days, days[indices], reference_values)
        yield key, days[indices], probabilities[indices], reference_values, evaluable, matches


def event_metrics(matched, evaluable, alerts, timing_errors):
    missed, unmatched = evaluable - matched, alerts - matched
    denominator = 2 * matched + missed + unmatched
    return {
        "evaluable_reference_starts": evaluable, "matched_starts": matched,
        "missed_evaluable_starts": missed, "unmatched_alerts": unmatched,
        "event_f1": float(2 * matched / denominator) if denominator else 0.0,
        "timing_mae_days": float(np.mean(timing_errors)) if timing_errors else np.nan,
    }


def alert_metrics(predictions, references, threshold, consecutive_days):
    matched = evaluable = alerts = 0
    errors = []
    for _, alert_days_out, _, _, flags, matches in iter_event_matches(
        predictions, references, threshold, consecutive_days,
    ):
        matched += len(matches)
        evaluable += int(flags.sum())
        alerts += len(alert_days_out)
        errors.extend(error for _, _, error in matches)
    return event_metrics(matched, evaluable, alerts, errors)


def evaluate_alerts(predictions, references, threshold, consecutive_days):
    """Return auditable alert and reference tables using the same matching as tuning."""
    alerts, reference_rows, alert_rows, errors = [], [], [], []
    for key, alert_values, probabilities, reference_values, evaluable, matches in iter_event_matches(
        predictions, references, threshold, consecutive_days,
    ):
        by_alert = {alert: (reference, error) for alert, reference, error in matches}
        by_reference = {reference: (alert, error) for alert, reference, error in matches}
        errors.extend(error for _, _, error in matches)
        for index, day in enumerate(reference_values):
            alert, error = by_reference.get(index, (None, np.nan))
            reference_rows.append({
                "id": key[0], "study_interval": key[1], "reference_day_in_study": day,
                "evaluable": bool(evaluable[index]), "matched": alert is not None,
                "alert_day_in_study": alert_values[alert] if alert is not None else np.nan,
                "timing_error_days": error,
            })
        for index, day in enumerate(alert_values):
            row = dict(zip(ALERT_COLUMNS, [*key, day, probabilities[index], threshold, consecutive_days]))
            alerts.append(row)
            reference, error = by_alert.get(index, (None, np.nan))
            alert_rows.append({**row, "matched": reference is not None,
                               "reference_day_in_study": reference_values[reference] if reference is not None else np.nan,
                               "timing_error_days": error})
    reference_table = pd.DataFrame(reference_rows, columns=REFERENCE_COLUMNS)
    alert_table = pd.DataFrame(alert_rows, columns=ALERT_COLUMNS + MATCH_COLUMNS)
    metrics = event_metrics(len(errors), int(reference_table.evaluable.sum()), len(alerts), errors)
    return pd.DataFrame(alerts, columns=ALERT_COLUMNS), reference_table, alert_table, metrics


def select_alert_rule(validation_predictions, references):
    """Choose the fixed-grid alert rule using mean validation event F1."""
    rows = []
    for threshold in protocol.ALERT_THRESHOLDS:
        for consecutive_days in protocol.ALERT_CONSECUTIVE_DAYS:
            scores = []
            for validation_fold in range(1, protocol.INNER_FOLDS + 1):
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

