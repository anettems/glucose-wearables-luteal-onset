"""Previous-calendar-day glucose shape and dynamics, without filling gaps."""

import numpy as np
import pandas as pd

import preprocessing as prep
from onset_alerts import validate_event_keys
from within_day_analysis import protocol


def period_coverage(seconds, start, end):
    """Count the union of nominal sensor intervals, including boundary overlap."""
    left = np.maximum(seconds, start)
    right = np.minimum(seconds + protocol.PAIR_SECONDS, end)
    keep = right > left
    left, right = left[keep], right[keep]
    previous = np.r_[start, np.maximum.accumulate(right)[:-1]]
    return float(np.maximum(0, right - np.maximum(left, previous)).sum() / (end - start))


def pair_features(seconds, values):
    elapsed = np.diff(seconds)
    usable = np.abs(elapsed - protocol.PAIR_SECONDS) <= protocol.PAIR_TOLERANCE_SECONDS
    count = int(usable.sum())
    rate, correlation = np.nan, np.nan
    if count >= protocol.MINIMUM_PAIRS:
        first, second = values[:-1][usable], values[1:][usable]
        rate = float(np.mean(np.abs(second - first) / (elapsed[usable] / 60)))
        # Constant sequences have no defined Pearson correlation; do not invent one.
        if np.std(first) > 0 and np.std(second) > 0:
            correlation = float(np.clip(np.corrcoef(first, second)[0, 1], -1, 1))
    return {"glucose_pattern_pairs": count,
            "glucose_mean_absolute_rate_mmol_l_min": rate,
            "glucose_lag5min_correlation": correlation}


def summarise_day(seconds, values):
    order = np.argsort(seconds)
    seconds, values = seconds[order], values[order]
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(seconds).all():
        raise ValueError("Expected finite, non-empty glucose observations")
    if (values <= 0).any() or (seconds < 0).any() or (seconds >= 86400).any():
        raise ValueError("Glucose values or within-day times are invalid")
    if (np.diff(seconds) <= 0).any():
        raise ValueError("Duplicate glucose timestamps within a participant-day")
    q25, median, q75, q90 = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    night = seconds < protocol.NIGHT_END_SECONDS
    night_coverage = period_coverage(seconds, 0, protocol.NIGHT_END_SECONDS)
    day_coverage = period_coverage(seconds, protocol.NIGHT_END_SECONDS, 86400)
    contrast = np.nan
    if min(night_coverage, day_coverage) >= protocol.MINIMUM_PERIOD_COVERAGE and night.any() and (~night).any():
        contrast = float(np.median(values[night]) - np.median(values[~night]))
    return {"glucose_iqr_mmol_l": float(q75 - q25),
            "glucose_p90_minus_median_mmol_l": float(q90 - median),
            "glucose_night_minus_day_mmol_l": contrast,
            "glucose_pattern_samples": len(values), "glucose_night_coverage": night_coverage,
            "glucose_day_coverage": day_coverage, **pair_features(seconds, values)}


def daily_patterns(glucose):
    """Use audited 2022 records and lag by calendar day, never by available row."""
    if not glucose.study_interval.eq(2022).all():
        raise ValueError("Within-day analysis requires only the 2022 collection interval")
    usable = glucose.loc[prep.primary_records(glucose, "glucose_value")]
    rows = []
    for key, group in usable.groupby(prep.KEY, sort=True):
        seconds = group.audit_end_seconds.to_numpy(dtype=float) - key[2] * 86400
        values = group.glucose_value.to_numpy(dtype=float)
        rows.append({**dict(zip(prep.KEY, key)), **summarise_day(seconds, values)})
    result = pd.DataFrame(rows, columns=prep.KEY + protocol.PATTERN_FEATURES + protocol.DIAGNOSTICS)
    result["day_in_study"] += 1
    validate_event_keys(result)
    return result


def select_cohort(baseline, patterns):
    validate_event_keys(baseline)
    validate_event_keys(patterns)
    audit = baseline.merge(patterns, on=prep.KEY, how="left", validate="one_to_one")
    for feature in protocol.PATTERN_FEATURES:
        audit[f"{feature}__available"] = np.isfinite(audit[feature])
    audit["within_day_eligible"] = np.isfinite(audit[protocol.PATTERN_FEATURES]).all(axis=1)
    cohort = audit.loc[audit.within_day_eligible].copy().reset_index(drop=True)
    if cohort.empty:
        raise ValueError("No eligible within-day glucose observations")
    return audit, cohort
