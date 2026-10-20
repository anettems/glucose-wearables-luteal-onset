"""Descriptive summaries of eligible days and saved held-out predictions."""

import numpy as np
import pandas as pd

from daily_metrics import daily_metrics, validate_paired_predictions
from model_inputs import FEATURE_SETS


FEATURE_LABELS = {
    "glucose_median_mmol_l": "Median glucose (mmol/L)",
    "glucose_cv_percent": "Glucose coefficient of variation (%)",
    "sleeping_hr_bpm": "Sleeping heart rate (beats/min)",
    "log_median_rmssd_ms": "ln[RMSSD / (1 ms)]",
    "skin_temperature_c": "Skin temperature (°C)",
}
MODEL_LABELS = {"glucose": "Glucose only", "wearable": "Wearable only", "combined": "Combined"}


def participant_coverage(observations, alignment):
    """Keep excluded participants visible; span counts calendar days, including gaps."""
    coverage = alignment.groupby(["id", "study_interval"]).agg(
        aligned_days=("day_in_study", "size"), first_aligned_day=("day_in_study", "min"),
        last_aligned_day=("day_in_study", "max"),
    )
    eligible = observations.groupby(["id", "study_interval"]).agg(
        eligible_days=("day_in_study", "size"), luteal_days=("luteal_status", "sum"),
        first_eligible_day=("day_in_study", "min"), last_eligible_day=("day_in_study", "max"),
    )
    coverage = coverage.join(eligible)
    for name in ["eligible_days", "luteal_days"]:
        coverage[name] = coverage[name].fillna(0).astype(int)
    coverage["non_luteal_days"] = coverage.eligible_days - coverage.luteal_days
    coverage["in_modelling_cohort"] = coverage.eligible_days.gt(0)
    coverage["aligned_calendar_span_days"] = coverage.last_aligned_day - coverage.first_aligned_day + 1
    coverage["eligible_fraction_of_aligned_days"] = coverage.eligible_days / coverage.aligned_days
    coverage["luteal_fraction_of_eligible_days"] = coverage.luteal_days / coverage.eligible_days.replace(0, np.nan)
    coverage["eligible_calendar_span_days"] = coverage.last_eligible_day - coverage.first_eligible_day + 1
    coverage["ineligible_days_within_eligible_span"] = coverage.eligible_calendar_span_days - coverage.eligible_days
    return coverage.reset_index()


def descriptive_tables(observations, alignment):
    """Summarise observed feature values before any split-specific replacement."""
    features = list(FEATURE_LABELS)
    missing_rows = []
    for participant, group in [("all", observations), *observations.groupby("id", sort=True)]:
        for feature in features:
            missing_rows.append({
                "id": participant, "feature": feature, "eligible_days": len(group),
                "observed_days": int(group[feature].notna().sum()),
                "missing_days": int(group[feature].isna().sum()),
                "missing_fraction": float(group[feature].isna().mean()),
            })
    distributions = observations[features].describe(percentiles=[0.25, 0.5, 0.75]).T
    distributions.index.name = "feature"
    distributions = distributions.rename(columns={"count": "observed_days", "50%": "median", "25%": "q25", "75%": "q75"})
    balance = observations.groupby("luteal_status").agg(
        eligible_days=("id", "size"), participants_with_outcome=("id", "nunique"),
    ).reindex([0, 1], fill_value=0).reset_index()
    balance["fraction_of_eligible_days"] = balance.eligible_days / len(observations)
    return {"participant_coverage": participant_coverage(observations, alignment),
            "outcome_balance": balance, "feature_distributions": distributions.reset_index(),
            "feature_missingness": pd.DataFrame(missing_rows)}


def calibration_tables(predictions):
    """Fixed probability bins are descriptive; do not fit or select calibration."""
    validate_paired_predictions(predictions)
    bins, summaries = [], []
    for name in FEATURE_SETS:
        group = predictions.loc[predictions.input_set.eq(name)].copy()
        group["bin"] = np.minimum(np.floor(group.probability * 10).astype(int), 9)
        for index in range(10):
            subset = group.loc[group.bin.eq(index)]
            bins.append({"input_set": name, "bin": index, "lower_bound": index / 10,
                         "upper_bound": (index + 1) / 10, "days": len(subset),
                         "participants": subset.id.nunique(), "mean_probability": subset.probability.mean(),
                         "observed_luteal_fraction": subset.luteal_status.mean()})
        summaries.append({"input_set": name, **daily_metrics(group),
                          "mean_probability": group.probability.mean(),
                          "brier_score": ((group.probability - group.luteal_status) ** 2).mean()})
    return {"calibration_bins": pd.DataFrame(bins), "probability_summary": pd.DataFrame(summaries)}


def matched_timing_table(reference_events):
    """Signed error is supplementary; absolute error remains the protocol measure."""
    matched = reference_events.loc[reference_events.matched].copy()
    matched["signed_error_days"] = matched.alert_day_in_study - matched.reference_day_in_study
    if not np.allclose(matched.signed_error_days.abs(), matched.timing_error_days):
        raise ValueError("Saved absolute timing errors disagree with matched event days")
    return matched
