"""Quality-approved history, strict calendar windows and shared cohort audit."""

import numpy as np
import pandas as pd

import preprocessing as prep
from model_inputs import FEATURE_SETS, GLUCOSE_FEATURES, WEARABLE_FEATURES, WEARABLE_FLAGS
from model_inputs import glucose_features, hrv_features, validate_features
from onset_alerts import reference_evaluability, validate_event_keys
from trend_analysis import protocol


FEATURES = FEATURE_SETS["combined"]


def load_quality_history(processed, configuration):
    """Keep valid individual signals even when another modality or the label is absent."""
    alignment = pd.read_csv(processed / "prediction_alignment_14_2022.csv")
    validate_event_keys(alignment)
    history = alignment[prep.KEY + ["sleeping_hr_bpm"]].copy()
    glucose = pd.read_csv(processed / "glucose_2022_audited.csv")
    history = prep.join_daily(history, glucose_features(glucose))
    intervals = pd.read_csv(processed / "hrv_sleep_alignment_2022.csv")
    history = prep.join_daily(history, hrv_features(intervals))
    sleep = prep.review_sleep_sessions(pd.read_csv(processed / "sleep_2022_audited.csv"),
                                      configuration["quality_review_settings"])
    temperature = prep.review_temperature_sessions(pd.read_csv(processed / "computed_temperature_2022_audited.csv"), sleep)
    selected = temperature.loc[temperature.review_selected, prep.KEY + ["nightly_temperature"]]
    history = prep.join_daily(history, selected.rename(columns={"nightly_temperature": "skin_temperature_c"}))
    for feature in GLUCOSE_FEATURES:
        history[feature] = history[feature].where(alignment.aligned_glucose)
    for feature, flag in zip(WEARABLE_FEATURES, WEARABLE_FLAGS):
        history[feature] = history[feature].where(alignment[flag])
    validate_history(history)
    return history.sort_values(prep.KEY).reset_index(drop=True)


def validate_history(history):
    validate_event_keys(history)
    if np.isinf(history[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("History contains infinite feature values")


def window_statistics(days, values, day):
    reference = (days >= day - protocol.REFERENCE_DAYS) & (days < day) & np.isfinite(values)
    recent = (days >= day - protocol.TREND_DAYS) & (days < day) & np.isfinite(values)
    count, recent_count = int(reference.sum()), int(recent.sum())
    baseline = float(np.median(values[reference])) if count >= protocol.REFERENCE_MINIMUM else np.nan
    slope = np.nan
    if recent_count >= protocol.TREND_MINIMUM:
        x, y = days[recent].astype(float), values[recent]
        centred = x - x.mean()
        slope = float(np.dot(centred, y - y.mean()) / np.dot(centred, centred))
    return {"reference_count": count, "trend_count": recent_count, "reference": baseline, "slope": slope,
            "last_reference_day": float(days[reference].max()) if count else np.nan,
            "last_trend_day": float(days[recent].max()) if recent_count else np.nan}


def add_personal_features(history):
    """Exclude the current aligned value and every future day from both history windows."""
    validate_history(history)
    parts = []
    for _, group in history.sort_values(prep.KEY).groupby(prep.PARTICIPANT_KEY, sort=True):
        group = group.copy()
        days = group.day_in_study.to_numpy(dtype=int)
        for feature in FEATURES:
            values = group[feature].to_numpy(dtype=float)
            statistics = pd.DataFrame([window_statistics(days, values, day) for day in days], index=group.index)
            for column in statistics:
                group[f"{feature}__{column}"] = statistics[column]
            group[f"{feature}__deviation"] = group[feature] - statistics.reference
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def select_history_cohort(observations, enriched):
    """Use one cohort for all input sets and all three feature representations."""
    for feature in FEATURES:
        actual = enriched[prep.KEY + [feature]]
        joined = observations[prep.KEY + [feature]].merge(actual, on=prep.KEY, how="left", validate="one_to_one",
                                                        suffixes=("", "_history"))
        if not np.allclose(joined[feature], joined[f"{feature}_history"], equal_nan=True):
            raise ValueError(f"History reconstruction disagrees with primary observations: {feature}")
    derived = [column for column in enriched if "__" in column]
    audit = observations.merge(enriched[prep.KEY + derived], on=prep.KEY, how="left", validate="one_to_one")
    ready = {}
    for feature in FEATURES:
        ready[feature] = audit[[feature, f"{feature}__deviation", f"{feature}__slope"]].notna().all(axis=1)
        audit[f"{feature}__history_ready"] = ready[feature]
    audit["glucose_history_ready"] = np.logical_and.reduce([ready[f] for f in GLUCOSE_FEATURES])
    audit["wearable_histories_ready"] = sum(ready[f].astype(int) for f in WEARABLE_FEATURES)
    audit["history_eligible"] = audit.glucose_history_ready & audit.wearable_histories_ready.ge(2)
    cohort = audit.loc[audit.history_eligible].copy().reset_index(drop=True)
    validate_features(cohort)
    return audit, cohort


def coverage_record(data, references, stage):
    starts = references.loc[references.id.isin(data.id)]
    return {"stage": stage, "participants": data.id.nunique(), "eligible_days": len(data),
            "luteal_days": int(data.luteal_status.sum()), "luteal_fraction": data.luteal_status.mean(),
            "reference_starts": len(starts),
            "evaluable_starts": int(reference_evaluability(data, starts).evaluable.sum())}


def audit_tables(observations, audit, cohort, references):
    availability = []
    for feature in FEATURES:
        availability.append({"feature": feature, "primary_days": len(audit),
                             "current_observed": int(audit[feature].notna().sum()),
                             "reference_available": int(audit[f"{feature}__reference"].notna().sum()),
                             "slope_available": int(audit[f"{feature}__slope"].notna().sum()),
                             "current_and_history_ready": int(audit[f"{feature}__history_ready"].sum())})
    participants = audit.groupby("id").agg(primary_days=("id", "size"),
                                          retained_days=("history_eligible", "sum"))
    coverage = [coverage_record(observations, references, "primary"),
                coverage_record(cohort, references, "shared_history_cohort")]
    rows = []
    primary_flags = reference_evaluability(observations, references)
    history_flags = reference_evaluability(cohort, references)
    for index, start in references.reset_index(drop=True).iterrows():
        same = cohort.loc[cohort.id.eq(start.id)]
        count = same.day_in_study.sub(start.day_in_study).abs().le(3).sum()
        rows.append({**start.to_dict(), "primary_evaluable": bool(primary_flags.iloc[index].evaluable),
                     "history_evaluable": bool(history_flags.iloc[index].evaluable), "history_days_in_window": int(count)})
    return {"coverage_summary": pd.DataFrame(coverage), "feature_availability": pd.DataFrame(availability),
            "participant_coverage": participants.reset_index(), "reference_availability": pd.DataFrame(rows)}
