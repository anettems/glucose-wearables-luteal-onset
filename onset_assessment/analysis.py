#### Align quality-approved 2022 measurements to the exact starts used by the onset task.
#### Keep actual calendar gaps and missing signals. Centre each event on its earlier days
#### -7 to -4, requiring three observations. Compare days -3 to -1 and 0 to +2 within
#### the same event, requiring two observations in each window. Average events within
#### participants before group summaries and resample participants for uncertainty.
#### These descriptive contrasts are not model predictors or estimates of prediction accuracy.

import numpy as np
import pandas as pd

from onset_prediction import protocol
from onset_prediction.outcomes import start_statuses, validate_keys


EVENT_KEY = ["id", "study_interval", "reference_day"]
MEASUREMENTS = {
    "glucose_median_mmol_l": ("Glucose median", "mmol/L"),
    "glucose_cv_percent": ("Glucose variability", "percentage points"),
    "skin_temperature_c": ("Skin temperature", "°C"),
    "sleeping_hr_bpm": ("Sleeping heart rate", "beats/min"),
    "rmssd_ms": ("HRV (RMSSD)", "ms"),
}
OFFSETS = np.arange(-7, 8)
SEED = 20261020
RESAMPLES = 2000


def select_events(audit, observations, links):
    #### Restrict to distinct starts represented by at least one eligible positive prediction.
    validate_keys(audit)
    validate_keys(observations)
    actual = start_statuses(audit)
    if not np.array_equal(actual.start_status.fillna(-1).to_numpy(),
                          audit.sort_values(protocol.KEY).start_status.fillna(-1).to_numpy()):
        raise ValueError("Saved start statuses differ from the consecutive-day label rule")
    linked = links.merge(observations[protocol.KEY + [protocol.TARGET]], on=protocol.KEY,
                         how="inner", validate="many_to_one")
    if not linked[protocol.TARGET].eq(1).all():
        raise ValueError("An onset link is attached to a non-positive prediction day")
    if not linked.reference_day.sub(linked.day_in_study).eq(linked.lead_days).all():
        raise ValueError("Onset links have incorrect calendar spacing")
    if not linked.lead_days.isin([1, 2, 3]).all():
        raise ValueError("Onset link is outside the three-day horizon")
    events = linked[EVENT_KEY].drop_duplicates().sort_values(EVENT_KEY).reset_index(drop=True)
    exact = audit.loc[audit.start_status.eq(1), protocol.KEY].rename(
        columns={"day_in_study": "reference_day"})
    if len(events.merge(exact, on=EVENT_KEY, validate="one_to_one")) != len(events):
        raise ValueError("A selected event is not an exact reference start")
    return events


def align_measurements(audit, events):
    #### Build a complete event/calendar grid and join only already quality-approved measurements.
    validate_keys(audit)
    if events.empty or events.duplicated(EVENT_KEY).any():
        raise ValueError("Expected distinct non-empty reference events")
    grid = events.merge(pd.DataFrame({"relative_day": OFFSETS}), how="cross")
    grid["day_in_study"] = grid.reference_day + grid.relative_day
    data = audit.copy()
    data["rmssd_ms"] = np.exp(data.log_median_rmssd_ms)
    columns = protocol.KEY + list(MEASUREMENTS) + ["supervised_eligible", "prior_day_source_day",
              "prediction_cutoff_seconds", "main_sleep_end_seconds"]
    aligned = grid.merge(data[columns], on=protocol.KEY, how="left", validate="many_to_one",
                         indicator="record_match")
    aligned["record_available"] = aligned.record_match.eq("both")
    aligned["supervised_eligible"] = aligned.supervised_eligible.eq(True)
    observed_glucose = aligned.glucose_median_mmol_l.notna()
    if not aligned.loc[observed_glucose, "prior_day_source_day"].eq(
            aligned.loc[observed_glucose, "day_in_study"] - 1).all():
        raise ValueError("Glucose is not aligned to the preceding calendar day")
    wearable = aligned[["skin_temperature_c", "sleeping_hr_bpm", "rmssd_ms"]].notna().any(axis=1)
    if not aligned.loc[wearable, "main_sleep_end_seconds"].le(
            aligned.loc[wearable, "prediction_cutoff_seconds"]).all():
        raise ValueError("Wearable measurements are unavailable by the cutoff")
    if np.isinf(aligned[list(MEASUREMENTS)].to_numpy(dtype=float)).any():
        raise ValueError("Infinite measurement value")
    return aligned.drop(columns="record_match")


def centre_measurements(aligned):
    #### Subtract each event's early observed mean; never fill missing days or measurements.
    identifiers = EVENT_KEY + ["relative_day", "day_in_study", "record_available", "supervised_eligible"]
    long = aligned.melt(id_vars=identifiers, value_vars=list(MEASUREMENTS),
                        var_name="measurement", value_name="value")
    keys = EVENT_KEY + ["measurement"]
    baseline = long.loc[long.relative_day.between(-7, -4)].groupby(keys).value.agg(
        baseline="mean", baseline_days="count").reset_index()
    baseline.loc[baseline.baseline_days.lt(3), "baseline"] = np.nan
    long = long.merge(baseline, on=keys, how="left", validate="many_to_one")
    long["change"] = long.value - long.baseline
    return long.sort_values(keys + ["relative_day"]).reset_index(drop=True)


def bootstrap_means(matrix):
    #### Resample whole participants, preserving all their event/day values and missing cells.
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if not len(matrix):
        raise ValueError("Cannot bootstrap an empty participant matrix")
    rng = np.random.default_rng(SEED + 1)
    draws = rng.integers(0, len(matrix), size=(RESAMPLES, len(matrix)))
    sampled = matrix[draws]
    counts = np.isfinite(sampled).sum(axis=1)
    totals = np.nansum(sampled, axis=1)
    means = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    records = []
    for column in means.T:
        finite = column[np.isfinite(column)]
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 2 else (np.nan, np.nan)
        records.append({"ci_lower": low, "ci_upper": high, "valid_resamples": len(finite)})
    return pd.DataFrame(records)


def trajectory_tables(long):
    #### Give every participant equal weight after averaging that person's available events.
    participants = long.groupby(["id", "measurement", "relative_day"], as_index=False).agg(
        change=("change", "mean"), raw_value=("value", "mean"), events=("change", "count"))
    rows = []
    for name in MEASUREMENTS:
        data = participants.loc[participants.measurement.eq(name)]
        matrix = data.pivot(index="id", columns="relative_day", values="change").reindex(columns=OFFSETS)
        intervals = bootstrap_means(matrix.to_numpy())
        for index, day in enumerate(OFFSETS):
            subset = long.loc[long.measurement.eq(name) & long.relative_day.eq(day)]
            values = matrix[day].dropna()
            rows.append({"measurement": name, "relative_day": day,
                         "mean_change": values.mean(), "participants": len(values),
                         "centred_events": int(subset.change.notna().sum()),
                         "observed_events": int(subset.value.notna().sum()),
                         "model_eligible_observed_events": int((subset.value.notna() & subset.supervised_eligible).sum()),
                         "total_events": len(subset), **intervals.iloc[index].to_dict()})
    return participants, pd.DataFrame(rows)


def direction_counts(values):
    #### Treat numerical round-off below 1e-12 native units as unchanged, not a real direction.
    values = np.asarray(values, dtype=float)
    unchanged = np.isclose(values, 0, atol=1e-12, rtol=0)
    return {"positive_participants": int(((values > 0) & ~unchanged).sum()),
            "negative_participants": int(((values < 0) & ~unchanged).sum()),
            "zero_participants": int(unchanged.sum())}


def window_contrasts(long):
    #### Compare early and post-onset changes using the same adequately observed events.
    keys = EVENT_KEY + ["measurement"]
    windows = []
    for name, start, end in (("pre", -3, -1), ("post", 0, 2)):
        summary = long.loc[long.relative_day.between(start, end)].groupby(keys).agg(
            **{f"{name}_change": ("change", "mean"), f"{name}_days": ("change", "count")}).reset_index()
        windows.append(summary)
    events = windows[0].merge(windows[1], on=keys, validate="one_to_one")
    events["paired_eligible"] = events.pre_days.ge(2) & events.post_days.ge(2)
    paired = events.loc[events.paired_eligible].copy()
    paired["post_minus_pre"] = paired.post_change - paired.pre_change
    participants = paired.groupby(["id", "measurement"], as_index=False).agg(
        pre_change=("pre_change", "mean"), post_change=("post_change", "mean"),
        post_minus_pre=("post_minus_pre", "mean"), events=("reference_day", "count"))
    summaries = []
    for name in MEASUREMENTS:
        data = participants.loc[participants.measurement.eq(name)]
        for contrast in ("pre_change", "post_change", "post_minus_pre"):
            values = data[contrast].to_numpy()
            interval = bootstrap_means(values).iloc[0].to_dict() if len(values) else {
                "ci_lower": np.nan, "ci_upper": np.nan, "valid_resamples": 0}
            summaries.append({"measurement": name, "contrast": contrast,
                              "mean_change": np.mean(values) if len(values) else np.nan,
                              "participants": len(values), "events": int(data.events.sum()),
                              **direction_counts(values), **interval})
    return events, participants, pd.DataFrame(summaries)
