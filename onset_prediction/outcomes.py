#### Construct the three-calendar-day outcome from the full supplied phase-label timeline.
#### Missing dates and uncertain transitions remain unknown, never assumed negative.
#### Outputs include per-day start statuses, future outcomes and links to distinct starts.

import numpy as np
import pandas as pd

from onset_prediction.protocol import HORIZON, KEY, TARGET
from preprocessing import PHASES


def validate_keys(data):
    #### Reject duplicate, fractional, missing or non-2022 observation keys before joins.
    for name in KEY:
        values = pd.to_numeric(data[name], errors="coerce")
        if not (np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"Invalid observation key: {name}")
    if data.empty or data.duplicated(KEY).any() or not data.study_interval.eq(2022).all():
        raise ValueError("Expected non-empty, unique 2022 observation keys")


def start_statuses(alignment):
    #### Identify starts by calendar joins, including known non-starts after missing labels.
    validate_keys(alignment)
    data = alignment[KEY + ["phase"]].sort_values(KEY).reset_index(drop=True).copy()
    if not data.phase.dropna().isin(PHASES).all():
        raise ValueError("Unrecognised phase label in prepared alignment")

    previous = data.rename(columns={"phase": "previous_phase"}).copy()
    previous["day_in_study"] += 1
    data = data.merge(previous, on=KEY, how="left", validate="one_to_one")
    current_luteal = data.phase.eq("luteal")
    previous_known = data.previous_phase.notna()
    known_zero = data.phase.notna() & (~current_luteal | data.previous_phase.eq("luteal"))
    exact = current_luteal & previous_known & data.previous_phase.ne("luteal")
    data["start_status"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data.loc[known_zero, "start_status"] = 0
    data.loc[exact, "start_status"] = 1
    return data


def future_outcomes(statuses):
    #### Look up exactly t+1, t+2 and t+3 without extending the window over missing days.
    data = statuses.copy()
    future_columns, label_columns, record_columns = [], [], []
    for offset in range(1, HORIZON + 1):
        future = statuses[KEY + ["start_status", "phase"]].copy()
        future["day_in_study"] -= offset
        status, label, present = f"start_t_plus_{offset}", f"phase_t_plus_{offset}", f"record_t_plus_{offset}"
        future = future.rename(columns={"start_status": status, "phase": label}).assign(**{present: True})
        data = data.merge(future, on=KEY, how="left", validate="one_to_one")
        future_columns.append(status)
        label_columns.append(label)
        record_columns.append(present)

    positive = data[future_columns].eq(1).any(axis=1)
    negative = data[future_columns].eq(0).fillna(False).all(axis=1)
    data[TARGET] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data.loc[positive, TARGET] = 1
    data.loc[negative, TARGET] = 0
    data["future_records_missing"] = data[record_columns].isna().sum(axis=1)
    data["future_labels_missing"] = data[label_columns].isna().sum(axis=1)
    data["future_start_statuses_unknown"] = data[future_columns].isna().sum(axis=1)
    data["outcome_reason"] = np.select(
        [positive, negative, data.future_records_missing.gt(0), data.future_labels_missing.gt(0)],
        ["known_start", "known_no_start", "missing_future_record", "missing_future_label"],
        default="uncertain_transition")
    return data


def event_links(outcomes):
    #### Keep every prediction-to-start link so repeated positive days are not extra events.
    parts = []
    for offset in range(1, HORIZON + 1):
        rows = outcomes.loc[outcomes[f"start_t_plus_{offset}"].eq(1).fillna(False), KEY].copy()
        rows["reference_day"] = rows.day_in_study + offset
        rows["lead_days"] = offset
        parts.append(rows)
    return pd.concat(parts, ignore_index=True).sort_values(KEY + ["reference_day"]).reset_index(drop=True)
