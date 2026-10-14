"""Audit of the mcPHASES 2022 exports.

Run from the thesis workspace: python code/preprocessing.py
Requires pandas and NumPy. Confirmed glucose units are standardised to mmol/L;
no model features or imputation are produced. Supplied phase labels are not
independently validated outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_INTERVAL = 2022
KEY = ["id", "study_interval", "day_in_study"]
PARTICIPANT_KEY = KEY[:2]
PHASES = {"menstrual", "follicular", "fertility", "ovulation", "luteal"}
SECONDS_PER_DAY = 86400
CGM_INTERVAL_SECONDS = 300
PRIMARY_CUTOFF_HOUR = 12
UNRESOLVED_GLUCOSE_IDS = ()
MG_DL_GLUCOSE_IDS = (6, 11)
MG_DL_TO_MMOL_L = 0.0555
SOURCE_REVIEW_DATE = "2026-10-12"
EVIDENCE_SOURCES = {
    "glucose_conversion": {
        "url": "https://www.niddk.nih.gov/-/media/Files/Strategic-Plans/Diabetes-in-America-3rd-Edition/DIA_Conversions.pdf",
        "finding": "The glucose conversion table specifies mg/dL multiplied by 0.0555 to obtain mmol/L.",
    },
    "dataset": {
        "url": "https://physionet.org/content/mcphases/1.0.0/",
        "finding": "Device exports are largely uncleaned; dates use relative study days. "
                   "The public description supplies no participant-specific glucose-unit corrections.",
    },
    "glucose_study": {
        "url": "https://www.nature.com/articles/s41746-023-00884-x",
        "finding": "The original study reports glucose in mmol/L, but does not establish "
                   "the units of individual participant records in this release.",
    },
    "fitbit_hrv": {
        "url": "https://dev.fitbit.com/build/reference/web-api/intraday/get-hrv-intraday-by-date/",
        "finding": "Coverage measures interbeat-interval completeness; no upper bound is "
                   "specified. RMSSD is in milliseconds. HRV is processed after sleep and "
                   "device synchronisation, so interval completion alone does not prove availability.",
    },
    "fitbit_sleep": {
        "url": "https://dev.fitbit.com/build/reference/web-api/sleep/get-sleep-log-by-date/",
        "finding": "The main-sleep flag distinguishes the principal sleep log from other "
                   "sessions. Original session timing and this flag are retained for review.",
    },
}
# These checks identify structural problems, not final physiological cutoffs.
TABLE_SPECS = {
    "hormones_and_selfreport": {
        "values": ["lh", "estrogen", "pdg"], "positive": [],
    },
    "glucose": {
        "values": ["glucose_value"], "positive": ["glucose_value"],
        "timestamp": "timestamp",
    },
    "sleep_score": {
        "values": ["resting_heart_rate"], "positive": ["resting_heart_rate"],
        "timestamp": "timestamp",
    },
    "resting_heart_rate": {
        "values": ["value", "error"], "positive": ["value"],
    },
    "heart_rate_variability_details": {
        "values": ["rmssd", "coverage", "low_frequency", "high_frequency"],
        "positive": ["rmssd", "coverage"], "timestamp": "timestamp",
    },
    "computed_temperature": {
        "values": ["nightly_temperature", "temperature_samples"],
        "positive": ["temperature_samples"], "session": True,
    },
    "sleep": {
        "values": ["duration", "minutesasleep", "timeinbed"],
        "positive": ["duration", "timeinbed"], "session": True,
    },
}


def parse_arguments() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=workspace / "code/data/raw")
    parser.add_argument(
        "--processed-dir", type=Path,
        default=workspace / "code/data/processed/audit_2022",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=workspace / "results/preprocessing_2022",
    )
    parser.add_argument("--eligibility-review", action="store_true",
                        help="Also generate quality-rule evidence, effects and a decision log")
    parser.add_argument("--temporal-review", action="store_true",
                        help="Also align morning inputs and require three usable calendar days")
    parser.add_argument("--min-glucose-coverage", type=float, default=0.70)
    parser.add_argument("--max-glucose-gap-hours", type=float, default=4.0)
    parser.add_argument("--min-hrv-coverage", type=float, default=0.80)
    parser.add_argument("--min-hrv-windows", type=int, default=6)
    return parser.parse_args()


def validate_output_paths(raw_dir: Path, destinations: list[Path]) -> None:
    """Refuse raw-data writes, overlapping destinations and existing outputs."""
    for destination in destinations:
        if destination == raw_dir or raw_dir in destination.parents:
            raise ValueError("Output directories must be outside the raw-data directory")
        if destination in raw_dir.parents:
            raise ValueError("Output directories must not contain the raw-data directory")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {destination}")
    first, second = destinations
    if first == second or first in second.parents or second in first.parents:
        raise ValueError("Processed and report directories must be separate")


def validate_integer_column(data: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(data[column], errors="coerce")
    valid = values.notna() & np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)
    if not valid.all():
        raise ValueError(f"Invalid positive integer key in {column}: {int((~valid).sum())} rows")
    data[column] = values.astype("int64")


def required_columns(spec: dict) -> set[str]:
    columns = set(KEY + spec["values"])
    if spec.get("session"):
        columns.remove("day_in_study")
        columns.update([
            "sleep_start_day_in_study", "sleep_end_day_in_study",
            "sleep_start_timestamp", "sleep_end_timestamp",
        ])
    if "timestamp" in spec:
        columns.add(spec["timestamp"])
    return columns


def load_2022(path: Path, spec: dict) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(path, low_memory=False)
    required = required_columns(spec)
    if path.stem == "hormones_and_selfreport":
        required.add("phase")
    if path.stem == "computed_temperature":
        required.add("type")
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    intervals = pd.to_numeric(data["study_interval"], errors="coerce")
    if intervals.isna().any() or not np.isfinite(intervals).all():
        raise ValueError(f"{path.name}: invalid study_interval values")
    selected = data.loc[intervals.eq(STUDY_INTERVAL)].copy()
    excluded_rows = len(data) - len(selected)
    day_columns = ["day_in_study"]
    if spec.get("session"):
        day_columns = ["sleep_start_day_in_study", "sleep_end_day_in_study"]
    for column in PARTICIPANT_KEY + day_columns:
        validate_integer_column(selected, column)
    if spec.get("session"):
        selected["day_in_study"] = selected["sleep_end_day_in_study"]
    return selected, excluded_rows


def clock_seconds(values: pd.Series) -> pd.Series:
    # Reject dates, durations and timezone strings: the exports contain clocks.
    pattern = r"\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    text = values.astype("string")
    valid_format = text.str.fullmatch(pattern, na=False)
    seconds = pd.to_timedelta(text.where(valid_format), errors="coerce").dt.total_seconds()
    parts = text.str.extract(r"^(\d{2}):(\d{2}):(\d{2})")
    components = parts.apply(pd.to_numeric, errors="coerce")
    valid_clock = components[0].lt(24) & components[1].lt(60) & components[2].lt(60)
    return seconds.where(valid_clock & seconds.ge(0) & seconds.lt(SECONDS_PER_DAY))


def add_timing(data: pd.DataFrame, spec: dict) -> pd.DataFrame:
    data = data.copy()
    data["audit_invalid_time"] = False
    if "timestamp" in spec:
        seconds = clock_seconds(data[spec["timestamp"]])
        data["audit_end_seconds"] = data["day_in_study"] * SECONDS_PER_DAY + seconds
        if "rmssd" in spec["values"]:
            data["audit_end_seconds"] += CGM_INTERVAL_SECONDS
        data["audit_invalid_time"] = seconds.isna()
    if spec.get("session"):
        start = clock_seconds(data["sleep_start_timestamp"])
        end = clock_seconds(data["sleep_end_timestamp"])
        data["audit_start_seconds"] = data["sleep_start_day_in_study"] * SECONDS_PER_DAY + start
        data["audit_end_seconds"] = data["sleep_end_day_in_study"] * SECONDS_PER_DAY + end
        data["audit_invalid_time"] = start.isna() | end.isna()
        data["audit_invalid_time"] |= data["audit_end_seconds"].le(data["audit_start_seconds"])
    return data


def measurement_key(spec: dict) -> list[str]:
    if spec.get("session"):
        columns = PARTICIPANT_KEY + [
            "sleep_start_day_in_study", "sleep_start_timestamp",
            "sleep_end_day_in_study", "sleep_end_timestamp",
        ]
        if "nightly_temperature" in spec["values"]:
            columns.append("type")
        return columns
    return KEY + ([spec["timestamp"]] if "timestamp" in spec else [])


def add_numeric_checks(data: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[dict]]:
    data = data.copy()
    data["audit_invalid_value"] = False
    checks = []
    for column in spec["values"]:
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & (values.isna() | ~np.isfinite(values))
        if column in spec["positive"]:
            invalid |= values.le(0)
        else:
            invalid |= values.lt(0)
        # Temperature is a descriptive field; no clinical limits are imposed.
        if column == "nightly_temperature":
            invalid = data[column].notna() & (values.isna() | ~np.isfinite(values))
        data["audit_invalid_value"] |= invalid
        data[f"audit_invalid_{column}"] = invalid
        checks.append({"column": column, "missing_rows": int(data[column].isna().sum()),
                       "invalid_rows": int(invalid.sum())})
    return data, checks


def audit_table(raw_dir: Path, name: str, spec: dict) -> tuple[pd.DataFrame, dict, list[dict]]:
    data, excluded_rows = load_2022(raw_dir / f"{name}.csv", spec)
    input_rows = len(data)
    data = data.drop_duplicates().copy()
    duplicates_removed = input_rows - len(data)
    data["audit_conflicting_key"] = data.duplicated(measurement_key(spec), keep=False)
    data = add_timing(data, spec)
    data, checks = add_numeric_checks(data, spec)
    if name == "glucose":
        data = standardise_glucose_units(data)
    data["audit_structurally_usable"] = ~data[[
        "audit_conflicting_key", "audit_invalid_time", "audit_invalid_value",
    ]].any(axis=1)
    summary = {
        "table": name, "rows_2022_before_deduplication": input_rows,
        "excluded_other_interval_rows": excluded_rows,
        "exact_duplicates_removed": duplicates_removed, "retained_rows": len(data),
        "participants": int(data["id"].nunique()),
        "conflicting_key_rows": int(data["audit_conflicting_key"].sum()),
        "invalid_time_rows": int(data["audit_invalid_time"].sum()),
        "invalid_value_rows": int(data["audit_invalid_value"].sum()),
    }
    if "coverage" in data:
        summary["hrv_coverage_above_one_rows"] = int(pd.to_numeric(data["coverage"], errors="coerce").gt(1).sum())
    return data.sort_values(measurement_key(spec)).reset_index(drop=True), summary, checks


def glucose_unit_policy() -> dict:
    return {"mg_dl_participants": list(MG_DL_GLUCOSE_IDS), "target_unit": "mmol/L",
            "mg_dl_to_mmol_l_multiplier": MG_DL_TO_MMOL_L,
            "confirmation_source": "User confirmed mg/dL for participants 6 and 11 in this project conversation",
            "confirmed_on": SOURCE_REVIEW_DATE,
            "provider_confirmation_independently_verified": False,
            "other_participants": "mmol/L provisionally follows the export README",
            "source_values_preserved_in": "glucose_source_value"}


def standardise_glucose_units(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the confirmed participant mapping once, after checking source values."""
    if "glucose_source_value" in data:
        raise ValueError("Glucose units have already been standardised; read the raw export")
    result = data.copy()
    converted = result["id"].isin(MG_DL_GLUCOSE_IDS) & result["study_interval"].eq(STUDY_INTERVAL)
    result["glucose_source_value"] = result["glucose_value"]
    result["glucose_source_unit"] = np.where(converted, "mg/dL", "mmol/L")
    result["glucose_unit"] = "mmol/L"
    result["glucose_unit_converted"] = converted
    result["glucose_conversion_factor"] = np.where(converted, MG_DL_TO_MMOL_L, 1.0)
    result["glucose_value"] = pd.to_numeric(result["glucose_source_value"], errors="coerce")
    result["glucose_value"] *= result["glucose_conversion_factor"]
    return result


def create_daily_grid(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    observed = pd.concat([data[KEY] for data in tables.values()], ignore_index=True)
    spans = observed.groupby(PARTICIPANT_KEY)["day_in_study"].agg(["min", "max"])
    grids = [pd.DataFrame({"id": participant, "study_interval": interval,
                          "day_in_study": range(int(span["min"]), int(span["max"]) + 1)})
             for (participant, interval), span in spans.iterrows()]
    if not grids:
        raise ValueError("No 2022 records were found in the audited tables")
    return pd.concat(grids, ignore_index=True).sort_values(KEY).reset_index(drop=True)


def join_daily(grid: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    result = grid.merge(daily, on=KEY, how="left", validate="one_to_one")
    if len(result) != len(grid) or result.duplicated(KEY).any():
        raise AssertionError("Daily join changed observation keys")
    return result


def add_labels(grid: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["phase"] = labels["phase"].astype("string").str.strip().str.lower().replace("", pd.NA)
    labels["audit_unknown_phase"] = labels["phase"].notna() & ~labels["phase"].isin(PHASES)
    # Conflicting daily records remain in the audited source, never resolved by averaging.
    reliable = labels.loc[~labels["audit_conflicting_key"] & ~labels["audit_unknown_phase"]]
    daily = join_daily(grid, reliable[KEY + ["phase"]]).sort_values(KEY).reset_index(drop=True)
    daily["label_record_present"] = pd.MultiIndex.from_frame(daily[KEY]).isin(
        pd.MultiIndex.from_frame(labels[KEY]))
    previous = daily.groupby(PARTICIPANT_KEY)["phase"].shift()
    known = daily["phase"].notna() & previous.notna()
    daily["luteal_status"] = daily["phase"].eq("luteal").astype("Int64").where(daily["phase"].notna())
    daily["exact_luteal_onset"] = (daily["phase"].eq("luteal") & previous.ne("luteal")).astype("Int64").where(known)
    daily["uncertain_luteal_entry"] = daily["phase"].eq("luteal").fillna(False) & previous.isna()
    return daily


def daily_record_counts(data: pd.DataFrame, name: str) -> pd.DataFrame:
    primary = TABLE_SPECS[name]["values"][0]
    available = data["audit_structurally_usable"] & data[primary].notna()
    counts = data.assign(_available=available).groupby(KEY).agg(
        records=("id", "size"), available_records=("_available", "sum"),
        conflicting_rows=("audit_conflicting_key", "sum"),
        invalid_time_rows=("audit_invalid_time", "sum"),
        invalid_value_rows=("audit_invalid_value", "sum"),
    )
    return counts.add_prefix(f"{name}_").reset_index()


def glucose_day_coverage(seconds: np.ndarray) -> dict:
    """Describe coverage using nominal forward five-minute intervals, clipped at midnight."""
    seconds = np.sort(np.unique(seconds))
    if not len(seconds):
        return {"glucose_nominal_observed_seconds": 0.0,
                "glucose_nominal_coverage_fraction": 0.0,
                "glucose_longest_uncovered_seconds": float(SECONDS_PER_DAY)}
    ends = np.minimum(seconds + CGM_INTERVAL_SECONDS, SECONDS_PER_DAY)
    previous_ends = np.r_[0.0, np.maximum.accumulate(ends)[:-1]]
    observed = np.maximum(0, ends - np.maximum(seconds, previous_ends)).sum()
    gaps = np.r_[np.maximum(0, seconds - previous_ends), SECONDS_PER_DAY - ends[-1]]
    return {"glucose_nominal_observed_seconds": float(observed),
            "glucose_nominal_coverage_fraction": float(observed / SECONDS_PER_DAY),
            "glucose_longest_uncovered_seconds": float(gaps.max())}


def glucose_coverage(data: pd.DataFrame) -> pd.DataFrame:
    valid = data.loc[data["audit_structurally_usable"] & data["glucose_value"].notna()]
    records = []
    for key, group in valid.groupby(KEY):
        seconds = group["audit_end_seconds"].to_numpy() % SECONDS_PER_DAY
        records.append(dict(zip(KEY, key)) | glucose_day_diagnostics(seconds))
    columns = KEY + list(glucose_day_diagnostics(np.array([])))
    return pd.DataFrame(records, columns=columns)


def glucose_day_diagnostics(seconds: np.ndarray) -> dict:
    seconds = np.sort(np.unique(seconds))
    result = glucose_day_coverage(seconds)
    gaps = np.diff(seconds)
    uncovered = np.maximum(0, gaps - CGM_INTERVAL_SECONDS)
    boundary = float(seconds[0] + max(0, SECONDS_PER_DAY - seconds[-1] - CGM_INTERVAL_SECONDS)) if len(seconds) else float(SECONDS_PER_DAY)
    return result | {
        "glucose_unique_samples": len(seconds),
        "glucose_occupied_5min_bins": len(np.unique(seconds // CGM_INTERVAL_SECONDS)),
        "glucose_first_sample_seconds": float(seconds[0]) if len(seconds) else np.nan,
        "glucose_last_sample_seconds": float(seconds[-1]) if len(seconds) else np.nan,
        "glucose_median_interval_seconds": float(np.median(gaps)) if len(gaps) else np.nan,
        "glucose_max_interval_seconds": float(gaps.max()) if len(gaps) else np.nan,
        "glucose_intervals_over_30min": int((gaps > 1800).sum()),
        "glucose_longest_interior_uncovered_seconds": float(uncovered.max()) if len(gaps) else 0.0,
        "glucose_boundary_uncovered_seconds": boundary,
    }


def eligibility_settings(args: argparse.Namespace) -> dict:
    settings = {
        "min_glucose_coverage": getattr(args, "min_glucose_coverage", 0.70),
        "max_glucose_gap_hours": getattr(args, "max_glucose_gap_hours", 4.0),
        "min_hrv_coverage": getattr(args, "min_hrv_coverage", 0.80),
        "min_hrv_windows": getattr(args, "min_hrv_windows", 6),
        "unresolved_glucose_ids": list(UNRESOLVED_GLUCOSE_IDS),
        "sleep_duration_tolerance_seconds": 60,
    }
    for name in ["min_glucose_coverage", "min_hrv_coverage"]:
        if not np.isfinite(settings[name]) or not 0 <= settings[name] <= 1:
            raise ValueError(f"{name} must be finite and between zero and one")
    if not np.isfinite(settings["max_glucose_gap_hours"]) or not 0 <= settings["max_glucose_gap_hours"] <= 24:
        raise ValueError("max_glucose_gap_hours must be finite and between zero and 24")
    windows = settings["min_hrv_windows"]
    if not np.isfinite(windows) or windows < 1 or int(windows) != windows:
        raise ValueError("min_hrv_windows must be a positive integer")
    return settings


def primary_records(data: pd.DataFrame, column: str) -> pd.Series:
    return data["audit_structurally_usable"] & data[column].notna()


def hrv_quality_counts(data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    coverage = pd.to_numeric(data["coverage"], errors="coerce")
    usable = primary_records(data, "rmssd") & coverage.ge(settings["min_hrv_coverage"])
    quality = data.assign(_usable=usable, _bounded=usable & coverage.le(1),
                          _above_one=coverage.gt(1))
    return quality.groupby(KEY).agg(
        hrv_quality_windows=("_usable", "sum"),
        hrv_bounded_quality_windows=("_bounded", "sum"),
        hrv_coverage_above_one_windows=("_above_one", "sum"),
    ).reset_index()


def overlapping_session_flags(data: pd.DataFrame, candidates: pd.Series) -> pd.Series:
    """Flag both members of overlapping main sessions, including across end days."""
    flags = pd.Series(False, index=data.index)
    for _, group in data.loc[candidates].groupby(PARTICIPANT_KEY):
        active = {}
        for index, row in group.sort_values("audit_start_seconds").iterrows():
            active = {key: end for key, end in active.items() if end > row["audit_start_seconds"]}
            if active:
                flags.loc[list(active)] = True
                flags.loc[index] = True
            active[index] = row["audit_end_seconds"]
    return flags


def review_sleep_sessions(data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = data.copy()
    if "mainsleep" not in data:
        raise ValueError("Eligibility review requires the sleep mainsleep field")
    data["review_main_sleep"] = data["mainsleep"].astype("string").str.lower().eq("true").fillna(False)
    elapsed = data["audit_end_seconds"] - data["audit_start_seconds"]
    data["review_duration_discrepancy_seconds"] = (elapsed - pd.to_numeric(data["duration"], errors="coerce") / 1000).abs()
    duration_ok = data["review_duration_discrepancy_seconds"].le(settings["sleep_duration_tolerance_seconds"])
    candidate = primary_records(data, "duration") & data["review_main_sleep"] & duration_ok
    data["review_main_sessions_on_end_day"] = candidate.groupby([data[column] for column in KEY]).transform("sum")
    data["review_overlapping_main_sleep"] = overlapping_session_flags(data, candidate)
    data["review_selected"] = candidate & data["review_main_sessions_on_end_day"].eq(1)
    data["review_selected"] &= ~data["review_overlapping_main_sleep"]
    return data


def review_sleeping_heart_rate(data: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    """Use exact session ends; naps, ambiguous ends and unmatched scores are excluded."""
    keys = KEY + ["audit_end_seconds"]
    ends = sleep.groupby(keys).agg(
        matched_sleep_sessions=("id", "size"),
        matched_main_sessions=("review_main_sleep", "sum"),
        matched_selected_sessions=("review_selected", "sum"),
    ).reset_index()
    result = data.merge(ends, on=keys, how="left", validate="many_to_one")
    counts = ["matched_sleep_sessions", "matched_main_sessions", "matched_selected_sessions"]
    result[counts] = result[counts].fillna(0).astype(int)
    unique = result["matched_sleep_sessions"].eq(1)
    selected = result["matched_selected_sessions"].eq(1)
    result["review_selected"] = primary_records(result, "resting_heart_rate") & unique & selected
    result["review_match_status"] = np.select(
        [result["matched_sleep_sessions"].eq(0), ~unique,
         result["matched_main_sessions"].eq(0), ~selected,
         ~primary_records(result, "resting_heart_rate")],
        ["unmatched_end", "ambiguous_end", "non_main_sleep", "unselected_main_sleep", "invalid_or_missing_hr"],
        default="selected_main_sleep",
    )
    result["sleeping_hr_bpm"] = pd.to_numeric(result["resting_heart_rate"], errors="coerce")
    main = sleep.loc[sleep["review_selected"], KEY + ["audit_end_seconds"]].rename(
        columns={"audit_end_seconds": "selected_main_sleep_end_seconds"})
    result = join_daily_scores_to_main(result, main)
    return result


def join_daily_scores_to_main(scores: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Retain end-time discrepancies as diagnostics, never as matching tolerances."""
    result = scores.merge(main, on=KEY, how="left", validate="many_to_one")
    result["score_minus_main_end_seconds"] = (
        result["audit_end_seconds"] - result["selected_main_sleep_end_seconds"])
    return result


def sleeping_hr_daily(scores: pd.DataFrame) -> pd.DataFrame:
    selected = scores.loc[scores["review_selected"], KEY + ["sleeping_hr_bpm", "timestamp"]]
    return selected.rename(columns={"timestamp": "sleeping_hr_source_timestamp"}).assign(
        quality_sleeping_hr=True)


def review_temperature_sessions(data: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    session_key = PARTICIPANT_KEY + ["audit_start_seconds", "audit_end_seconds"]
    main = sleep.loc[sleep["review_selected"], session_key].assign(_main_match=True)
    data = data.merge(main, on=session_key, how="left", validate="many_to_one")
    data["review_matches_selected_main_sleep"] = data.pop("_main_match").fillna(False).astype(bool)
    samples = pd.to_numeric(data["temperature_samples"], errors="coerce")
    candidate = primary_records(data, "nightly_temperature") & samples.gt(0) & samples.mod(1).eq(0)
    candidate &= data["type"].astype("string").str.upper().eq("SKIN").fillna(False)
    candidate &= data["review_matches_selected_main_sleep"]
    count = candidate.groupby([data[column] for column in KEY]).transform("sum")
    data["review_selected"] = candidate & count.eq(1)
    return data


def session_diagnostics(data: pd.DataFrame, table: str) -> pd.DataFrame:
    columns = KEY + ["sleep_start_day_in_study", "sleep_start_timestamp",
                     "sleep_end_day_in_study", "sleep_end_timestamp",
                     "audit_start_seconds", "audit_end_seconds", "audit_invalid_time"]
    columns += [column for column in data if column.startswith("review_")]
    result = data[columns].copy()
    result.insert(0, "table", table)
    result["cross_midnight"] = result["sleep_end_day_in_study"].gt(result["sleep_start_day_in_study"])
    result["sessions_on_end_day"] = data.groupby(KEY)["id"].transform("size")
    result["end_clock_seconds"] = result["audit_end_seconds"] % SECONDS_PER_DAY
    return result


def phase_transition_review(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily[KEY + ["phase", "exact_luteal_onset", "uncertain_luteal_entry"]].copy()
    known_day = data["day_in_study"].where(data["phase"].notna())
    grouping = [data[column] for column in PARTICIPANT_KEY]
    data["previous_known_phase"] = data["phase"].groupby(grouping).ffill().groupby(grouping).shift()
    data["previous_known_day"] = known_day.groupby(grouping).ffill().groupby(grouping).shift()
    candidate = data["exact_luteal_onset"].eq(1).fillna(False) | data["uncertain_luteal_entry"]
    data = data.loc[candidate].copy()
    data["transition_status"] = "exact"
    uncertain = data["uncertain_luteal_entry"]
    previous = data["previous_known_phase"]
    data.loc[uncertain & previous.isna(), "transition_status"] = "left_censored"
    data.loc[uncertain & previous.eq("luteal").fillna(False), "transition_status"] = "possible_continuation"
    interval = uncertain & previous.notna() & previous.ne("luteal").fillna(False)
    data.loc[interval, "transition_status"] = "interval_censored"
    data["earliest_onset_day"] = data["day_in_study"].astype("Int64")
    data.loc[interval, "earliest_onset_day"] = (data.loc[interval, "previous_known_day"] + 1).astype("Int64")
    data.loc[uncertain & ~interval, "earliest_onset_day"] = pd.NA
    data["latest_onset_day"] = data["day_in_study"].astype("Int64")
    data.loc[data["transition_status"].eq("possible_continuation"), "latest_onset_day"] = pd.NA
    return data.reset_index(drop=True)


def selected_day_flag(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data.groupby(KEY)["review_selected"].any().rename(column).reset_index()


def temporal_settings() -> dict:
    return {"prediction_cutoff": f"{PRIMARY_CUTOFF_HOUR:02d}:00:00",
            "history_calendar_days": 3, "required_history_days": 3,
            "glucose_lag_days": 1, "sleeping_heart_rate_lag_days": 0,
            "heart_rate_source": "sleep_score.csv.resting_heart_rate",
            "heart_rate_unit": "bpm",
            "heart_rate_interpretation": "Fitbit-reported sleeping HR; aggregation unspecified",
            "sleep_score_timestamp_assumption": "Score timestamp is matched exactly to selected main-sleep end; not a publication time",
            "clock_assumption": "Exported clocks are assumed to be participant-local and mutually aligned; unverified",
            "availability_assumption": "Completed prior-day summaries and completed main-sleep measurements are assumed available by cutoff; sync times unverified",
            "synchronisation_verified": False, "timezone_verified": False}


def align_hrv_to_sleep(data: pd.DataFrame, sleep: pd.DataFrame,
                       settings: dict) -> pd.DataFrame:
    """Match whole five-minute intervals to selected, non-overlapping main sleep."""
    columns = KEY + ["timestamp", "rmssd", "coverage", "audit_end_seconds",
                     "audit_structurally_usable"]
    result = data[columns].copy().reset_index(drop=True)
    result["audit_start_seconds"] = result["audit_end_seconds"] - CGM_INTERVAL_SECONDS
    result["matched_sleep_end_day"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    selected = sleep.loc[sleep["review_selected"]]
    for key, indices in result.groupby(PARTICIPANT_KEY).indices.items():
        sessions = selected.loc[selected["id"].eq(key[0]) & selected["study_interval"].eq(key[1])]
        sessions = sessions.sort_values("audit_start_seconds")
        if sessions.empty:
            continue
        starts = result.loc[indices, "audit_start_seconds"].to_numpy()
        ends = result.loc[indices, "audit_end_seconds"].to_numpy()
        positions = np.searchsorted(sessions["audit_start_seconds"].to_numpy(), starts, side="right") - 1
        clipped = np.maximum(positions, 0)
        matched = (positions >= 0) & np.isfinite(starts) & np.isfinite(ends)
        matched &= ends <= sessions["audit_end_seconds"].to_numpy()[clipped]
        days = sessions["day_in_study"].to_numpy()[clipped]
        result.loc[indices[matched], "matched_sleep_end_day"] = days[matched]
    result["matched_main_sleep"] = result["matched_sleep_end_day"].notna()
    coverage = pd.to_numeric(result["coverage"], errors="coerce")
    result["usable_in_main_sleep"] = primary_records(result, "rmssd") & result["matched_main_sleep"]
    result["usable_in_main_sleep"] &= coverage.ge(settings["min_hrv_coverage"])
    return result


def sleep_hrv_counts(intervals: pd.DataFrame) -> pd.DataFrame:
    matched = intervals.loc[intervals["matched_main_sleep"]].copy()
    matched["day_in_study"] = matched["matched_sleep_end_day"].astype("int64")
    return matched.groupby(KEY)["usable_in_main_sleep"].sum().rename(
        "hrv_main_sleep_quality_windows").reset_index()


def lag_daily_columns(daily: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Join by the preceding calendar day, never by the preceding available row."""
    result = daily[KEY + list(columns)].rename(columns=columns).copy()
    result["day_in_study"] += 1
    return result


def add_calendar_history(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result["history_usable_days"] = 0
    for lag in range(temporal_settings()["history_calendar_days"]):
        previous = daily[KEY + ["aligned_inputs_available"]].copy()
        previous["day_in_study"] += lag
        previous = previous.rename(columns={"aligned_inputs_available": "_history_usable"})
        result = join_daily(result, previous)
        result["history_usable_days"] += result.pop("_history_usable").fillna(False).astype(int)
    # Historical phase labels are not predictors and need not be present.
    result["eligible_prediction"] = result["history_usable_days"].eq(3) & result["phase"].notna()
    result["history_first_aligned_day"] = result["day_in_study"] - 2
    result["history_first_glucose_day"] = result["day_in_study"] - 3
    return result


def aligned_morning_days(quality: pd.DataFrame, sleep: pd.DataFrame,
                         temperature: pd.DataFrame, intervals: pd.DataFrame,
                         settings: dict, cutoff_hour: int) -> pd.DataFrame:
    columns = KEY + ["phase", "luteal_status", "exact_luteal_onset",
                     "uncertain_luteal_entry", "quality_units_accepted"]
    result = quality[columns].copy()
    glucose = {column: f"prior_day_{column}" for column in quality if column.startswith("glucose_")}
    glucose["quality_glucose"] = "aligned_glucose"
    result = join_daily(result, lag_daily_columns(quality, glucose))
    result = join_daily(result, quality[KEY + ["quality_sleeping_hr", "sleeping_hr_bpm",
                                               "sleeping_hr_source_timestamp"]])
    selected = sleep.loc[sleep["review_selected"], KEY + ["audit_start_seconds", "audit_end_seconds"]]
    result = join_daily(result, selected.rename(columns={"audit_start_seconds": "main_sleep_start_seconds",
                                                       "audit_end_seconds": "main_sleep_end_seconds"}))
    result["prediction_cutoff_seconds"] = result["day_in_study"] * SECONDS_PER_DAY + cutoff_hour * 3600
    result["prior_day_source_day"] = result["day_in_study"] - 1
    result["aligned_main_sleep"] = result["main_sleep_end_seconds"].le(result["prediction_cutoff_seconds"])
    result["sleeping_hr_source_day"] = result["day_in_study"]
    result["aligned_sleeping_hr"] = result.pop("quality_sleeping_hr").fillna(False).astype(bool)
    result["aligned_sleeping_hr"] &= result["aligned_main_sleep"]
    result = join_daily(result, sleep_hrv_counts(intervals))
    result["hrv_main_sleep_quality_windows"] = result["hrv_main_sleep_quality_windows"].fillna(0).astype(int)
    result["aligned_hrv"] = result["hrv_main_sleep_quality_windows"].ge(settings["min_hrv_windows"])
    result["aligned_hrv"] &= result["aligned_main_sleep"]
    result = join_daily(result, selected_day_flag(temperature, "_matched_temperature"))
    result["aligned_temperature"] = result.pop("_matched_temperature").fillna(False).astype(bool)
    result["aligned_temperature"] &= result["aligned_main_sleep"]
    flags = ["quality_units_accepted", "aligned_glucose", "aligned_main_sleep", "aligned_sleeping_hr",
             "aligned_hrv", "aligned_temperature"]
    result[flags] = result[flags].fillna(False).astype(bool)
    result["aligned_inputs_available"] = result[flags].all(axis=1)
    return add_calendar_history(result)


def temporal_attrition(daily: pd.DataFrame, cutoff_hour: int) -> pd.DataFrame:
    mask = daily["phase"].notna()
    rows = [{"stage": "recognised_labels", **availability_counts(daily, mask)}]
    for column in ["quality_units_accepted", "aligned_glucose", "aligned_main_sleep", "aligned_sleeping_hr",
                   "aligned_hrv", "aligned_temperature", "eligible_prediction"]:
        mask &= daily[column]
        rows.append({"stage": column, **availability_counts(daily, mask)})
    return pd.DataFrame(rows).assign(cutoff_hour=cutoff_hour)


def temporal_protocol_text(attrition: pd.DataFrame, intervals: pd.DataFrame) -> str:
    lines = ["# Fixed-cutoff alignment and three-day eligibility: mcPHASES 2022", "",
             f"Primary cutoff: {PRIMARY_CUTOFF_HOUR:02d}:00. One three-calendar-day history is used.",
             "These are analyst-selected protocol choices, not optimised or clinically validated settings. Only noon is evaluated.",
             "Noon was selected before modelling to admit more completed main-sleep sessions; no current-day daytime inputs are added.",
             "Sleeping HR replaces the daily RHR requirement and comes from sleep_score.csv.resting_heart_rate (bpm).", "",
             "For prediction day t, glucose uses day t-1; sleeping HR uses the selected main sleep ending on t.",
             "Sleep and matching temperature use the selected main sleep ending on day t by cutoff.",
             "Sleep-score HR requires an exact end-day/end-time match to one selected main sleep; naps and unmatched or ambiguous scores are excluded.",
             "The supplied sleeping HR aggregation is unspecified; the score timestamp is not assumed to be a publication time.",
             "HRV intervals must fit entirely inside that sleep (start included, end included);",
             "five-minute interval completion is checked, including midnight crossings.",
             "The existing CGM quality thresholds and HRV coverage/count rules are retained.",
             "IDs 6 and 11 have user-confirmed mg/dL values, standardised to mmol/L; they pass the unit gate.",
             "Their inclusion still requires the same signal, timing and history rules as every participant.", "",
             "Each prediction requires usable aligned inputs on t-2, t-1 and t, with no missing days.",
             "This uses glucose source dates t-3 through t-1 and sleeping HR/HRV/temperature sleep end dates t-2 through t.",
             "Only the target day's recognised phase label is required; historical labels are not inputs.",
             "No imputation, full-cycle normalisation or future data is used. No seven-day history is required.", "",
             "## Cumulative eligibility", "",
             "| Cutoff | Requirement added | Participants | Labelled days | Exact transition dates |",
             "|---|---|---:|---:|---:|"]
    for row in attrition.itertuples():
        lines.append(f"| {row.cutoff_hour:02d}:00 | {row.stage} | {row.participants} | {row.labelled_days} | {row.exact_transitions} |")
    lines += ["", f"HRV interval diagnostics: {len(intervals)} retained intervals; "
              f"{int(intervals['matched_main_sleep'].sum())} fit a selected main sleep; "
              f"{int(intervals['usable_in_main_sleep'].sum())} also pass HRV quality checks.", "",
              "## Assumptions and limitations", "",
              "Export clocks are assumed to be mutually aligned participant-local clocks. Timezones and",
              "daylight-saving changes are not verified; no timezone conversion is applied.",
              "Completed prior-day summaries and completed sleep measurements are assumed available by cutoff.",
              "Sync, publication and revision times are unavailable/unverified. A sleep end before cutoff",
              "does not prove actual availability; no arbitrary processing buffer is treated as proof.",
              "This is a retrospective timing-constrained cohort under stated availability assumptions,",
              "not demonstrated real-time deployment. No model-ready feature matrix or data splits are produced.",
              "Transition counts only require an eligible reference date. Event matching, neighbouring",
              "evaluation days and onset scoring remain to be specified before model evaluation.",
              "All three modalities use identical eligible observation keys at the single noon cutoff."]
    return "\n".join(lines) + "\n"


def save_hr_source_comparison(aligned: pd.DataFrame, quality: pd.DataFrame,
                              args: argparse.Namespace) -> None:
    """Isolate the HR source change on the same calendar and unchanged other rules."""
    legacy = join_daily(aligned, lag_daily_columns(quality, {"quality_rhr": "_legacy_rhr"}))
    flags = ["quality_units_accepted", "aligned_glucose", "aligned_main_sleep",
             "aligned_hrv", "aligned_temperature"]
    legacy["aligned_inputs_available"] = legacy[flags].all(axis=1) & legacy["_legacy_rhr"].fillna(False)
    legacy = add_calendar_history(legacy)
    rows = [{"hr_source": name, **availability_counts(data, data["eligible_prediction"])}
            for name, data in [("legacy_prior_day_rhr", legacy), ("same_sleep_score_hr", aligned)]]
    pd.DataFrame(rows).to_csv(args.report_dir / "hr_source_cohort_comparison.csv", index=False)
    changes = aligned[KEY + ["eligible_prediction"]].rename(
        columns={"eligible_prediction": "eligible_sleeping_hr"})
    changes["eligible_legacy_rhr"] = legacy["eligible_prediction"]
    changes.loc[changes.eligible_sleeping_hr.ne(changes.eligible_legacy_rhr)].to_csv(
        args.report_dir / "hr_source_eligibility_changes.csv", index=False)
    gained = int((changes.eligible_sleeping_hr & ~changes.eligible_legacy_rhr).sum())
    lost = int((~changes.eligible_sleeping_hr & changes.eligible_legacy_rhr).sum())
    lines = ["# Sleeping HR replacement decision", "",
             "Use Fitbit-reported sleep_score.csv.resting_heart_rate in bpm instead of daily RHR.",
             "Require finite positive HR and an exact unambiguous selected-main-sleep end match.",
             "Exclude non-main sessions, unmatched ends and conflicting keys; do not average or guess matches.",
             "Sleeping HR/HRV/temperature use the same main sleep ending on t by noon; CGM uses t-1.",
             "All required signals must pass on t-2, t-1 and t. Daily RHR is descriptive only.", "",
             "| HR source | Participants | Labelled days | Exact transition dates |",
             "|---|---:|---:|---:|"]
    lines += [f"| {row['hr_source']} | {row['participants']} | {row['labelled_days']} | {row['exact_transitions']} |" for row in rows]
    lines += ["", f"Compared with daily RHR: {gained} eligible days gained and {lost} lost.",
              "See hr_source_eligibility_changes.csv for observation keys and sleeping_hr_excluded_scores.csv for excluded sources.",
              "Unmatched ends are excluded; end-time discrepancies are retained in sleeping_hr_session_review_2022.csv and their cause is unverified.",
              "Noon, CGM/HRV/temperature quality and history rules are unchanged. These are cohort counts, not model performance.",
              "The HR aggregation algorithm and actual publication/sync times remain unspecified."]
    (args.report_dir / "sleeping_hr_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_temporal_review(tables: dict, daily: pd.DataFrame, args: argparse.Namespace) -> None:
    settings = eligibility_settings(args)
    quality, sleep, temperature = build_quality_review(tables, daily, settings)
    intervals = align_hrv_to_sleep(tables["heart_rate_variability_details"], sleep, settings)
    intervals.to_csv(args.processed_dir / "hrv_sleep_alignment_2022.csv", index=False)
    hour = PRIMARY_CUTOFF_HOUR
    aligned = aligned_morning_days(quality, sleep, temperature, intervals, settings, hour)
    aligned.to_csv(args.processed_dir / f"morning_alignment_{hour:02d}_2022.csv", index=False)
    aligned.loc[aligned["eligible_prediction"]].to_csv(
        args.processed_dir / f"eligible_observations_{hour:02d}_2022.csv", index=False)
    save_hr_source_comparison(aligned, quality, args)
    attrition = temporal_attrition(aligned, hour)
    attrition.to_csv(args.report_dir / "temporal_attrition.csv", index=False)
    (args.report_dir / "temporal_protocol.md").write_text(
        temporal_protocol_text(attrition, intervals), encoding="utf-8")


def build_quality_review(tables: dict, daily: pd.DataFrame, settings: dict) -> tuple:
    sleep = review_sleep_sessions(tables["sleep"], settings)
    temperature = review_temperature_sessions(tables["computed_temperature"], sleep)
    scores = review_sleeping_heart_rate(tables["sleep_score"], sleep)
    result = join_daily(daily, sleeping_hr_daily(scores))
    result["quality_sleeping_hr"] = result["quality_sleeping_hr"].fillna(False).astype(bool)
    result = join_daily(result, hrv_quality_counts(tables["heart_rate_variability_details"], settings))
    result = join_daily(result, selected_day_flag(sleep, "quality_main_sleep"))
    result = join_daily(result, selected_day_flag(temperature, "quality_temperature"))
    for column in ["hrv_quality_windows", "hrv_bounded_quality_windows", "hrv_coverage_above_one_windows"]:
        result[column] = result[column].fillna(0).astype("int64")
    result[["quality_main_sleep", "quality_temperature"]] = result[["quality_main_sleep", "quality_temperature"]].fillna(False).astype(bool)
    result["quality_units_accepted"] = ~result["id"].isin(settings["unresolved_glucose_ids"])
    result["quality_glucose"] = result["glucose_unique_samples"].gt(0)
    result["quality_glucose"] &= result["glucose_nominal_coverage_fraction"].ge(settings["min_glucose_coverage"])
    result["quality_glucose"] &= result["glucose_longest_uncovered_seconds"].le(settings["max_glucose_gap_hours"] * 3600)
    result["quality_hrv"] = result["hrv_quality_windows"].ge(settings["min_hrv_windows"])
    result["quality_rhr"] = result["resting_heart_rate_available_records"].gt(0)
    flags = ["quality_units_accepted", "quality_glucose", "quality_hrv", "quality_sleeping_hr",
             "quality_main_sleep", "quality_temperature"]
    result["same_day_quality_overlap"] = result[flags].all(axis=1) & result["phase"].notna()
    return result, sleep, temperature


def availability_counts(daily: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False).astype(bool)
    return {"participants": int(daily.loc[mask, "id"].nunique()), "days": int(mask.sum()),
            "labelled_days": int((mask & daily["phase"].notna()).sum()),
            "exact_transitions": int((mask & daily["exact_luteal_onset"].eq(1).fillna(False)).sum()),
            "uncertain_entries": int((mask & daily["uncertain_luteal_entry"]).sum())}


def rule_effect(daily: pd.DataFrame, rule: str, before: pd.Series, after: pd.Series) -> dict:
    result = {"rule": rule}
    for prefix, mask in [("before", before), ("after", after)]:
        result.update({f"{prefix}_{key}": value for key, value in availability_counts(daily, mask).items()})
    return result


def quality_rule_effects(daily: pd.DataFrame, settings: dict) -> pd.DataFrame:
    labelled = daily["phase"].notna()
    cases = [
        ("standardised_glucose_units_from_confirmed_mapping", labelled, labelled & daily["quality_units_accepted"]),
        ("sleeping_heart_rate_selected_main_sleep", daily["sleep_score_records"].gt(0), daily["quality_sleeping_hr"]),
        ("hrv_minimum_coverage_and_window_count", daily["heart_rate_variability_details_available_records"].gt(0), daily["quality_hrv"]),
        ("hrv_optional_upper_bound_one_sensitivity", daily["quality_hrv"], daily["hrv_bounded_quality_windows"].ge(settings["min_hrv_windows"])),
        ("glucose_observed_time_and_longest_gap", daily["glucose_available_records"].gt(0), daily["quality_glucose"]),
        ("one_valid_nonoverlapping_main_sleep", daily["sleep_available_records"].gt(0), daily["quality_main_sleep"]),
        ("skin_temperature_matches_selected_main_sleep", daily["computed_temperature_available_records"].gt(0), daily["quality_temperature"]),
    ]
    return pd.DataFrame(rule_effect(daily, name, before, after) for name, before, after in cases)


def quality_attrition(daily: pd.DataFrame) -> pd.DataFrame:
    mask = daily["phase"].notna()
    rows = [{"stage": "recognised_labels", **availability_counts(daily, mask)}]
    for column in ["quality_units_accepted", "quality_glucose", "quality_main_sleep", "quality_sleeping_hr", "quality_hrv",
                   "quality_temperature"]:
        mask &= daily[column]
        rows.append({"stage": column, **availability_counts(daily, mask)})
    return pd.DataFrame(rows)


def quality_sensitivity(daily: pd.DataFrame, hrv: pd.DataFrame, settings: dict) -> pd.DataFrame:
    rows = []
    for fraction in sorted({0.7, 0.8, 0.9, settings["min_glucose_coverage"]}):
        for gap_hours in sorted({2.0, 4.0, settings["max_glucose_gap_hours"]}):
            mask = daily["glucose_unique_samples"].gt(0) & daily["glucose_nominal_coverage_fraction"].ge(fraction)
            mask &= daily["glucose_longest_uncovered_seconds"].le(gap_hours * 3600)
            rows.append({"signal": "glucose", "minimum_coverage": fraction,
                         "maximum_gap_hours": gap_hours, **availability_counts(daily, mask)})
    for fraction in sorted({0.7, 0.8, 0.9, settings["min_hrv_coverage"]}):
        counts = hrv_quality_counts(hrv, settings | {"min_hrv_coverage": fraction})
        joined = join_daily(daily[KEY], counts)
        for bounded in [False, True]:
            for windows in sorted({3, 6, 12, settings["min_hrv_windows"]}):
                column = "hrv_bounded_quality_windows" if bounded else "hrv_quality_windows"
                mask = joined[column].fillna(0).ge(windows)
                rows.append({"signal": "hrv", "minimum_coverage": fraction,
                             "upper_bound_one": bounded, "minimum_windows": windows,
                             **availability_counts(daily, mask)})
    return pd.DataFrame(rows)


def session_end_sensitivity(daily: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    rows = []
    for name, data in sessions.items():
        for hour in [6, 8, 10, 12]:
            selected = data["review_selected"] & (data["audit_end_seconds"] % 86400).le(hour * 3600)
            keys = pd.MultiIndex.from_frame(data.loc[selected, KEY])
            mask = pd.Series(pd.MultiIndex.from_frame(daily[KEY]).isin(keys), index=daily.index)
            rows.append({"table": name, "hypothetical_end_clock_hour": hour,
                         **availability_counts(daily, mask)})
    return pd.DataFrame(rows)


def unit_evidence(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in data.groupby(PARTICIPANT_KEY):
        values = pd.to_numeric(group["glucose_value"], errors="coerce")
        values = values.loc[values.notna() & np.isfinite(values) & values.gt(0)]
        source = pd.to_numeric(group["glucose_source_value"], errors="coerce")
        source = source.loc[source.notna() & np.isfinite(source) & source.gt(0)]
        stats = {"minimum": values.min(), "q25": values.quantile(0.25),
                 "median": values.median() if len(values) else np.nan,
                 "q75": values.quantile(0.75), "maximum": values.max()}
        rows.append(dict(zip(PARTICIPANT_KEY, key)) | stats | {
            "numeric_records": len(values), "integer_value_fraction": float(values.mod(1).eq(0).mean()) if len(values) else np.nan,
            "source_minimum": source.min(), "source_median": source.median() if len(source) else np.nan,
            "source_maximum": source.max(),
            "source_integer_value_fraction": float(source.mod(1).eq(0).mean()) if len(source) else np.nan,
            "source_unit": group["glucose_source_unit"].iloc[0], "standardised_unit": "mmol/L",
            "conversion_factor": group["glucose_conversion_factor"].iloc[0],
            "converted_records": int(group["glucose_unit_converted"].sum()),
            "units_unconfirmed": key[0] in UNRESOLVED_GLUCOSE_IDS,
            "primary_participant_exclusion": key[0] in UNRESOLVED_GLUCOSE_IDS,
        })
    return pd.DataFrame(rows)


def decision_log_text(effects: pd.DataFrame, attrition: pd.DataFrame,
                      sessions: pd.DataFrame, transitions: pd.DataFrame, settings: dict,
                      evidence: dict) -> str:
    lines = ["# Eligibility decision log: mcPHASES 2022", "",
             "Effects below use each signal's exported calendar day. They do not apply a morning cutoff,",
             "prior-day CGM lag, or trailing histories. They must not be used as the final modelling cohort.",
             "A transition counts as available here when its reference day passes the rule; neighbouring",
             "days and detection performance are not assessed. Independent effects are not additive.", "",
             "1. **Glucose units:** the user confirmed mg/dL for IDs 6 and 11 on 2026-10-12.",
             "   Standardise these participants to mmol/L by multiplying by 0.0555 in processed data only.",
             "   Preserve source values, source units, conversion flags and factors; raw exports are unchanged.",
             "   Remove their previous unit-based exclusion from all three primary input sets.",
             "   Other participants provisionally follow the documented units, without individual unit validation.",
             "   The mapping follows the user's confirmation, not magnitude alone; provider confirmation was not independently verified.",
             "   The unit-evidence table reports both source-scale and standardised distributions.",
             "2. **Resting heart rate:** exact duplicates are removed and conflicting daily keys excluded.",
             f"   Retained 2022 RHR records have {evidence['rhr_conflicting_rows']} conflicting-key rows; no aggregation is needed when this is zero.",
             "   Equal values on different days",
             "   are retained for descriptive audit only. Daily RHR is not an eligibility requirement.",
             "   Sleeping HR uses finite positive sleep_score.csv.resting_heart_rate with an unambiguous exact selected-main-sleep end match.",
             "   Naps, unmatched ends and conflicting score keys are excluded without averaging or nearest-time substitution.",
             "3. **HRV coverage:** Fitbit defines interbeat-interval completeness without an upper bound;",
             "   the local README describes a nominal five-minute ratio but cautions that definitions may be approximate.",
             f"   Require coverage >= {settings['min_hrv_coverage']:g} and >= {settings['min_hrv_windows']} usable intervals/day; retain overshoots",
             "   without clipping or coverage weighting. Their cause is unverified. Report exclusion of coverage >1",
             "   as a sensitivity analysis. These are analyst-selected quality rules, not validated clinical thresholds.",
             f"   Observed coverage range: {evidence['hrv_coverage_min']:g} to {evidence['hrv_coverage_max']:g}; {evidence['hrv_above_one_rows']} records exceed one.",
             f"4. **CGM completeness:** require nominal observed-time fraction >= {settings['min_glucose_coverage']:g} and longest",
             f"   uncovered gap <= {settings['max_glucose_gap_hours']:g} hours, including boundaries. Counts and cadence are reported separately.",
             "   Forward five-minute intervals are a sampling assumption; no interpolation or gap filling is used.",
             "   These analyst-selected rules are compared with alternative coverage/gap settings before modelling.",
             "5. **Sleep/temperature:** select one structurally usable, non-overlapping main sleep per end day;",
             "   reject ambiguous multiple-main days without choosing by length. Allow <=60 seconds disagreement",
             "   between sleep elapsed time and duration to accommodate minute granularity (an analyst assumption).",
             "   Require positive integer skin-temperature samples and an exact start/end match to selected main sleep.",
             "   Retain unmatched sessions for review. Hypothetical end-clock counts do not select a cutoff or establish",
             "   actual availability: Fitbit reports post-sleep, post-sync HRV processing; sync times/timezones are unverified.",
             f"   {evidence['temperature_unmatched_rows']} temperature records do not match a selected main sleep session.",
             "6. **Phase transitions:** require adjacent recognised non-luteal/luteal labels for exact onset.",
             "   Separate interval-censored candidates, left-censored entries, and possible luteal continuations",
             "   after gaps. Do not impute labels or count uncertain entries as additional exact events.", "",
             "## Rule effects (independent comparisons)", "",
             "| Rule | Participants before → after | Labelled days before → after | Exact transitions before → after |",
             "|---|---:|---:|---:|"]
    for row in effects.itertuples():
        lines.append(f"| {row.rule} | {row.before_participants} → {row.after_participants} | "
                     f"{row.before_labelled_days} → {row.after_labelled_days} | {row.before_exact_transitions} → {row.after_exact_transitions} |")
    lines += ["", "## Cumulative same-day overlap preview", "",
              "| Stage | Participants | Labelled days | Exact transitions |", "|---|---:|---:|---:|"]
    for row in attrition.itertuples():
        lines.append(f"| {row.stage} | {row.participants} | {row.labelled_days} | {row.exact_transitions} |")
    lines += ["", "Transition categories: " + ", ".join(f"{key}={value}" for key, value in transitions["transition_status"].value_counts().items()) + ".",
              "Session counts (cross-midnight / days with multiple sessions):"]
    for name, group in sessions.groupby("table"):
        multiple = group.loc[group["sessions_on_end_day"].gt(1), KEY].drop_duplicates()
        lines.append(f"- {name}: {int(group['cross_midnight'].sum())} / {len(multiple)}.")
    lines += ["", "Sources reviewed " + SOURCE_REVIEW_DATE + ": raw/README.txt and"]
    lines += [f"- [{name}]({source['url']})" for name, source in EVIDENCE_SOURCES.items()]
    lines += ["", "Remaining: cause of HRV overshoot, fixed morning cutoff,",
              "clock/sync availability assumptions, trailing-history rules, and the final common cohort."]
    return "\n".join(lines) + "\n"


def save_eligibility_review(tables: dict, daily: pd.DataFrame, args: argparse.Namespace) -> None:
    settings = eligibility_settings(args)
    quality, sleep, temperature = build_quality_review(tables, daily, settings)
    scores = review_sleeping_heart_rate(tables["sleep_score"], sleep)
    scores.to_csv(args.processed_dir / "sleeping_hr_session_review_2022.csv", index=False)
    scores.loc[~scores["review_selected"]].to_csv(args.report_dir / "sleeping_hr_excluded_scores.csv", index=False)
    transitions = phase_transition_review(daily)
    sessions = pd.concat([session_diagnostics(sleep, "sleep"),
                          session_diagnostics(temperature, "computed_temperature")], ignore_index=True)
    effects = quality_rule_effects(quality, settings)
    attrition = quality_attrition(quality)
    quality.to_csv(args.processed_dir / "quality_review_daily_2022.csv", index=False)
    sessions.to_csv(args.processed_dir / "session_review_2022.csv", index=False)
    transitions.to_csv(args.processed_dir / "phase_transition_review_2022.csv", index=False)
    effects.to_csv(args.report_dir / "rule_effects.csv", index=False)
    attrition.to_csv(args.report_dir / "same_day_quality_attrition.csv", index=False)
    unit_evidence(tables["glucose"]).to_csv(args.report_dir / "glucose_unit_evidence.csv", index=False)
    quality_sensitivity(quality, tables["heart_rate_variability_details"], settings).to_csv(
        args.report_dir / "coverage_sensitivity.csv", index=False)
    session_end_sensitivity(quality, {"sleep": sleep, "computed_temperature": temperature}).to_csv(
        args.report_dir / "session_end_time_sensitivity.csv", index=False)
    provenance = {"reviewed_on": SOURCE_REVIEW_DATE, "sources": EVIDENCE_SOURCES,
                  "local_readme_sha256": file_sha256(args.raw_dir / "README.txt") if (args.raw_dir / "README.txt").exists() else None,
                  "settings": settings, "glucose_unit_policy": glucose_unit_policy(), "model_ready": False}
    (args.report_dir / "decision_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    coverage = pd.to_numeric(tables["heart_rate_variability_details"]["coverage"], errors="coerce")
    evidence = {"rhr_conflicting_rows": int(tables["resting_heart_rate"]["audit_conflicting_key"].sum()),
                "hrv_coverage_min": coverage.min(), "hrv_coverage_max": coverage.max(),
                "hrv_above_one_rows": int(coverage.gt(1).sum()),
                "temperature_unmatched_rows": int((~temperature["review_matches_selected_main_sleep"]).sum())}
    (args.report_dir / "decision_log.md").write_text(
        decision_log_text(effects, attrition, sessions, transitions, settings, evidence), encoding="utf-8")


def describe_values(data: pd.DataFrame, name: str, spec: dict) -> list[dict]:
    records = []
    for key, group in data.groupby(PARTICIPANT_KEY):
        for column in spec["values"]:
            values = pd.to_numeric(group[column], errors="coerce")
            values = values.loc[np.isfinite(values)]
            records.append(dict(zip(PARTICIPANT_KEY, key)) | {
                "table": name, "column": column, "nonmissing_numeric_rows": int(values.count()),
                "minimum": values.min(), "q25": values.quantile(0.25),
                "median": values.median() if len(values) else np.nan,
                "q75": values.quantile(0.75), "maximum": values.max(),
            })
    return records


def participant_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    grouped = daily.groupby(PARTICIPANT_KEY)
    result = grouped["day_in_study"].agg(first_day="min", last_day="max", calendar_days="size")
    result["labelled_days"] = grouped["phase"].count()
    result["exact_luteal_onsets"] = grouped["exact_luteal_onset"].sum()
    result["uncertain_luteal_entries"] = grouped["uncertain_luteal_entry"].sum()
    for name in TABLE_SPECS:
        if name == "hormones_and_selfreport":
            continue
        result[f"{name}_observed_days"] = daily[f"{name}_records"].gt(0).groupby(
            [daily[column] for column in PARTICIPANT_KEY]).sum()
        result[f"{name}_available_days"] = daily[f"{name}_available_records"].gt(0).groupby(
            [daily[column] for column in PARTICIPANT_KEY]).sum()
    return result.reset_index()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_files(raw_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw_dir.glob("*.csv")):
        audited = path.stem in TABLE_SPECS
        records.append({"file": path.name, "bytes": path.stat().st_size,
                        "columns": ",".join(pd.read_csv(path, nrows=0).columns),
                        "fully_audited_2022": audited,
                        "sha256": file_sha256(path) if audited else ""})
    return pd.DataFrame(records)


def build_daily_audit(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    daily = add_labels(create_daily_grid(tables), tables["hormones_and_selfreport"])
    for name, data in tables.items():
        daily = join_daily(daily, daily_record_counts(data, name))
    count_columns = [column for column in daily if column.endswith(("_records", "_rows"))]
    daily[count_columns] = daily[count_columns].fillna(0).astype("int64")
    daily = join_daily(daily, glucose_coverage(tables["glucose"]))
    defaults = glucose_day_diagnostics(np.array([]))
    daily[list(defaults)] = daily[list(defaults)].fillna(defaults)
    return daily.sort_values(KEY).reset_index(drop=True)


def write_report(report_dir: Path, daily: pd.DataFrame) -> None:
    exact = int(daily["exact_luteal_onset"].sum())
    uncertain = int(daily["uncertain_luteal_entry"].sum())
    text = f"""# mcPHASES 2022 descriptive audit

Participants: {daily['id'].nunique()}
Calendar participant-days across the union of audited recording spans: {len(daily)}
Days with an unambiguous recognised supplied phase: {daily['phase'].notna().sum()}
Exact observed non-luteal-to-luteal transitions: {exact}
Luteal entries with an unknown immediately preceding label: {uncertain}

## Scope and interpretation

Only study_interval == 2022 is retained. Seven tables are fully audited:
{', '.join(TABLE_SPECS)}. Other CSV files are inventoried by size and header
only. In particular, continuous heart_rate.csv has not been quality-audited.
This is descriptive preparation, not a model-ready cohort or a performance evaluation.
When the eligibility review is enabled, decision_log.md describes additional
quality policies and their effects separately from this structural audit.
No morning cutoff, history requirement, imputation, scaling or cohort threshold
has been applied. Days span the union of the audited tables, not an assumed
fixed 90-day study. Session counts are assigned to their exported end day.

## Duplicate and quality policy

Exact duplicate rows are removed within 2022. Every member of a remaining
duplicate measurement key is flagged as conflicting and retained for review;
none contributes to available-record counts. Conflicting label days have no
reference outcome. Invalid clock times, reversed/zero-duration sessions,
non-finite or non-numeric values, non-positive primary sensor values (except
temperature), and negative hormone values are flagged. Missing values remain
missing. Available-record counts require the primary value and no structural
flags across the audited numeric fields. These are conservative audit counts,
not final physiological validity or modelling eligibility.
Unknown phase strings are excluded from reference outcomes and listed separately.
No measurement values are averaged or imputed. Confirmed glucose units are standardised.

## Coverage and units

CGM coverage is an explicit nominal estimate: each usable measurement covers
the following 300 seconds, clipped at midnight; overlapping intervals count
once. Longest uncovered time includes day boundaries. This assumption does not
prove continuous sensor validity and is not a final minimum-coverage rule.
HRV timestamps denote the start of five-minute intervals; audit_end_seconds
includes those five minutes. HRV coverage above one is flagged in the summary
but retained pending verification of its definition.

Units follow raw/README.txt provisionally: glucose mmol/L, resting HR bpm,
skin temperature degrees Celsius. HRV units and device coverage semantics
need verification. On 2026-10-12 the user confirmed mg/dL for participants 6 and 11.
Their processed glucose_value is multiplied by 0.0555 to obtain mmol/L; source values,
units, factors and conversion flags are preserved. This confirmation was not
independently verified with the provider. Other participants follow the documented units.
Both source and standardised distributions are reported. Missing 2022 PdG values
must not be imputed as independent confirmation of phase labels.

## Reference labels and next steps

Phases are normalised only by whitespace and case. Fertility is retained as
supplied, not relabelled as confirmed ovulation. Luteal status is unknown on
missing, unknown or conflicting label days. Exact onset requires an observed
non-luteal label on the immediately preceding calendar day. Initial luteal
observations and entries after missing labels are uncertain.
Supplied labels are not independently hormone-validated ground truth.

Before modelling, verify timestamp and
availability conventions, fix a morning cutoff, align prior-day CGM and eligible
overnight wearable measurements, and define shared coverage/history rules.
No clinical or causal conclusions can be drawn from this audit.

## Reproducibility

run_config.json records arguments, software versions, and source SHA-256 hashes.
Hashes of all audited CSVs are checked again before saving outputs.
Daily joins enforce one-to-one keys. Outputs never overwrite a non-empty
directory; use new --processed-dir and --report-dir paths for another run.
"""
    (report_dir / "audit_report.md").write_text(text, encoding="utf-8")


def save_outputs(tables: dict, daily: pd.DataFrame, summaries: list,
                 checks: list, distributions: list, inventory: pd.DataFrame,
                 args: argparse.Namespace) -> None:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        data.to_csv(args.processed_dir / f"{name}_2022_audited.csv", index=False)
        flags = data["audit_conflicting_key"] | data["audit_invalid_time"] | data["audit_invalid_value"]
        data.loc[flags].to_csv(args.report_dir / f"{name}_flagged_rows.csv", index=False)
        data.drop(columns=[column for column in data if column.startswith("audit_")]).isna().sum().rename(
            "missing_rows").rename_axis("column").to_csv(args.report_dir / f"{name}_missingness.csv")
    daily.to_csv(args.processed_dir / "daily_audit_2022.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.report_dir / "table_audit.csv", index=False)
    pd.DataFrame(checks).to_csv(args.report_dir / "numeric_missingness.csv", index=False)
    pd.DataFrame(distributions).to_csv(args.report_dir / "value_distributions.csv", index=False)
    participant_coverage(daily).to_csv(args.report_dir / "participant_coverage.csv", index=False)
    inventory.to_csv(args.report_dir / "file_inventory.csv", index=False)
    labels = tables["hormones_and_selfreport"]["phase"].astype("string").str.strip().str.lower()
    labels.value_counts(dropna=False).rename_axis("phase").rename("rows").to_csv(args.report_dir / "phase_counts.csv")
    unknown = labels.notna() & ~labels.isin(PHASES) & labels.ne("")
    tables["hormones_and_selfreport"].loc[unknown].to_csv(args.report_dir / "unknown_phase_rows.csv", index=False)
    write_report(args.report_dir, daily)


def run_audit(args: argparse.Namespace) -> pd.DataFrame:
    settings = eligibility_settings(args)
    if getattr(args, "temporal_review", False):
        args.eligibility_review = True
    for attribute in ["raw_dir", "processed_dir", "report_dir"]:
        setattr(args, attribute, getattr(args, attribute).resolve())
    validate_output_paths(args.raw_dir, [args.processed_dir, args.report_dir])
    inventory = inventory_files(args.raw_dir)
    tables, summaries, checks, distributions = {}, [], [], []
    for name, spec in TABLE_SPECS.items():
        print(f"Auditing {name}: 2022 only", flush=True)
        data, summary, numeric_checks = audit_table(args.raw_dir, name, spec)
        tables[name] = data
        summaries.append(summary)
        checks.extend({"table": name, **check} for check in numeric_checks)
        distributions.extend(describe_values(data, name, spec))
    daily = build_daily_audit(tables)
    hashes = dict(inventory.loc[inventory["fully_audited_2022"], ["file", "sha256"]].itertuples(index=False, name=None))
    for filename, expected in hashes.items():
        if file_sha256(args.raw_dir / filename) != expected:
            raise RuntimeError(f"Source changed during audit: {filename}")
    save_outputs(tables, daily, summaries, checks, distributions, inventory, args)
    if getattr(args, "eligibility_review", False):
        save_eligibility_review(tables, daily, args)
    if getattr(args, "temporal_review", False):
        save_temporal_review(tables, daily, args)
    config = {"study_interval": STUDY_INTERVAL, "raw_dir": str(args.raw_dir),
              "processed_dir": str(args.processed_dir), "report_dir": str(args.report_dir),
              "python_version": platform.python_version(), "pandas_version": pd.__version__,
              "numpy_version": np.__version__, "source_sha256": hashes,
              "script_sha256": file_sha256(Path(__file__)),
              "glucose_unit_policy": glucose_unit_policy(),
              "cgm_nominal_interval_seconds": CGM_INTERVAL_SECONDS,
              "prediction_cutoff": temporal_settings()["prediction_cutoff"] if getattr(args, "temporal_review", False) else None,
              "model_ready": False,
              "temporal_review": getattr(args, "temporal_review", False),
              "temporal_settings": temporal_settings() if getattr(args, "temporal_review", False) else None,
              "eligibility_review": getattr(args, "eligibility_review", False),
              "quality_review_settings": settings if getattr(args, "eligibility_review", False) else None}
    (args.report_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Audit complete: {daily['id'].nunique()} participants; {len(daily)} calendar participant-days.")
    print(f"Report: {args.report_dir / 'audit_report.md'}")
    return daily


if __name__ == "__main__":
    run_audit(parse_arguments())
