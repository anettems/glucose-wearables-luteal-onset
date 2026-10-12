"""Reproducible, descriptive audit of the mcPHASES 2022 exports.

Run from the thesis workspace: python code/preprocessing.py
Requires pandas and NumPy. No model features, imputation or unit conversions
are produced. Supplied phase labels are not independently validated outcomes.
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
# These checks identify structural problems, not final physiological cutoffs.
TABLE_SPECS = {
    "hormones_and_selfreport": {
        "values": ["lh", "estrogen", "pdg"], "positive": [],
    },
    "glucose": {
        "values": ["glucose_value"], "positive": ["glucose_value"],
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
        summary["hrv_coverage_above_one_rows"] = int(pd.to_numeric(data["coverage"]).gt(1).sum())
    return data.sort_values(measurement_key(spec)).reset_index(drop=True), summary, checks


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
        records.append(dict(zip(KEY, key)) | glucose_day_coverage(seconds))
    columns = KEY + list(glucose_day_coverage(np.array([])))
    return pd.DataFrame(records, columns=columns)


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
    defaults = glucose_day_coverage(np.array([]))
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

Only study_interval == 2022 is retained. Six tables are fully audited:
{', '.join(TABLE_SPECS)}. Other CSV files are inventoried by size and header
only. In particular, continuous heart_rate.csv has not been quality-audited.
This is descriptive preparation, not a model-ready cohort or a performance evaluation.
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
No measurement values are averaged, converted or imputed.

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
need verification. Glucose units for participants 6 and 11 remain unresolved;
their distributions are reported without conversion. Missing 2022 PdG values
must not be imputed as independent confirmation of phase labels.

## Reference labels and next steps

Phases are normalised only by whitespace and case. Fertility is retained as
supplied, not relabelled as confirmed ovulation. Luteal status is unknown on
missing, unknown or conflicting label days. Exact onset requires an observed
non-luteal label on the immediately preceding calendar day. Initial luteal
observations and entries after missing labels are uncertain.
Supplied labels are not independently hormone-validated ground truth.

Before modelling, resolve units and duplicate semantics, verify timestamp and
availability conventions, fix a morning cutoff, align prior-day CGM and eligible
overnight wearable measurements, and define shared coverage/history rules.
No clinical or causal conclusions can be drawn from this audit.

## Reproducibility

run_config.json records arguments, software versions, and source SHA-256 hashes.
Hashes of the six audited CSVs are checked again before saving outputs.
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
    config = {"study_interval": STUDY_INTERVAL, "raw_dir": str(args.raw_dir),
              "processed_dir": str(args.processed_dir), "report_dir": str(args.report_dir),
              "python_version": platform.python_version(), "pandas_version": pd.__version__,
              "numpy_version": np.__version__, "source_sha256": hashes,
              "script_sha256": file_sha256(Path(__file__)),
              "cgm_nominal_interval_seconds": CGM_INTERVAL_SECONDS,
              "prediction_cutoff": None, "model_ready": False}
    (args.report_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Audit complete: {daily['id'].nunique()} participants; {len(daily)} calendar participant-days.")
    print(f"Report: {args.report_dir / 'audit_report.md'}")
    return daily


if __name__ == "__main__":
    run_audit(parse_arguments())
