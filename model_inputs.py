"""One-day features and training-only wearable replacement for the 14:00 protocol."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as prep


GLUCOSE_FEATURES = ["glucose_median_mmol_l", "glucose_cv_percent"]
WEARABLE_FEATURES = ["sleeping_hr_bpm", "log_median_rmssd_ms", "skin_temperature_c"]
FEATURE_SETS = {"glucose": GLUCOSE_FEATURES, "wearable": WEARABLE_FEATURES,
                "combined": GLUCOSE_FEATURES + WEARABLE_FEATURES}
WEARABLE_FLAGS = ["aligned_sleeping_hr", "aligned_hrv", "aligned_temperature"]
OUTCOMES = ["phase", "luteal_status", "exact_luteal_onset", "uncertain_luteal_entry"]


def glucose_features(glucose):
    usable = glucose.loc[prep.primary_records(glucose, "glucose_value")]
    result = usable.groupby(prep.KEY).glucose_value.agg(
        glucose_median_mmol_l="median", mean="mean", sd="std").reset_index()
    # pandas std uses the sample standard deviation (ddof=1).
    result["glucose_cv_percent"] = 100 * result.pop("sd") / result.pop("mean")
    result["day_in_study"] += 1  # Calendar lag, never the preceding available row.
    return result


def hrv_features(intervals):
    usable = intervals.loc[intervals["usable_in_main_sleep"]].copy()
    usable["day_in_study"] = usable["matched_sleep_end_day"].astype("int64")
    result = usable.groupby(prep.KEY).rmssd.median().rename("log_median_rmssd_ms")
    return np.log(result).reset_index()


def construct_features(eligible, glucose, intervals, temperature):
    """Keep invalid/unavailable summaries blank even when a source value exists."""
    metadata = prep.KEY + OUTCOMES + ["prior_day_source_day", "sleeping_hr_source_day",
        "prediction_cutoff_seconds", "main_sleep_start_seconds", "main_sleep_end_seconds",
        "valid_wearable_count"] + WEARABLE_FLAGS
    result = eligible[metadata + ["sleeping_hr_bpm"]].copy()
    result = prep.join_daily(result, glucose_features(glucose))
    result = prep.join_daily(result, hrv_features(intervals))
    selected = temperature.loc[temperature.review_selected, prep.KEY + ["nightly_temperature"]]
    result = prep.join_daily(result, selected.rename(columns={"nightly_temperature": "skin_temperature_c"}))
    for column, flag in zip(WEARABLE_FEATURES, WEARABLE_FLAGS):
        result[column] = result[column].where(result[flag])
        result[column + "_missing"] = result[column].isna()
    result = result.sort_values(prep.KEY).reset_index(drop=True)
    validate_features(result)
    return result


def validate_features(data):
    if data.duplicated(prep.KEY).any() or not data.study_interval.eq(2022).all():
        raise ValueError("Expected unique 2022 observation keys")
    features = data[FEATURE_SETS["combined"]]
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature value")
    if features[GLUCOSE_FEATURES].isna().any().any():
        raise ValueError("Required glucose summaries are missing")
    if features[WEARABLE_FEATURES].notna().sum(axis=1).lt(2).any():
        raise ValueError("Fewer than two valid wearable summaries")
    if not features[WEARABLE_FEATURES].notna().sum(axis=1).eq(data.valid_wearable_count).all():
        raise ValueError("Wearable count disagrees with feature availability")
    if data.luteal_status.isna().any():
        raise ValueError("Classification outcome is missing")
    for feature, flag in zip(WEARABLE_FEATURES, WEARABLE_FLAGS):
        if not data[feature].notna().eq(data[flag]).all():
            raise ValueError(f"Feature availability disagrees with quality flag: {feature}")
    if not data.prior_day_source_day.eq(data.day_in_study - 1).all():
        raise ValueError("Glucose must use the preceding calendar day")
    if not data.main_sleep_end_seconds.le(data.prediction_cutoff_seconds).all():
        raise ValueError("Sleep ended after the prediction time")


def replace_missing_wearables(training, evaluation):
    """Fit on training participants; return two filled copies and the fitted medians.

    Call separately for every validation split, then refit on development
    participants before final testing. Select FEATURE_SETS columns for models.
    No scaling, labels, IDs or missingness flags are added to the predictors.
    """
    if training.empty or evaluation.empty:
        raise ValueError("Training and evaluation data must both be non-empty")
    if set(training.id) & set(evaluation.id):
        raise ValueError("Training and evaluation participants must be separate")
    for data in (training, evaluation):
        validate_features(data)
    medians = training[WEARABLE_FEATURES].median()
    if medians.isna().any():
        raise ValueError("A wearable feature has no observed training values")
    outputs = []
    for data in (training, evaluation):
        result = data.copy()
        result[WEARABLE_FEATURES] = result[WEARABLE_FEATURES].fillna(medians)
        outputs.append(result)
    return outputs[0], outputs[1], medians


def prepare_datasets(processed_dir, report_dir):
    """Build shared, unimputed features from the reproducible audit outputs."""
    processed_dir, report_dir = Path(processed_dir), Path(report_dir)
    config_path = report_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("temporal_settings") != prep.temporal_settings():
        raise ValueError("Audit timing settings do not match the active protocol")
    eligible = pd.read_csv(processed_dir / "eligible_observations_14_2022.csv")
    glucose = pd.read_csv(processed_dir / "glucose_2022_audited.csv")
    intervals = pd.read_csv(processed_dir / "hrv_sleep_alignment_2022.csv")
    sleep = pd.read_csv(processed_dir / "sleep_2022_audited.csv")
    temperature = pd.read_csv(processed_dir / "computed_temperature_2022_audited.csv")
    sleep = prep.review_sleep_sessions(sleep, config["quality_review_settings"])
    temperature = prep.review_temperature_sessions(temperature, sleep)
    data = construct_features(eligible, glucose, intervals, temperature)
    save_datasets(data, processed_dir, report_dir)
    config.update(features_ready=True, model_ready=False, feature_sets=FEATURE_SETS,
                  imputation_fitted=False, participant_splits_assigned=False,
                  feature_script_sha256=prep.file_sha256(Path(__file__)),
                  entry_point_sha256=prep.file_sha256(Path(__file__).with_name("main.py")))
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return data


def save_datasets(data, processed_dir, report_dir):
    outputs = {"observations_2022.csv": data,
               "outcomes_2022.csv": data[prep.KEY + OUTCOMES]}
    for name, features in FEATURE_SETS.items():
        outputs[f"{name}_features_2022.csv"] = data[prep.KEY + features]
    if any((processed_dir / name).exists() for name in outputs):
        raise FileExistsError("Feature outputs already exist; choose a fresh output directory")
    for name, frame in outputs.items():
        frame.to_csv(processed_dir / name, index=False)
    attrition = pd.read_csv(report_dir / "temporal_attrition.csv")
    stages = ["recognised_labels", "aligned_glucose", "aligned_main_sleep", "sufficient_wearables"]
    attrition.loc[attrition.stage.isin(stages)].to_csv(report_dir / "data_reduction.csv", index=False)
    summary = {"participants": int(data.id.nunique()), "days": len(data),
               "luteal_days": int(data.luteal_status.eq(1).sum()),
               "luteal_starts": int(data.exact_luteal_onset.eq(1).sum()),
               "complete_wearable_days": int(data.valid_wearable_count.eq(3).sum()),
               "one_missing_wearable_days": int(data.valid_wearable_count.eq(2).sum()),
               "missing_by_feature": data[WEARABLE_FEATURES].isna().sum().astype(int).to_dict()}
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (processed_dir / "README.md").write_text(dataset_readme(summary), encoding="utf-8")


def dataset_readme(summary):
    return f"""# Active analysis datasets: 14:00, one-day inputs, 2022 only

{summary['participants']} participants; {summary['days']} days; {summary['luteal_days']} luteal days;
{summary['luteal_starts']} dataset luteal starts. {summary['one_missing_wearable_days']} days need one wearable replacement.

Use observations_2022.csv for outcomes, features and timing/quality diagnostics.
The glucose_features_2022.csv, wearable_features_2022.csv and combined_features_2022.csv
files have identical keys and row order. Join outcomes_2022.csv by id, study_interval,
day_in_study. Only the 2/3/5 columns listed in model_inputs.FEATURE_SETS are predictors.
Never use keys, phase information, onset labels or missing-value flags as predictors.

Missing wearable values intentionally remain blank. Choose participant-disjoint
training, validation and test sets, then call model_inputs.replace_missing_wearables
on each training/evaluation pair. Its medians use only the training participants.
The resulting values must be shared between wearable-only and combined models.
Scaling must also be fitted on training data only. No split, imputation or model
has been fitted globally. Do not fill blanks with medians of this whole dataset.

Source: previous full calendar day's glucose and the selected main sleep ending
on the prediction day by 14:00. Glucose requires >=70% coverage and gaps <=4 hours;
at least two wearable summaries must pass the existing quality checks. No trends
or multi-day history are required. A sleep ending after 14:00 is excluded, not truncated.

Earlier audit/noon/three-day folders are historical, not inputs to this comparison.
Timing assumes aligned participant-local clocks and availability of completed
measurements by 14:00; actual publication/upload times remain unverified.
Luteal starts follow the dataset's phase information, not independently confirmed
biological dates. Alert-matching rules and participant splits remain to be fixed.
"""
