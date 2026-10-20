"""Orchestrate the fixed participant-held-out comparison and its saved outputs."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import preprocessing
import primary_protocol as protocol
from daily_metrics import bootstrap_metrics, daily_metrics, summary_tables, validate_paired_predictions
from daily_models import fit_and_predict, tune_logistic_model
from model_inputs import FEATURE_SETS, validate_features
from onset_alerts import evaluate_test_alerts, select_alert_rule
from participant_folds import build_fold_assignments


SOURCE_FILES = ["main.py", "primary_protocol.py", "primary_evaluation.py", "participant_folds.py",
                "daily_models.py", "daily_metrics.py", "onset_alerts.py", "model_inputs.py", "preprocessing.py",
                "requirements.txt"]
INPUT_FILES = ["observations_2022.csv", "prediction_alignment_14_2022.csv"]


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_evaluation_directory(path, processed_dir, report_dir, raw_dir=None):
    """Reject overwrites and overlap with source data or preparation outputs."""
    path = Path(path).resolve()
    default_raw = Path(__file__).resolve().parent / "data/raw"
    protected = [Path(processed_dir).resolve(), Path(report_dir).resolve(), default_raw.resolve()]
    if raw_dir is not None:
        protected.append(Path(raw_dir).resolve())
    if any(path == item or item in path.parents or path in item.parents for item in protected):
        raise ValueError("Evaluation output must not overlap raw data or prepared input/report directories")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Evaluation output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_primary_inputs(processed_dir, report_dir):
    """Check feature eligibility and agreement with the full supplied-label alignment."""
    processed_dir, report_dir = Path(processed_dir), Path(report_dir)
    configuration = json.loads((report_dir / "run_config.json").read_text(encoding="utf-8"))
    if not configuration.get("features_ready") or configuration.get("prediction_cutoff") != "14:00:00":
        raise ValueError("Prepared inputs do not match the fixed 14:00 feature protocol")
    observations = pd.read_csv(processed_dir / INPUT_FILES[0])
    validate_features(observations)
    alignment = pd.read_csv(processed_dir / INPUT_FILES[1])
    for key in protocol.KEY:
        preprocessing.validate_integer_column(alignment, key)
    if alignment.duplicated(protocol.KEY).any() or not alignment.study_interval.eq(2022).all():
        raise ValueError("Expected unique 2022 reference-label keys")
    outcomes = ["luteal_status", "exact_luteal_onset"]
    joined = observations[protocol.KEY + outcomes].merge(
        alignment[protocol.KEY + outcomes], on=protocol.KEY, how="left", validate="one_to_one",
        suffixes=("_observation", "_reference"), indicator=True,
    )
    if not joined._merge.eq("both").all() or any(
        not joined[f"{name}_observation"].eq(joined[f"{name}_reference"]).all() for name in outcomes
    ):
        raise ValueError("Eligible observation labels disagree with the full reference alignment")
    references = alignment.loc[
        alignment.id.isin(observations.id.unique()) & alignment.exact_luteal_onset.eq(1), protocol.KEY,
    ].sort_values(protocol.KEY).reset_index(drop=True)
    if references.empty:
        raise ValueError("No reference luteal starts are available for onset evaluation")
    return observations.sort_values(protocol.KEY).reset_index(drop=True), references, configuration


def software_versions():
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl", "narwhals", "cloudpickle"]
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def protocol_record():
    """State the precise objective and all choices before generating test predictions."""
    return {
        "version": protocol.PRIMARY_PROTOCOL_VERSION, "study_interval": 2022,
        "seed": protocol.PRIMARY_SEED,
        "validation_seeds": [protocol.PRIMARY_SEED + k for k in range(1, protocol.OUTER_FOLDS + 1)],
        "outer_folds": protocol.OUTER_FOLDS, "inner_folds": protocol.INNER_FOLDS,
        "logistic_inverse_penalty_strengths": protocol.LOGISTIC_C_VALUES,
        "feature_sets": FEATURE_SETS, "observation_key": protocol.KEY,
        "prediction_cutoff": "14:00:00",
        "wearable_replacement": "feature_median_from_training_participants_only",
        "logistic_objective": "mean_binary_log_loss + sum(non_intercept_coefficients_squared) / (2 * C)",
        "model_implementation": "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)",
        "model_solver": "lbfgs", "class_weight": None,
        "sklearn_c_conversion": "protocol_C / number_of_training_observations",
        "model_max_iterations": protocol.LOGISTIC_MAX_ITERATIONS, "model_tolerance": protocol.LOGISTIC_TOLERANCE,
        "primary_metric": "pooled_out_of_fold_average_precision", "secondary_daily_metric": "pooled_out_of_fold_roc_auc",
        "primary_comparison": "combined_minus_wearable_average_precision",
        "model_selection": "highest_mean_validation_AP_then_smaller_C",
        "alert_thresholds": protocol.ALERT_THRESHOLDS, "alert_consecutive_days": protocol.ALERT_CONSECUTIVE_DAYS,
        "alert_selection": "highest_mean_validation_event_F1_then_higher_threshold_then_two_days",
        "alert_reset": "only_next_available_prediction_below_threshold; missing_day_breaks_pending_streak",
        "alert_match_window_days": protocol.ALERT_MATCH_WINDOW_DAYS,
        "event_matching": "greedy_smallest_absolute_error_then_earlier_alert_then_earlier_reference; one_to_one",
        "minimum_evaluable_starts_per_fold": 1,
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES, "bootstrap_seed": protocol.BOOTSTRAP_SEED,
        "bootstrap_interval": "95_percentile_participant_resampling",
        "bootstrap_scope": "conditional on the fitted cross-validation models",
        "undefined_bootstrap_metrics": "retain draw with NaN; use finite replicates and report their count",
        "fold_allocation": f"seeded_capacity_balanced_greedy_then_up_to_{protocol.MAX_FOLD_SWAP_PASSES}_swap_passes",
        "software_versions": software_versions(),
    }


def provenance_record(processed_dir, report_dir):
    root = Path(__file__).resolve().parent
    return {
        "inputs": {name: preprocessing.file_sha256(processed_dir / name) for name in INPUT_FILES},
        "preparation_config_sha256": preprocessing.file_sha256(report_dir / "run_config.json"),
        "sources": {name: preprocessing.file_sha256(root / name) for name in SOURCE_FILES},
    }


def parameter_rows(test_fold, input_set, parameters):
    rows = []
    for category, values in parameters.items():
        values = values if isinstance(values, dict) else {category: values}
        rows.extend({"outer_test_fold": test_fold, "input_set": input_set,
                     "parameter_category": category, "parameter": name, "value": value}
                    for name, value in values.items())
    return pd.DataFrame(rows)


def evaluate_input_set(development, testing, references, validation_assignments, input_set, test_fold):
    """Choose settings using development participants, then refit and test once."""
    features = FEATURE_SETS[input_set]
    selected_c, tuning, validation = tune_logistic_model(development, validation_assignments, features)
    threshold, consecutive, alert_tuning = select_alert_rule(validation, references)
    probabilities, parameters = fit_and_predict(development, testing, features, selected_c)
    tags = {"input_set": input_set, "outer_test_fold": test_fold}
    prediction = testing[protocol.KEY + ["luteal_status"]].assign(probability=probabilities, **tags)
    return {
        "model_predictions": prediction,
        "tuning_results": tuning.assign(**tags),
        "validation_predictions": validation.assign(inverse_penalty_strength=selected_c, **tags),
        "alert_tuning_results": alert_tuning.assign(**tags),
        "selected_settings": pd.DataFrame([{**tags, "inverse_penalty_strength": selected_c,
                                             "threshold": threshold, "consecutive_days": consecutive}]),
        "fitted_parameters": parameter_rows(test_fold, input_set, parameters),
    }


def evaluate_participant_folds(observations, references, test_folds, validation_folds):
    parts = {}
    for test_fold in range(1, protocol.OUTER_FOLDS + 1):
        test_ids = test_folds.loc[test_folds.outer_test_fold.eq(test_fold), "id"]
        development = observations.loc[~observations.id.isin(test_ids)].copy()
        testing = observations.loc[observations.id.isin(test_ids)].copy()
        assignments = validation_folds.loc[
            validation_folds.outer_test_fold.eq(test_fold), ["id", "validation_fold"],
        ]
        development_references = references.loc[references.id.isin(development.id)]
        for input_set in FEATURE_SETS:
            result = evaluate_input_set(development, testing, development_references, assignments, input_set, test_fold)
            for name, frame in result.items():
                parts.setdefault(name, []).append(frame)
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def add_evaluation_tables(tables, observations, references, test_folds):
    predictions = tables["model_predictions"].sort_values(["input_set"] + protocol.KEY).reset_index(drop=True)
    validate_paired_predictions(predictions, observations)
    assigned = predictions.id.map(test_folds.set_index("id").outer_test_fold)
    if not assigned.eq(predictions.outer_test_fold).all():
        raise ValueError("Held-out predictions do not match the saved participant test folds")
    tables["model_predictions"] = predictions
    alerts, reference_events, alert_events, event_metrics = evaluate_test_alerts(
        predictions, references, tables["selected_settings"],
    )
    tables.update(onset_alerts=alerts, onset_reference_events=reference_events,
                  onset_alert_events=alert_events, onset_fold_metrics=event_metrics)
    fold_rows = []
    for (input_set, test_fold), group in predictions.groupby(["input_set", "outer_test_fold"]):
        event = event_metrics.loc[(event_metrics.input_set == input_set) &
                                  (event_metrics.outer_test_fold == test_fold)].iloc[0].to_dict()
        fold_rows.append({**daily_metrics(group), **event})
    tables["model_fold_metrics"] = pd.DataFrame(fold_rows)
    tables["bootstrap_metrics"] = bootstrap_metrics(predictions)
    tables["model_summary"], tables["paired_comparison"] = summary_tables(predictions, tables["bootstrap_metrics"])
    tables["onset_summary"] = onset_summary(event_metrics, reference_events)
    return tables


def onset_summary(fold_metrics, reference_events):
    rows = []
    for input_set, group in fold_metrics.groupby("input_set", sort=True):
        matched = reference_events.loc[reference_events.input_set.eq(input_set) & reference_events.matched]
        counts = group[["evaluable_reference_starts", "matched_starts", "missed_evaluable_starts", "unmatched_alerts"]].sum()
        denominator = 2 * counts.matched_starts + counts.missed_evaluable_starts + counts.unmatched_alerts
        rows.append({"input_set": input_set, **counts.astype(int).to_dict(),
                     "event_f1": float(2 * counts.matched_starts / denominator) if denominator else 0.0,
                     "timing_mae_days": matched.timing_error_days.mean()})
    return pd.DataFrame(rows)


def save_tables(output_dir, tables):
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "README.md").write_text(
        "# Primary participant-held-out evaluation\n\n"
        "Daily estimates use pooled out-of-fold predictions on identical observation keys. "
        "Fold metrics provide a stability check. The primary effect is combined minus wearable-only AP.\n\n"
        "Percentile intervals resample participants and are conditional on the fitted cross-validation models. "
        "Undefined metrics remain missing in bootstrap_metrics.csv; summary tables report valid replicate counts.\n\n"
        "Onset counts and timing errors assess agreement with the supplied dataset labels. "
        "The labels are not independently hormone-validated ground truth. "
        "This retrospective analysis assumes that completed measurements were available by 14:00; "
        "actual upload times are unavailable.\n", encoding="utf-8",
    )


def run_primary_evaluation(processed_dir, report_dir, evaluation_dir, *, raw_dir=None):
    """Persist the protocol and feasible folds before making any test prediction."""
    processed_dir, report_dir = Path(processed_dir).resolve(), Path(report_dir).resolve()
    output_dir = create_evaluation_directory(evaluation_dir, processed_dir, report_dir, raw_dir)
    try:
        observations, references, preparation = load_primary_inputs(processed_dir, report_dir)
        write_json(output_dir / "evaluation_protocol.json", protocol_record())
        write_json(output_dir / "provenance.json", provenance_record(processed_dir, report_dir))
        test_folds, validation_folds, feasibility = build_fold_assignments(observations, references)
        save_tables(output_dir, {"participant_folds": test_folds, "validation_fold_assignments": validation_folds,
                                 "fold_feasibility": feasibility})
        write_json(output_dir / "run_status.json", {"status": "running"})
        tables = evaluate_participant_folds(observations, references, test_folds, validation_folds)
        tables = add_evaluation_tables(tables, observations, references, test_folds)
        save_tables(output_dir, tables)
        write_json(output_dir / "run_config.json", {
            "protocol_version": protocol.PRIMARY_PROTOCOL_VERSION, "prepared_input_config": preparation,
            "processed_dir": str(processed_dir), "report_dir": str(report_dir), "evaluation_dir": str(output_dir),
            "participants": int(observations.id.nunique()), "eligible_days": len(observations),
            "eligible_observation_starts": int(observations.exact_luteal_onset.eq(1).sum()),
            "reference_starts_in_full_alignment": len(references), "software_versions": software_versions(),
            "model_ready": True,
        })
        write_json(output_dir / "run_status.json", {"status": "complete"})
    except Exception as error:
        write_json(output_dir / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
    return {"participants": int(observations.id.nunique()), "days": len(observations), "output_dir": output_dir}
