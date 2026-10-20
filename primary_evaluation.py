"""Orchestrate the fixed participant-held-out comparison and its saved outputs."""

import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import preprocessing
import primary_protocol as protocol
from daily_metrics import bootstrap_metrics, daily_metrics, summary_tables, validate_paired_predictions
from daily_models import fit_and_predict, tune_logistic_model
from model_inputs import FEATURE_SETS, validate_features
from onset_alerts import evaluate_test_alerts, reference_evaluability, select_alert_rule
from participant_folds import build_fold_assignments, validate_fold_assignment


SOURCE_FILES = ["main.py", "primary_protocol.py", "primary_evaluation.py", "participant_folds.py",
                "daily_models.py", "daily_metrics.py", "onset_alerts.py", "model_inputs.py", "preprocessing.py",
                "requirements.txt"]
INPUT_FILES = ["observations_2022.csv", "prediction_alignment_14_2022.csv"]
FREEZE_FILES = ["evaluation_protocol.json", "software_versions.json", "provenance.json",
                "participant_folds.csv", "validation_fold_assignments.csv", "fold_feasibility.csv",
                "cohort_summary.json", "methods_snapshot.tex", "methods_consistency.md", "README.md"]


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
        "scaling": "training_only_feature_mean_and_population_standard_deviation; constant_scale_1",
        "reference_outcome": "supplied_phase_label_luteal_versus_other_recognised_phases",
        "reference_start": "observed_non_luteal_to_luteal_on_consecutive_calendar_days_in_full_alignment",
        "logistic_objective": "sum_binary_log_loss + sum(non_intercept_coefficients_squared) / (2 * C)",
        "model_implementation": "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)",
        "model_solver": "lbfgs", "class_weight": None,
        "c_convention": "scikit_learn_standard",
        "sklearn_c_conversion": "identity: sklearn_C = protocol_C",
        "model_max_iterations": protocol.LOGISTIC_MAX_ITERATIONS, "model_tolerance": protocol.LOGISTIC_TOLERANCE,
        "primary_metric": "pooled_out_of_fold_average_precision", "secondary_daily_metric": "pooled_out_of_fold_roc_auc",
        "primary_comparison": "combined_minus_wearable_average_precision",
        "model_selection": "highest_mean_validation_AP_then_smaller_C",
        "alert_thresholds": protocol.ALERT_THRESHOLDS, "alert_consecutive_days": protocol.ALERT_CONSECUTIVE_DAYS,
        "positive_day": "probability_greater_than_or_equal_to_threshold",
        "alert_validation_predictions": "validation_predictions_for_the_validation_selected_C",
        "event_f1": "2 * matched / (2 * matched + missed_evaluable + unmatched_alerts); zero_if_denominator_zero",
        "alert_selection": "highest_mean_validation_event_F1_then_higher_threshold_then_two_days",
        "alert_reset": "only_next_available_prediction_below_threshold; missing_day_breaks_pending_streak",
        "alert_match_window_days": protocol.ALERT_MATCH_WINDOW_DAYS,
        "event_matching": "greedy_smallest_absolute_error_then_earlier_alert_then_earlier_reference; one_to_one",
        "minimum_evaluable_starts_per_fold": 1,
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES, "bootstrap_seed": protocol.BOOTSTRAP_SEED,
        "bootstrap_interval": "95_percentile_participant_resampling",
        "bootstrap_quantiles": [0.025, 0.975], "bootstrap_quantile_method": "linear",
        "bootstrap_scope": "conditional on the fitted cross-validation models",
        "undefined_bootstrap_metrics": "retain draw with NaN; use finite replicates and report their count",
        "fold_allocation": f"seeded_capacity_balanced_greedy_then_up_to_{protocol.MAX_FOLD_SWAP_PASSES}_swap_passes",
        "software_versions": software_versions(),
    }


def methods_snapshot(thesis_path):
    text = Path(thesis_path).read_text(encoding="utf-8")
    start = text.index(r"\section{Research material and methods}")
    end = text.index(r"\clearpage", start)
    return text[start:end].rstrip() + "\n"


def freeze_primary_protocol(processed_dir, report_dir, freeze_dir, *, raw_dir=None, thesis_path=None):
    """Save a complete protocol and feasible assignments without fitting any model."""
    processed_dir, report_dir = Path(processed_dir).resolve(), Path(report_dir).resolve()
    output = create_evaluation_directory(freeze_dir, processed_dir, report_dir, raw_dir)
    thesis_path = Path(thesis_path or Path(__file__).resolve().parents[1] / "latex/thesis.tex")
    observations, references, _ = load_primary_inputs(processed_dir, report_dir)
    test, validation, feasibility = build_fold_assignments(observations, references)
    write_json(output / "evaluation_protocol.json", protocol_record())
    write_json(output / "software_versions.json", software_versions())
    provenance = {**provenance_record(processed_dir, report_dir),
                  "thesis_source_sha256": preprocessing.file_sha256(thesis_path)}
    write_json(output / "provenance.json", provenance)
    for name, frame in [("participant_folds", test), ("validation_fold_assignments", validation),
                        ("fold_feasibility", feasibility)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "cohort_summary.json", {
        "study_interval": 2022, "participants": int(observations.id.nunique()), "eligible_days": len(observations),
        "luteal_days": int(observations.luteal_status.sum()),
        "eligible_day_exact_starts": int(observations.exact_luteal_onset.eq(1).sum()),
        "full_alignment_exact_starts": len(references),
        "evaluable_exact_starts": int(reference_evaluability(observations, references).evaluable.sum()),
    })
    (output / "methods_snapshot.tex").write_text(methods_snapshot(thesis_path), encoding="utf-8")
    shutil.copyfile(Path(__file__).parent / "PROTOCOL_REVIEW.md", output / "methods_consistency.md")
    (output / "README.md").write_text(
        "# Frozen primary evaluation protocol v4\n\n"
        "This directory contains the final protocol, methods snapshot, software versions, "
        "input/source hashes and saved participant assignments. No model was fitted by the freeze command.\n\n"
        "Evaluation must load these assignments with --frozen-protocol-dir. "
        "The loader checks every saved file, current protocol, software, input and analysis-source hash. "
        "Changed inputs or settings require a new documented freeze in a fresh directory.\n",
        encoding="utf-8",
    )
    write_json(output / "freeze_manifest.json", {
        "status": "frozen", "protocol_version": protocol.PRIMARY_PROTOCOL_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "predictions_generated": False,
        "files": {name: preprocessing.file_sha256(output / name) for name in FREEZE_FILES},
    })
    return {"participants": int(observations.id.nunique()), "days": len(observations), "output_dir": output}


def validate_saved_assignments(observations, references, test, validation):
    counts = [validate_fold_assignment(observations, references, test, "outer_test_fold",
                                      protocol.OUTER_FOLDS).assign(stage="outer")]
    if set(validation.outer_test_fold) != set(range(1, protocol.OUTER_FOLDS + 1)):
        raise ValueError("Frozen validation assignments do not cover all test folds")
    for test_fold in range(1, protocol.OUTER_FOLDS + 1):
        ids = test.loc[test.outer_test_fold.ne(test_fold), "id"]
        development = observations.loc[observations.id.isin(ids)]
        starts = references.loc[references.id.isin(ids)]
        assignments = validation.loc[validation.outer_test_fold.eq(test_fold), ["id", "validation_fold"]]
        counted = validate_fold_assignment(development, starts, assignments, "validation_fold", protocol.INNER_FOLDS)
        counts.append(counted.assign(stage="validation", outer_test_fold=test_fold))
    return pd.concat(counts, ignore_index=True)


def load_frozen_protocol(freeze_dir, processed_dir, report_dir, observations, references):
    """Verify the freeze and return saved folds; never create a new allocation."""
    freeze_dir = Path(freeze_dir).resolve()
    manifest = json.loads((freeze_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "frozen" or manifest.get("protocol_version") != protocol.PRIMARY_PROTOCOL_VERSION
            or manifest.get("predictions_generated") is not False
            or set(manifest.get("files", {})) != set(FREEZE_FILES)):
        raise ValueError("Incomplete frozen protocol manifest")
    for name in FREEZE_FILES:
        if preprocessing.file_sha256(freeze_dir / name) != manifest["files"][name]:
            raise ValueError(f"Frozen protocol file changed: {name}")
    saved = json.loads((freeze_dir / "evaluation_protocol.json").read_text(encoding="utf-8"))
    if json.dumps(saved, sort_keys=True) != json.dumps(protocol_record(), sort_keys=True):
        raise ValueError("Current protocol or software versions differ from the frozen protocol")
    provenance = json.loads((freeze_dir / "provenance.json").read_text(encoding="utf-8"))
    if any(provenance.get(key) != value for key, value in provenance_record(processed_dir, report_dir).items()):
        raise ValueError("Current inputs or analysis sources differ from the frozen protocol")
    test = pd.read_csv(freeze_dir / "participant_folds.csv")
    validation = pd.read_csv(freeze_dir / "validation_fold_assignments.csv")
    feasibility = pd.read_csv(freeze_dir / "fold_feasibility.csv")
    actual = validate_saved_assignments(observations, references, test, validation)
    try:
        pd.testing.assert_frame_equal(actual, feasibility, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Frozen fold counts disagree with the current cohort") from error
    return test, validation, feasibility


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


def run_primary_evaluation(processed_dir, report_dir, evaluation_dir, *, frozen_protocol_dir, raw_dir=None):
    """Use the verified frozen protocol and assignments to generate test predictions."""
    processed_dir, report_dir = Path(processed_dir).resolve(), Path(report_dir).resolve()
    output_dir = create_evaluation_directory(evaluation_dir, processed_dir, report_dir, raw_dir)
    try:
        observations, references, preparation = load_primary_inputs(processed_dir, report_dir)
        test_folds, validation_folds, feasibility = load_frozen_protocol(
            frozen_protocol_dir, processed_dir, report_dir, observations, references,
        )
        for name in ["evaluation_protocol.json", "provenance.json", "software_versions.json"]:
            shutil.copyfile(Path(frozen_protocol_dir) / name, output_dir / name)
        write_json(output_dir / "frozen_protocol_reference.json", {
            "directory": str(Path(frozen_protocol_dir).resolve()),
            "manifest_sha256": preprocessing.file_sha256(Path(frozen_protocol_dir) / "freeze_manifest.json"),
        })
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
