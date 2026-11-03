#### Evaluate either model family on the verified frozen three-day onset cohort.
#### Save all candidate validation probabilities, selected settings, fitted parameters and
#### one test probability per input set/key, followed by paired probability summaries.

from pathlib import Path

import joblib
import pandas as pd

from onset_prediction import protocol
from onset_prediction.freezing import check_results_destination, load_frozen
from onset_prediction.logistic import fit_logistic, parameter_rows, predict_logistic, tune_logistic
from onset_prediction.metrics import (INPUT_SETS, bootstrap_metrics, calibration_bins,
                                       probability_metrics, summary_tables, validate_paired)
from onset_prediction.report import write_report
from onset_prediction.workflow import write_json
from preprocessing import file_sha256


def model_components(model_family):
    #### Select the agreed model functions without changing their training or tuning rules.
    if model_family == "logistic_regression":
        return tune_logistic, fit_logistic, predict_logistic, parameter_rows, "C"
    if model_family == "random_forest":
        from onset_prediction.forest import fit_forest, parameter_rows as forest_rows, predict_forest, tune_forest
        return tune_forest, fit_forest, predict_forest, forest_rows, "min_samples_leaf"
    raise ValueError("Unknown model family")


def evaluate_one(development, testing, assignments, input_set, test_fold, model_directory,
                 model_family="logistic_regression"):
    #### Tune on validation participants, refit all development participants, then test once.
    tune, fit, predict, record_parameters, setting = model_components(model_family)
    chosen, tuning, validation, validation_parameters = tune(development, assignments, input_set)
    fitted = fit(development, input_set, chosen)
    probabilities = predict(fitted, testing)
    prediction = testing[protocol.KEY + [protocol.TARGET]].assign(probability=probabilities)
    parameters = record_parameters(fitted)
    selected = {setting: chosen, "mean_validation_AP": float(tuning.loc[tuning[setting].eq(chosen), "average_precision"].mean()),
                "training_participants": len(fitted.training_ids), "training_days": fitted.training_days,
                "test_participants": testing.id.nunique(), "test_days": len(testing)}
    if model_family == "logistic_regression":
        selected["iterations"] = int(fitted.pipeline.named_steps["logisticregression"].n_iter_[0])
    else:
        selected["n_estimators"] = len(fitted.estimator.estimators_)
        selected["maximum_fitted_depth"] = max(tree.tree_.max_depth for tree in fitted.estimator.estimators_)
    tags = {"model_family": model_family, "input_set": input_set, "test_fold": test_fold}
    joblib.dump(fitted, model_directory / f"test_fold_{test_fold}_{input_set}.joblib")
    return {"test_predictions": prediction.assign(**tags), "tuning_results": tuning.assign(**tags),
            "validation_predictions": validation.assign(**tags),
            "validation_parameters": validation_parameters.assign(**tags),
            "fitted_parameters": parameters.assign(**tags),
            "selected_settings": pd.DataFrame([{**selected, **tags}])}


def check_shared_replacements(parameters):
    #### Verify that shared wearable inputs use exactly the same medians in every common fit.
    group_columns = ["test_fold", "feature"]
    if "validation_fold" in parameters:
        setting = "C" if "C" in parameters else "min_samples_leaf"
        group_columns.extend(["validation_fold", setting])
    shared = parameters.loc[parameters.replacement_allowed]
    if shared.groupby(group_columns).training_median.nunique().gt(1).any():
        raise ValueError("Wearable-only and combined models have different shared replacement medians")


def fit_test_folds(data, test, validation, model_directory, model_family="logistic_regression"):
    #### Reuse saved participant assignments for all three input sets and collect fitted records.
    outputs = {}
    for fold in range(1, protocol.TEST_FOLDS + 1):
        ids = test.loc[test.test_fold.eq(fold), "id"]
        development, testing = data.loc[~data.id.isin(ids)], data.loc[data.id.isin(ids)]
        assignments = validation.loc[validation.test_fold.eq(fold), ["id", "validation_fold"]]
        for name in INPUT_SETS:
            tables = evaluate_one(development, testing, assignments, name, fold, model_directory, model_family)
            for key, frame in tables.items():
                outputs.setdefault(key, []).append(frame)
        print(f"Finished {model_family} test fold {fold}/{protocol.TEST_FOLDS}", flush=True)
    outputs = {name: pd.concat(parts, ignore_index=True) for name, parts in outputs.items()}
    for name in ("fitted_parameters", "validation_parameters"):
        check_shared_replacements(outputs[name])
    validate_paired(outputs["test_predictions"], data, test, model_family)
    return outputs


def probability_tables(predictions, model_family="logistic_regression"):
    #### Describe pooled calibration, fold performance and the distribution of probabilities.
    calibration, fold_rows, probability_rows = [], [], []
    for name in INPUT_SETS:
        data = predictions.loc[predictions.input_set.eq(name)]
        calibration.append(calibration_bins(data).assign(input_set=name, model_family=model_family))
        for fold, group in data.groupby("test_fold", sort=True):
            fold_rows.append({"input_set": name, "test_fold": fold, **probability_metrics(group)})
        probability_rows.append({"input_set": name, "participants": data.id.nunique(),
                                 **probability_metrics(data), "mean_probability": data.probability.mean(),
                                 "minimum_probability": data.probability.min(), "maximum_probability": data.probability.max()})
    return {"calibration_bins": pd.concat(calibration, ignore_index=True),
            "fold_metrics": pd.DataFrame(fold_rows), "probability_summary": pd.DataFrame(probability_rows)}


def run_logistic(frozen, output):
    #### Run logistic regression from a current, verified logistic freeze.
    return run_evaluation(frozen, output, "logistic_regression")


def run_forest(frozen, output):
    #### Run random forest from its own freeze with shared logistic cohort and participant draws.
    return run_evaluation(frozen, output, "random_forest")


def run_evaluation(frozen, output, model_family):
    #### Check the model-specific freeze and save a complete paired evaluation in fresh outputs.
    frozen, output = Path(frozen), Path(output)
    check_results_destination(output, (frozen,))
    if model_family == "random_forest":
        from onset_prediction.forest_freezing import load_forest_frozen
        data, test, validation, draws = load_forest_frozen(frozen)
    elif model_family == "logistic_regression":
        data, test, validation, draws = load_frozen(frozen)
    else:
        raise ValueError("Unknown model family")
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    models.mkdir()
    write_json(output / "run_status.json", {"status": "running", "model_family": model_family})
    write_json(output / "frozen_protocol_reference.json", {
        "directory": str(frozen.resolve()), "manifest_sha256": file_sha256(frozen / "freeze_manifest.json")})
    try:
        tables = fit_test_folds(data, test, validation, models, model_family)
        tables.update(probability_tables(tables["test_predictions"], model_family))
        print("Calculating 2,000 paired participant bootstrap resamples", flush=True)
        tables["bootstrap_metrics"] = bootstrap_metrics(tables["test_predictions"], draws)
        tables["model_summary"], tables["paired_comparison"] = summary_tables(
            tables["test_predictions"], tables["bootstrap_metrics"], model_family)
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        write_report(output, tables, model_family)
        paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "run_status.json")
        write_json(output / "output_manifest.json", {
            "files": {str(path.relative_to(output)): file_sha256(path) for path in paths}})
        write_json(output / "run_status.json", {"status": "complete", "model_family": model_family,
                   "participants": int(data.id.nunique()), "days": len(data), "models_fitted": True,
                   "test_predictions": len(tables["test_predictions"]), "bootstrap_resamples": len(draws)})
    except Exception as error:
        write_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__,
                   "message": str(error)})
        raise
    return output
