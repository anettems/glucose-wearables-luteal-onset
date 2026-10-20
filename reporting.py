"""Generate descriptive tables and figures without fitting or selecting models."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_metrics import summary_tables, validate_paired_predictions
from model_inputs import FEATURE_SETS
from onset_alerts import evaluate_test_alerts
import preprocessing
import primary_evaluation as evaluation
import primary_protocol as protocol
from reporting_tables import calibration_tables, descriptive_tables, matched_timing_table


ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = "2026-10-20-reporting-v1"
REPORT_SOURCES = ["reporting.py", "reporting_tables.py", "reporting_figures.py", "requirements-reporting.txt"]
EVALUATION_TABLES = ["model_predictions", "model_summary", "paired_comparison", "model_fold_metrics",
                     "selected_settings", "bootstrap_metrics", "onset_summary", "onset_fold_metrics",
                     "onset_reference_events", "onset_alert_events"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_same_table(actual, expected, sort_columns):
    try:
        pd.testing.assert_frame_equal(
            actual[expected.columns].sort_values(sort_columns).reset_index(drop=True),
            expected.sort_values(sort_columns).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError("Saved evaluation tables disagree with predictions, events or frozen assignments") from error


def verify_prediction_folds(predictions, folds):
    expected = predictions[protocol.KEY + ["outer_test_fold"]].merge(
        folds[["id", "outer_test_fold"]], on="id", how="left", validate="many_to_one", suffixes=("", "_frozen"),
    )
    if not expected.outer_test_fold.eq(expected.outer_test_fold_frozen).all():
        raise ValueError("Prediction folds differ from the frozen participant assignments")


def verify_event_tables(tables, references):
    settings = tables["selected_settings"]
    expected_pairs = {(name, fold) for name in FEATURE_SETS for fold in range(1, protocol.OUTER_FOLDS + 1)}
    pairs = list(zip(settings.input_set, settings.outer_test_fold))
    if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
        raise ValueError("Expected one selected setting per model and test fold")
    if (not settings.threshold.isin(protocol.ALERT_THRESHOLDS).all()
            or not settings.consecutive_days.isin(protocol.ALERT_CONSECUTIVE_DAYS).all()
            or not settings.inverse_penalty_strength.isin(protocol.LOGISTIC_C_VALUES).all()):
        raise ValueError("Selected settings are outside the frozen tuning grids")
    _, events, alerts, metrics = evaluate_test_alerts(tables["model_predictions"], references, settings)
    require_same_table(tables["onset_reference_events"], events, ["input_set", "id", "reference_day_in_study"])
    require_same_table(tables["onset_alert_events"], alerts, ["input_set", "id", "alert_day_in_study"])
    require_same_table(tables["onset_fold_metrics"], metrics, ["input_set", "outer_test_fold"])
    require_same_table(tables["onset_summary"], evaluation.onset_summary(metrics, events), ["input_set"])


def load_completed_evaluation(path, frozen, observations, references, folds):
    """Reject incomplete runs and check plots against their underlying saved evidence."""
    path, frozen = Path(path).resolve(), Path(frozen).resolve()
    if read_json(path / "run_status.json").get("status") != "complete":
        raise ValueError("Model reporting requires a completed evaluation")
    reference = read_json(path / "frozen_protocol_reference.json")
    if reference.get("manifest_sha256") != preprocessing.file_sha256(frozen / "freeze_manifest.json"):
        raise ValueError("Evaluation does not refer to this frozen protocol manifest")
    for name in ["evaluation_protocol.json", "provenance.json", "software_versions.json"]:
        if read_json(path / name) != read_json(frozen / name):
            raise ValueError(f"Evaluation evidence differs from the freeze: {name}")
    require_same_table(pd.read_csv(path / "participant_folds.csv"), folds, ["id"])
    tables = {name: pd.read_csv(path / f"{name}.csv") for name in EVALUATION_TABLES}
    predictions = tables["model_predictions"]
    validate_paired_predictions(predictions, observations)
    verify_prediction_folds(predictions, folds)
    summary, paired = summary_tables(predictions, tables["bootstrap_metrics"])
    require_same_table(tables["model_summary"], summary, ["input_set", "metric"])
    require_same_table(tables["paired_comparison"], paired, ["comparison", "metric"])
    verify_event_tables(tables, references)
    expected_fold_rows = []
    for (name, fold), group in predictions.groupby(["input_set", "outer_test_fold"]):
        events = tables["onset_fold_metrics"].query("input_set == @name and outer_test_fold == @fold").iloc[0].to_dict()
        expected_fold_rows.append({**evaluation.daily_metrics(group), **events})
    require_same_table(tables["model_fold_metrics"], pd.DataFrame(expected_fold_rows), ["input_set", "outer_test_fold"])
    return tables


def create_report_directory(output, processed, preparation, frozen, completed):
    path = Path(output).resolve()
    protected = [Path(frozen).resolve()]
    if completed is not None:
        protected.append(Path(completed).resolve())
    if any(path == item or item in path.parents or path in item.parents for item in protected):
        raise ValueError("Reporting output must not overlap evaluation or frozen protocol directories")
    return evaluation.create_evaluation_directory(path, processed, preparation)


def reporting_versions():
    packages = ["matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",
                "pillow", "packaging", "python-dateutil", "six"]
    return {**evaluation.software_versions(), **{name: version(name) for name in packages}}


def report_manifest(processed, preparation, frozen, completed):
    inputs = [processed / name for name in evaluation.INPUT_FILES]
    inputs.extend([preparation / "run_config.json", frozen / "freeze_manifest.json"])
    inputs.extend(frozen / name for name in evaluation.FREEZE_FILES)
    if completed is not None:
        names = [f"{name}.csv" for name in EVALUATION_TABLES]
        names += ["run_status.json", "frozen_protocol_reference.json", "participant_folds.csv",
                  "evaluation_protocol.json", "provenance.json", "software_versions.json"]
        inputs.extend(completed / name for name in names)
    return {"report_version": REPORT_VERSION, "protocol_version": protocol.PRIMARY_PROTOCOL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "descriptive_only" if completed is None else "completed_evaluation",
            "calibration_bins": "10 equal-width bins; [lower,upper), final bin includes 1",
            "calibration_scope": "descriptive pooled held-out predictions; no recalibration or test tuning",
            "feature_scope": "shared eligible days, observed values before replacement",
            "software_versions": reporting_versions(),
            "sources": {name: preprocessing.file_sha256(Path(__file__).parent / name) for name in REPORT_SOURCES},
            "inputs": {str(path.resolve()): preprocessing.file_sha256(path) for path in inputs}}


def save_report_tables(output, tables):
    # Predictions and all bootstrap draws already have authoritative files in the evaluation.
    for name, frame in tables.items():
        if name not in {"model_predictions", "bootstrap_metrics"}:
            frame.to_csv(output / f"{name}.csv", index=False)


def write_report_index(output, observations, tables, completed):
    lines = ["# 2022 descriptive and model report", "",
             f"Shared eligible cohort: {observations.id.nunique()} participants, {len(observations):,} days.", "",
             "## Descriptive outputs", "",
             "Coverage includes all participants in the prepared alignment, including those with no eligible days. "
             "The denominator is aligned days, not an assumed recording duration. Gaps are measured in calendar days.", "",
             "Feature distributions and missingness describe observed values in the shared eligible cohort before "
             "training-only replacement. Eligibility already requires complete glucose and at least two wearable features; "
             "these percentages do not describe missingness in the source population. See the preparation audit for exclusions.", ""]
    if completed is None:
        lines += ["## Model outputs pending", "", "No model results were loaded and no models were fitted. "
                  "Run this command with --evaluation-dir after the frozen primary evaluation has completed.", ""]
    else:
        lines += ["## Held-out performance and onset assessment", "", f"Evaluation source: `{completed}`.", "",
                  "Pooled AP is primary; ROC-AUC is complementary and fold results assess stability. "
                  "95% participant-bootstrap intervals are conditional on the fitted cross-validation models.", "",
                  "Calibration uses ten fixed probability bins, with counts and observed fractions, plus the Brier "
                  "score (mean squared probability error; lower is better). Empty bins remain missing. "
                  "These are descriptive, day-weighted summaries of correlated participant days. No recalibration "
                  "or test-based model or alert selection is performed.", "",
                  "Onset counts report matched starts, missed evaluable starts and unmatched alerts together with "
                  "mean absolute timing error. Timing is undefined when there are no matches. Signed timing is "
                  "supplementary: negative means an earlier alert. All reference and alert event rows are retained.", ""]
        for row in tables["onset_summary"].itertuples():
            mae = f"{row.timing_mae_days:.2f} days" if np.isfinite(row.timing_mae_days) else "undefined"
            lines.append(f"- {row.input_set}: {row.matched_starts} matched, {row.missed_evaluable_starts} missed evaluable, "
                         f"{row.unmatched_alerts} unmatched alerts; mean absolute timing error {mae}.")
        lines += [""]
    lines += ["## Interpretation", "", "These analyses assess agreement with supplied phase labels, not independent "
              "hormonal validation or clinical performance. Availability by 14:00 is assumed; actual upload times are unavailable.", "",
              "## Tables", ""]
    lines.extend(f"- [{path.name}]({path.name})" for path in sorted(output.glob("*.csv")))
    lines += ["", "## Figures", "", "Each figure is also saved as a vector PDF for thesis use.", ""]
    for path in sorted(output.glob("*.png")):
        lines += [f"### {path.stem.replace('_', ' ').capitalize()}", "", f"![{path.stem}]({path.name})", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_report(processed, preparation, frozen, output, *, completed=None):
    processed, preparation, frozen = (Path(path).resolve() for path in [processed, preparation, frozen])
    completed = Path(completed).resolve() if completed is not None else None
    observations, references, _ = evaluation.load_primary_inputs(processed, preparation)
    folds, _, _ = evaluation.load_frozen_protocol(frozen, processed, preparation, observations, references)
    alignment = pd.read_csv(processed / "prediction_alignment_14_2022.csv")
    tables = descriptive_tables(observations, alignment)
    if completed is not None:
        tables.update(load_completed_evaluation(completed, frozen, observations, references, folds))
        tables.update(calibration_tables(tables["model_predictions"]))
        tables["matched_timing"] = matched_timing_table(tables["onset_reference_events"])
    manifest = report_manifest(processed, preparation, frozen, completed)
    output = create_report_directory(output, processed, preparation, frozen, completed)
    try:
        evaluation.write_json(output / "run_status.json", {"status": "running"})
        from reporting_figures import descriptive_figures, model_figures
        save_report_tables(output, tables)
        descriptive_figures(observations, alignment, tables, output)
        if completed is not None:
            model_figures(tables["model_predictions"], tables, output)
        write_report_index(output, observations, tables, completed)
        manifest["outputs"] = {p.name: preprocessing.file_sha256(p) for p in sorted(output.iterdir()) if p.name != "run_status.json"}
        evaluation.write_json(output / "report_manifest.json", manifest)
        evaluation.write_json(output / "run_status.json", {"status": "complete"})
    except Exception as error:
        evaluation.write_json(output / "run_status.json", {"status": "failed", "message": str(error)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "code/data/processed/primary_1400_2022")
    parser.add_argument("--preparation-dir", type=Path, default=ROOT / "results/primary_1400_2022")
    parser.add_argument("--frozen-protocol-dir", type=Path, default=ROOT / "results/primary_protocol_2022_v4")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--descriptive-only", action="store_true")
    mode.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory outside all input directories")
    arguments = parser.parse_args()
    print(generate_report(arguments.processed_dir, arguments.preparation_dir, arguments.frozen_protocol_dir,
                          arguments.output_dir, completed=arguments.evaluation_dir))


if __name__ == "__main__":
    main()
