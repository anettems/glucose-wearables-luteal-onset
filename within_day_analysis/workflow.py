"""Freeze new feature definitions and paired cohorts before evaluating models."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as prep
import primary_evaluation as primary
from model_inputs import validate_features
from trend_analysis import workflow as history_workflow
from trend_analysis.features import coverage_record
from within_day_analysis import protocol
from within_day_analysis.features import daily_patterns, select_cohort
from within_day_analysis.evaluation import evaluate_models, event_tables, bootstrap_tables
from within_day_analysis.report import write_report


PROCESSED_FILES = ["daily_patterns.csv", "cohort_audit.csv", "observations.csv", "references.csv"]
FREEZE_TABLES = ["participant_folds", "validation_fold_assignments", "fold_feasibility", "coverage_summary",
                 "feature_availability", "feature_training_feasibility", "participant_coverage"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    folder = Path(__file__).parent
    sources = sorted(folder.rglob("*.py")) + [folder / "README.md"]
    hashes = {str(p.relative_to(folder)): prep.file_sha256(p) for p in sources}
    hashes.update({f"reused/{name}": digest for name, digest in history_workflow.source_hashes().items()})
    return hashes


def input_hashes(baseline_results, glucose_path):
    manifest_path = baseline_results / "freeze_manifest.json"
    manifest = read_json(manifest_path)
    if str(glucose_path.resolve()) not in manifest["inputs"]:
        raise ValueError("CGM path is not an input of the frozen baseline")
    paths = [manifest_path, *map(Path, manifest["inputs"]), *map(Path, manifest["files"])]
    return {str(p.resolve()): prep.file_sha256(p) for p in paths}


def training_feasibility(data, test, validation):
    # Reuse the checks for existing raw, deviation and slope predictors.
    base = history_workflow.training_feasibility(data, test, validation)
    rows = []
    for fold in sorted(test.outer_test_fold.unique()):
        development = data.loc[~data.id.isin(test.loc[test.outer_test_fold.eq(fold), "id"])]
        cohorts = [("development", 0, development)]
        for inner in sorted(validation.validation_fold.unique()):
            ids = validation.loc[validation.outer_test_fold.eq(fold) & validation.validation_fold.eq(inner), "id"]
            cohorts.append(("validation_training", inner, development.loc[~development.id.isin(ids)]))
        for stage, inner, training in cohorts:
            for feature in protocol.PATTERN_FEATURES:
                observed = int(np.isfinite(training[feature]).sum())
                if observed != len(training) or not observed:
                    raise ValueError(f"Missing within-day glucose training values: {feature}")
                rows.append({"outer_test_fold": fold, "validation_fold": inner, "stage": stage,
                             "feature": feature, "observed_training_days": observed})
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def preparation_tables(baseline, audit, cohort, references, test, validation):
    feasibility = primary.validate_saved_assignments(cohort, references, test, validation)
    availability = [{"feature": feature, "baseline_days": len(audit),
                     "available_days": int(audit[feature].notna().sum()),
                     "unavailable_days": int(audit[feature].isna().sum())} for feature in protocol.PATTERN_FEATURES]
    coverage = [coverage_record(baseline, references, "frozen_history_cohort"),
                coverage_record(cohort, references, "shared_within_day_cohort")]
    participants = audit.groupby("id").agg(baseline_days=("id", "size"), retained_days=("within_day_eligible", "sum"))
    return {"participant_folds": test, "validation_fold_assignments": validation,
            "fold_feasibility": feasibility, "coverage_summary": pd.DataFrame(coverage),
            "feature_availability": pd.DataFrame(availability), "participant_coverage": participants.reset_index(),
            "feature_training_feasibility": training_feasibility(cohort, test, validation)}


def prepare(baseline_processed, baseline_results, glucose_path, processed_output, results):
    paths = list(map(Path, [baseline_processed, baseline_results, glucose_path, processed_output, results]))
    baseline_processed, baseline_results, glucose_path, processed_output, results = paths
    history_workflow.check_destinations(processed_output, results, paths[:3])
    baseline, references, test, validation = history_workflow.load_frozen(baseline_processed, baseline_results)
    inputs, sources = input_hashes(baseline_results, glucose_path), source_hashes()
    patterns = daily_patterns(pd.read_csv(glucose_path))
    audit, cohort = select_cohort(baseline, patterns)
    validate_features(cohort)
    test = test.loc[test.id.isin(cohort.id)].reset_index(drop=True)
    validation = validation.loc[validation.id.isin(cohort.id)].reset_index(drop=True)
    tables = preparation_tables(baseline, audit, cohort, references, test, validation)
    if inputs != input_hashes(baseline_results, glucose_path) or sources != source_hashes():
        raise ValueError("Sources or inputs changed during within-day preparation")
    processed_output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(PROCESSED_FILES, [patterns, audit, cohort, references]):
        frame.to_csv(processed_output / name, index=False)
    for name, frame in tables.items():
        frame.to_csv(results / f"{name}.csv", index=False)
    primary.write_json(results / "evaluation_protocol.json", protocol.record())
    primary.write_json(results / "software_versions.json", primary.software_versions())
    frozen_files = [processed_output / name for name in PROCESSED_FILES]
    frozen_files += [results / f"{name}.csv" for name in FREEZE_TABLES]
    frozen_files += [results / "evaluation_protocol.json", results / "software_versions.json"]
    primary.write_json(results / "freeze_manifest.json", {
        "status": "frozen", "version": protocol.VERSION, "predictions_generated": False,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
        "files": {str(p.resolve()): prep.file_sha256(p) for p in frozen_files},
    })
    return tables["coverage_summary"]


def load_frozen(processed_output, results):
    processed_output, results = Path(processed_output), Path(results)
    manifest = read_json(results / "freeze_manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("version") != protocol.VERSION or manifest.get("predictions_generated") is not False:
        raise ValueError("Incomplete within-day freeze")
    if manifest["sources"] != source_hashes():
        raise ValueError("Within-day source files changed after freezing")
    if read_json(results / "software_versions.json") != primary.software_versions():
        raise ValueError("Modelling software changed after freezing")
    if read_json(results / "evaluation_protocol.json") != json.loads(json.dumps(protocol.record())):
        raise ValueError("Within-day protocol changed after freezing")
    for path, digest in {**manifest["inputs"], **manifest["files"]}.items():
        if prep.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input or output changed: {path}")
    if not {str((processed_output / name).resolve()) for name in PROCESSED_FILES}.issubset(manifest["files"]):
        raise ValueError("Processed directory differs from the frozen directory")
    data = pd.read_csv(processed_output / "observations.csv")
    references = pd.read_csv(processed_output / "references.csv")
    test = pd.read_csv(results / "participant_folds.csv")
    validation = pd.read_csv(results / "validation_fold_assignments.csv")
    validate_features(data)
    primary.validate_saved_assignments(data, references, test, validation)
    training_feasibility(data, test, validation)
    return data, references, test, validation


def run(processed_output, results):
    results = Path(results)
    data, references, test, validation = load_frozen(processed_output, results)
    if (results / "REPORT.md").exists():
        raise FileExistsError("A within-day report already exists")
    output = primary.create_evaluation_directory(results / "evaluation", processed_output, results / "protected_preparation")
    primary.write_json(output / "frozen_protocol_reference.json", {"manifest_sha256": prep.file_sha256(results / "freeze_manifest.json")})
    try:
        primary.write_json(output / "run_status.json", {"status": "running"})
        tables = evaluate_models(data, references, test, validation)
        tables["references"] = references
        tables.update(event_tables(tables))
        tables.update(bootstrap_tables(tables["model_predictions"], data))
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        write_report(results, tables)
        primary.write_json(output / "output_manifest.json", {
            "files": {p.name: prep.file_sha256(p) for p in sorted(output.iterdir()) if p.name != "run_status.json"},
            "report_sha256": prep.file_sha256(results / "REPORT.md"),
        })
        primary.write_json(output / "run_status.json", {"status": "complete"})
    except Exception as error:
        primary.write_json(output / "run_status.json", {"status": "failed", "error": str(error)})
        raise
    return output
