"""Prepare and freeze the extension, then evaluate only verified saved inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import preprocessing as prep
import primary_evaluation as primary
from reporting import reporting_versions
from trend_analysis import protocol
from trend_analysis.features import load_quality_history, add_personal_features, select_history_cohort, audit_tables
from trend_analysis.evaluation import evaluate_variants, event_and_calibration_tables
from trend_analysis.statistics import bootstrap_tables


PROCESSED_FILES = ["quality_history.csv", "personal_history.csv", "observations.csv", "references.csv"]
FREEZE_TABLES = ["participant_folds", "validation_fold_assignments", "fold_feasibility", "coverage_summary",
                 "feature_availability", "participant_coverage", "reference_availability", "feature_training_feasibility"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    folder = Path(__file__).parent
    paths = sorted(folder.rglob("*.py")) + [folder / "README.md"]
    hashes = {str(path.relative_to(folder)): prep.file_sha256(path) for path in paths}
    dependencies = set(primary.SOURCE_FILES) | {"reporting.py", "reporting_tables.py", "reporting_figures.py"}
    hashes.update({f"primary/{name}": prep.file_sha256(folder.parent / name) for name in sorted(dependencies)})
    return hashes


def input_hashes(processed, preparation, frozen):
    paths = [processed / name for name in protocol.HISTORY_INPUTS]
    paths += [preparation / "run_config.json", frozen / "freeze_manifest.json"]
    return {str(path.resolve()): prep.file_sha256(path) for path in paths}


def check_destinations(processed_output, results, protected):
    processed_output, results = Path(processed_output).resolve(), Path(results).resolve()
    generated_root = (protocol.ROOT / "data/processed").resolve()
    if generated_root not in processed_output.parents:
        raise ValueError("Processed history outputs must be under code/data/processed")
    if processed_output in results.parents or results in processed_output.parents or processed_output == results:
        raise ValueError("Processed and results destinations must be separate")
    for destination in [processed_output, results]:
        for item in [*protected, protocol.ROOT / "data/raw"]:
            item = Path(item).resolve()
            if destination == item or item in destination.parents or destination in item.parents:
                raise ValueError("Output overlaps a protected source or primary analysis directory")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"Output directory is not empty: {destination}")


def training_feasibility(observations, test, validation):
    rows = []
    for fold in sorted(test.outer_test_fold.unique()):
        development = observations.loc[~observations.id.isin(test.loc[test.outer_test_fold.eq(fold), "id"])]
        cohorts = [("development", 0, development)]
        for inner in sorted(validation.validation_fold.unique()):
            ids = validation.loc[validation.outer_test_fold.eq(fold) & validation.validation_fold.eq(inner), "id"]
            cohorts.append(("validation_training", inner, development.loc[~development.id.isin(ids)]))
        for stage, inner, training in cohorts:
            for feature in protocol.feature_columns("combined", "deviations_trends"):
                observed = int(training[feature].notna().sum())
                if not observed:
                    raise ValueError(f"No observed training values for {feature}")
                rows.append({"outer_test_fold": fold, "validation_fold": inner, "stage": stage,
                             "feature": feature, "observed_training_days": observed})
    return pd.DataFrame(rows)


def prepare(processed, preparation, frozen, processed_output, results):
    processed, preparation, frozen, processed_output, results = map(Path, [processed, preparation, frozen, processed_output, results])
    check_destinations(processed_output, results, [processed, preparation, frozen, protocol.ROOT.parent / "results/primary_evaluation_2022_v4"])
    observations, references, configuration = primary.load_primary_inputs(processed, preparation)
    test, validation, _ = primary.load_frozen_protocol(frozen, processed, preparation, observations, references)
    before = input_hashes(processed, preparation, frozen)
    history = load_quality_history(processed, configuration)
    enriched = add_personal_features(history)
    audit, cohort = select_history_cohort(observations, enriched)
    test = test.loc[test.id.isin(cohort.id)].reset_index(drop=True)
    validation = validation.loc[validation.id.isin(cohort.id)].reset_index(drop=True)
    starts = references.loc[references.id.isin(cohort.id)].reset_index(drop=True)
    feasibility = primary.validate_saved_assignments(cohort, starts, test, validation)
    tables = {**audit_tables(observations, audit, cohort, references), "participant_folds": test,
              "validation_fold_assignments": validation, "fold_feasibility": feasibility,
              "feature_training_feasibility": training_feasibility(cohort, test, validation)}
    if before != input_hashes(processed, preparation, frozen):
        raise ValueError("Input files changed while preparing history")
    processed_output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    frames = [history, audit, cohort, starts]
    for name, frame in zip(PROCESSED_FILES, frames):
        frame.to_csv(processed_output / name, index=False)
    for name, frame in tables.items():
        frame.to_csv(results / f"{name}.csv", index=False)
    primary.write_json(results / "evaluation_protocol.json", protocol.record())
    primary.write_json(results / "software_versions.json", reporting_versions())
    paths = [processed_output / name for name in PROCESSED_FILES]
    paths += [results / f"{name}.csv" for name in FREEZE_TABLES]
    paths += [results / "evaluation_protocol.json", results / "software_versions.json"]
    manifest = {"status": "frozen", "version": protocol.VERSION, "predictions_generated": False,
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "sources": source_hashes(), "inputs": before,
                "files": {str(path.resolve()): prep.file_sha256(path) for path in paths}}
    primary.write_json(results / "freeze_manifest.json", manifest)
    return tables["coverage_summary"]


def load_frozen(processed_output, results):
    processed_output, results = Path(processed_output), Path(results)
    manifest = read_json(results / "freeze_manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("version") != protocol.VERSION or manifest.get("predictions_generated") is not False:
        raise ValueError("Incomplete exploratory freeze")
    if manifest["sources"] != source_hashes():
        raise ValueError("Exploratory source files changed after freezing")
    if json.dumps(read_json(results / "evaluation_protocol.json"), sort_keys=True) != json.dumps(protocol.record(), sort_keys=True):
        raise ValueError("Protocol or modelling versions changed after freezing")
    if read_json(results / "software_versions.json") != reporting_versions():
        raise ValueError("Reporting software changed after freezing")
    for path, digest in {**manifest["inputs"], **manifest["files"]}.items():
        if prep.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input or output changed: {path}")
    expected = {str((processed_output / name).resolve()) for name in PROCESSED_FILES}
    if not expected.issubset(manifest["files"]):
        raise ValueError("Processed input directory differs from the frozen directory")
    data = pd.read_csv(processed_output / "observations.csv")
    references = pd.read_csv(processed_output / "references.csv")
    test = pd.read_csv(results / "participant_folds.csv")
    validation = pd.read_csv(results / "validation_fold_assignments.csv")
    primary.validate_saved_assignments(data, references, test, validation)
    training_feasibility(data, test, validation)
    return data, references, test, validation


def run(processed_output, results):
    results = Path(results)
    data, references, test, validation = load_frozen(processed_output, results)
    if (results / "REPORT.md").exists():
        raise FileExistsError("An exploratory report already exists")
    output = primary.create_evaluation_directory(results / "evaluation", processed_output, results / "protected_preparation")
    primary.write_json(output / "frozen_protocol_reference.json", {"manifest_sha256": prep.file_sha256(results / "freeze_manifest.json")})
    try:
        primary.write_json(output / "run_status.json", {"status": "running"})
        tables = evaluate_variants(data, references, test, validation)
        tables["references"] = references
        tables.update(event_and_calibration_tables(tables, data))
        tables.update(bootstrap_tables(tables["model_predictions"], data))
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        from trend_analysis.report import write_report
        write_report(results, tables)
        primary.write_json(output / "output_manifest.json", {
            "files": {p.name: prep.file_sha256(p) for p in sorted(output.iterdir()) if p.name != "run_status.json"},
            "review_sha256": prep.file_sha256(results / "REPORT.md"),
        })
        primary.write_json(output / "run_status.json", {"status": "complete"})
    except Exception as error:
        primary.write_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
    return output
