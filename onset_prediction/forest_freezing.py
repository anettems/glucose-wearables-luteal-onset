#### Freeze random-forest settings while reusing the verified logistic cohort and draws.
#### Earlier snapshots are immutable input records, not current-source evaluation freezes.
#### Copy their cohort and assignment files exactly, and capture current forest sources anew.

from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from onset_prediction import protocol
from onset_prediction.features import validate_cohort
from onset_prediction.folds import audit_folds
from onset_prediction.forest import create_forest
from onset_prediction.freezing import (EVALUATION_VERSION as LOGISTIC_VERSION, check_results_destination,
                                        load_feasibility, read_json, verify_hashes, versions)
from onset_prediction.metrics import participant_draws
from onset_prediction.outcomes import validate_keys
from onset_prediction.workflow import source_hashes, write_json
from preprocessing import file_sha256


VERSION = "2026-11-03-onset-3d-forest-v1"
SHARED_FILES = ("observations.csv", "reference_links.csv", "participant_folds.csv",
                "validation_fold_assignments.csv", "fold_feasibility.csv",
                "feature_training_feasibility.csv", "bootstrap_participants.csv")


def forest_record():
    #### State the forest execution settings without changing the agreed outcome or feature plan.
    record = protocol.record()
    record.update({"evaluation_version": VERSION, "stage": "random_forest_evaluation",
                   "enabled_model_families": ["random_forest"], "scaling": "none",
                   "wearable_replacement": "training_participant_feature_median_for_wearables_only",
                   "forest_implementation": create_forest(1).get_params(),
                   "bootstrap_quantile_method": "linear",
                   "bootstrap_scope": "conditional_on_fitted_cross_validation_models",
                   "shared_inputs": "byte_identical_logistic_frozen_cohort_folds_and_bootstrap_draws",
                   "calibration_bin_edges": np.linspace(0, 1, 11).tolist()})
    return record


def read_shared_snapshot(frozen):
    #### Load and validate shared cohort files without reading any model performance results.
    data = pd.read_csv(frozen / "observations.csv")
    test = pd.read_csv(frozen / "participant_folds.csv")
    validation = pd.read_csv(frozen / "validation_fold_assignments.csv")
    links = pd.read_csv(frozen / "reference_links.csv")
    draws = pd.read_csv(frozen / "bootstrap_participants.csv")
    validate_keys(data)
    validate_cohort(data)
    counts, _ = audit_folds(data, links, test, validation)
    if not counts.feasible.all():
        raise ValueError("Shared participant folds are infeasible")
    expected = participant_draws(data.id.unique())
    if not draws.equals(expected):
        raise ValueError("Shared participant draws differ from the agreed seed and cohort")
    return data, test, validation, draws


def verify_logistic_snapshot(frozen):
    #### Verify an earlier saved logistic input record without requiring its historical code today.
    frozen = Path(frozen)
    manifest = read_json(frozen / "freeze_manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("version") != LOGISTIC_VERSION:
        raise ValueError("Expected the completed logistic input freeze")
    verify_hashes(manifest["files"], frozen)
    verify_hashes(manifest["inputs"])
    if read_json(frozen / "software_versions.json") != versions():
        raise ValueError("Numerical versions differ from the logistic snapshot")
    saved, current = read_json(frozen / "evaluation_protocol.json"), protocol.record()
    for name in ("version", "target", "horizon_days", "feature_sets", "models", "seed", "bootstrap"):
        if saved[name] != current[name]:
            raise ValueError(f"Agreed shared design differs from logistic snapshot: {name}")
    if not set(SHARED_FILES).issubset(manifest["files"]):
        raise ValueError("The logistic snapshot lacks shared input files")
    return read_shared_snapshot(frozen)


def check_prepared_pairing(data, prepared, previous_results, frozen):
    #### Require the same keys, targets, feature values and participant assignments as feasibility.
    key_columns = protocol.KEY + [protocol.TARGET]
    if not data[key_columns].equals(prepared[key_columns]):
        raise ValueError("Logistic snapshot and prepared outcome keys differ")
    columns = protocol.feature_columns("combined")
    if not np.allclose(data[columns], prepared[columns], equal_nan=True, rtol=1e-12, atol=1e-12):
        raise ValueError("Logistic snapshot and prepared predictors differ")
    for name in ("participant_folds.csv", "validation_fold_assignments.csv"):
        if not pd.read_csv(previous_results / name).equals(pd.read_csv(frozen / name)):
            raise ValueError("Logistic participant assignments differ from feasibility")


def freeze_forest(processed, feasibility_results, logistic_frozen, frozen):
    #### Save current forest sources/settings before fitting, copying the existing shared inputs.
    processed, feasibility_results, logistic_frozen, frozen = map(
        Path, (processed, feasibility_results, logistic_frozen, frozen))
    check_results_destination(frozen, (processed, feasibility_results, logistic_frozen))
    prepared, inputs = load_feasibility(processed, feasibility_results)
    data, _, _, _ = verify_logistic_snapshot(logistic_frozen)
    check_prepared_pairing(data, prepared, feasibility_results, logistic_frozen)
    frozen.mkdir(parents=True, exist_ok=True)
    for name in (*SHARED_FILES, "protocol_snapshot.md"):
        shutil.copyfile(logistic_frozen / name, frozen / name)
        inputs[str((logistic_frozen / name).resolve())] = file_sha256(logistic_frozen / name)
    inputs[str((logistic_frozen / "freeze_manifest.json").resolve())] = file_sha256(logistic_frozen / "freeze_manifest.json")
    write_json(frozen / "evaluation_protocol.json", forest_record())
    write_json(frozen / "software_versions.json", versions())
    files = {path.name: file_sha256(path) for path in sorted(frozen.iterdir())}
    write_json(frozen / "freeze_manifest.json", {"status": "frozen", "version": VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(), "models_fitted": False,
               "predictions_generated": False, "shared_logistic_freeze": str(logistic_frozen.resolve()),
               "sources": source_hashes(), "inputs": inputs, "files": files})
    return {"participants": int(data.id.nunique()), "days": len(data), "directory": str(frozen.resolve())}


def load_forest_frozen(frozen):
    #### Reject any source, software, settings or input change after the new forest freeze.
    frozen = Path(frozen)
    manifest = read_json(frozen / "freeze_manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("version") != VERSION:
        raise ValueError("Incomplete or incompatible forest freeze")
    verify_hashes(manifest["sources"], protocol.ROOT)
    if manifest["sources"] != source_hashes():
        raise ValueError("Analysis sources were added or removed after forest freezing")
    verify_hashes(manifest["inputs"])
    verify_hashes(manifest["files"], frozen)
    if read_json(frozen / "evaluation_protocol.json") != forest_record():
        raise ValueError("Forest settings changed after freezing")
    if read_json(frozen / "software_versions.json") != versions():
        raise ValueError("Numerical software changed after freezing")
    for name in SHARED_FILES:
        previous = Path(manifest["shared_logistic_freeze"]) / name
        if file_sha256(frozen / name) != file_sha256(previous):
            raise ValueError("Forest and logistic shared snapshot files differ")
    return read_shared_snapshot(frozen)
