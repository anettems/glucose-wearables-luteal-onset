#### Freeze the verified feasibility cohort and logistic settings before model fitting.
#### Preserve earlier preparation outputs and check saved hashes, participant folds and inputs.
#### The new freeze captures current evaluation sources and software, with no predictions.

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from onset_prediction import protocol
from onset_prediction.features import validate_cohort
from onset_prediction.folds import audit_folds
from onset_prediction.logistic import create_classifier
from onset_prediction.metrics import INPUT_SETS, participant_draws
from onset_prediction.outcomes import validate_keys
from onset_prediction.workflow import source_hashes, write_json
from preprocessing import file_sha256


EVALUATION_VERSION = "2026-11-03-onset-3d-logistic-v1"


def read_json(path):
    #### Read one UTF-8 protocol, provenance or manifest record.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def versions():
    #### Record the numerical packages used in fitting, prediction and uncertainty estimates.
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def evaluation_record():
    #### Extend the unchanged feasibility design with this logistic-only execution stage.
    record = protocol.record()
    record.update({"evaluation_version": EVALUATION_VERSION, "stage": "logistic_evaluation",
                   "enabled_model_families": ["logistic_regression"],
                   "deferred_model_families": ["random_forest"],
                   "wearable_replacement": "training_participant_feature_median_for_wearables_only",
                   "logistic_implementation": create_classifier(1).get_params(),
                   "L2_interface": "l1_ratio=0; unpenalised_intercept; no_class_or_sample_weights",
                   "bootstrap_quantile_method": "linear",
                   "bootstrap_scope": "conditional_on_fitted_cross_validation_models",
                   "calibration_bin_edges": np.linspace(0, 1, 11).tolist()})
    return record


def verify_hashes(hashes, root=None):
    #### Refuse a changed or absent saved source, input or generated snapshot file.
    for name, expected in hashes.items():
        path = Path(name) if root is None else Path(root) / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Saved file changed or is missing: {path}")


def check_results_destination(destination, protected=()):
    #### Require a fresh results directory that cannot overlap a preparation or frozen input.
    destination = Path(destination).resolve()
    allowed = (protocol.ROOT.parent / "results").resolve()
    if allowed not in destination.parents:
        raise ValueError("Evaluation and freeze outputs must be below results/")
    for item in protected:
        item = Path(item).resolve()
        if destination == item or item in destination.parents or destination in item.parents:
            raise ValueError("Output overlaps protected analysis inputs")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Choose a fresh results directory: {destination}")


def construction_sources(sources):
    #### Keep feature/outcome dependencies while accepting either saved Windows path separator.
    entry_points = {"onset_prediction/main.py", "onset_prediction/README.md"}
    return {name: digest for name, digest in sources.items()
            if name.replace("\\", "/") not in entry_points}


def load_feasibility(processed, results):
    #### Verify the completed preparation and its unchanged feature/outcome construction sources.
    processed, results = Path(processed).resolve(), Path(results).resolve()
    manifest, provenance = read_json(results / "manifest.json"), read_json(results / "provenance.json")
    if manifest.get("stage") != "feasibility_only" or manifest.get("feasible") is not True:
        raise ValueError("A completed feasible preparation is required")
    verify_hashes(manifest["files"])
    verify_hashes(provenance["inputs"])
    if read_json(results / "evaluation_protocol.json") != protocol.record():
        raise ValueError("The saved feasibility design differs from the current design")
    #### Only the entry point and user documentation may change when adding an evaluation mode.
    verify_hashes(construction_sources(provenance["sources"]), protocol.ROOT)
    observations = pd.read_csv(processed / "observations.csv")
    for name in INPUT_SETS:
        columns = protocol.KEY + protocol.feature_columns(name)
        table = pd.read_csv(processed / f"{name}_features.csv")
        if table.columns.tolist() != columns or not table.equals(observations[columns]):
            raise ValueError("Prepared predictor tables are not identical to the shared cohort")
    validate_keys(observations)
    validate_cohort(observations)
    inputs = {str(path.resolve()): file_sha256(path) for path in [
        processed / "observations.csv", processed / "reference_links.csv", results / "manifest.json",
        results / "provenance.json", results / "participant_folds.csv", results / "validation_fold_assignments.csv"]}
    inputs.update(manifest["files"])
    inputs.update(provenance["inputs"])
    return observations, inputs


def freeze_logistic(processed, feasibility_results, frozen):
    #### Save the existing cohort, assignments, bootstrap draws and settings before any fitting.
    processed, feasibility_results, frozen = map(Path, (processed, feasibility_results, frozen))
    check_results_destination(frozen, (processed, feasibility_results))
    data, inputs = load_feasibility(processed, feasibility_results)
    test = pd.read_csv(feasibility_results / "participant_folds.csv")
    validation = pd.read_csv(feasibility_results / "validation_fold_assignments.csv")
    links = pd.read_csv(processed / "reference_links.csv")
    counts, observed = audit_folds(data, links, test, validation)
    if not counts.feasible.all():
        raise ValueError("The saved participant folds are no longer feasible")
    frozen.mkdir(parents=True, exist_ok=True)
    frames = {"observations": data, "reference_links": links, "participant_folds": test,
              "validation_fold_assignments": validation, "fold_feasibility": counts,
              "feature_training_feasibility": observed,
              "bootstrap_participants": participant_draws(data.id.unique())}
    for name, frame in frames.items():
        frame.to_csv(frozen / f"{name}.csv", index=False)
    write_json(frozen / "evaluation_protocol.json", evaluation_record())
    write_json(frozen / "software_versions.json", versions())
    shutil.copyfile(protocol.ROOT.parent / "EVALUATION_PROTOCOL.md", frozen / "protocol_snapshot.md")
    files = {path.name: file_sha256(path) for path in sorted(frozen.iterdir())}
    write_json(frozen / "freeze_manifest.json", {"status": "frozen", "version": EVALUATION_VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(), "models_fitted": False,
               "predictions_generated": False, "sources": source_hashes(), "inputs": inputs, "files": files})
    return {"participants": int(data.id.nunique()), "days": len(data), "directory": str(frozen.resolve())}


def load_frozen(frozen):
    #### Verify current sources, numerical versions and every saved file before allowing fitting.
    frozen = Path(frozen)
    manifest = read_json(frozen / "freeze_manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("version") != EVALUATION_VERSION:
        raise ValueError("Incomplete or incompatible logistic freeze")
    verify_hashes(manifest["sources"], protocol.ROOT)
    if manifest["sources"] != source_hashes():
        raise ValueError("Analysis sources were added or removed after freezing")
    verify_hashes(manifest["inputs"])
    verify_hashes(manifest["files"], frozen)
    if read_json(frozen / "evaluation_protocol.json") != evaluation_record():
        raise ValueError("Logistic settings changed after freezing")
    if read_json(frozen / "software_versions.json") != versions():
        raise ValueError("Numerical software changed after freezing")
    data = pd.read_csv(frozen / "observations.csv")
    test = pd.read_csv(frozen / "participant_folds.csv")
    validation = pd.read_csv(frozen / "validation_fold_assignments.csv")
    links = pd.read_csv(frozen / "reference_links.csv")
    validate_keys(data)
    validate_cohort(data)
    counts, _ = audit_folds(data, links, test, validation)
    if not counts.feasible.all():
        raise ValueError("Frozen cohort is infeasible")
    return data, test, validation, pd.read_csv(frozen / "bootstrap_participants.csv")
