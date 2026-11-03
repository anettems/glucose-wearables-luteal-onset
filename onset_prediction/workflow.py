#### Run the feasibility stage using existing audited 2022 inputs, without fitting models.
#### Save new outcomes and paired features under processed data, and counts, folds, provenance
#### and a readable report under results. Refuse to overwrite existing generated outputs.

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from onset_prediction import protocol
from onset_prediction.features import build_features, select_cohort
from onset_prediction.folds import build_folds, participant_profile, represented_events
from onset_prediction.outcomes import event_links, future_outcomes, start_statuses
from preprocessing import file_sha256, temporal_settings


def write_json(path, record):
    #### Save portable JSON and reject undefined numeric values.
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_destinations(processed_output, results):
    #### Require separate fresh generated destinations within the designated output roots.
    processed_root = (protocol.ROOT / "data/processed").resolve()
    results_root = (protocol.ROOT.parent / "results").resolve()
    for destination, allowed in [(processed_output, processed_root), (results, results_root)]:
        destination = Path(destination).resolve()
        if allowed not in destination.parents:
            raise ValueError(f"Destination must be a subdirectory of {allowed}")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"Choose a fresh output directory: {destination}")


def source_hashes():
    #### Record the new code and existing measurement/history dependencies without changing them.
    paths = list(Path(__file__).parent.rglob("*.py"))
    paths += list(protocol.ROOT.glob("*.py"))
    paths += [protocol.ROOT / "trend_analysis/features.py", protocol.ROOT / "trend_analysis/protocol.py",
              protocol.ROOT / "requirements.txt", Path(__file__).with_name("README.md")]
    return {str(path.relative_to(protocol.ROOT)): file_sha256(path) for path in sorted(set(paths))}


def input_hashes(processed, preparation, history_reference):
    #### Hash the audited inputs, unchanged-feature comparison and human-readable protocol.
    paths = [processed / name for name in protocol.HISTORY_FILES]
    paths += [preparation / "run_config.json", history_reference / "observations.csv",
              protocol.ROOT.parent / "EVALUATION_PROTOCOL.md"]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def coverage_tables(audit, links):
    #### Count selection stages and distinguish daily positives from distinct onset events.
    masks = [("full_alignment", pd.Series(True, index=audit.index)),
             ("sensor_eligible", audit.aligned_inputs_available),
             ("history_eligible", audit.history_eligible),
             ("known_future_outcome", audit.supervised_eligible)]
    rows = []
    for name, mask in masks:
        data = audit.loc[mask]
        positive = int(data[protocol.TARGET].eq(1).sum())
        negative = int(data[protocol.TARGET].eq(0).sum())
        rows.append({"stage": name, "participants": data.id.nunique(), "days": len(data),
                     "positive_days": positive, "negative_days": negative,
                     "unknown_outcomes": int(data[protocol.TARGET].isna().sum()),
                     "represented_starts": len(represented_events(data, links))})
    participants = audit.groupby("id").agg(
        recording_days=("id", "size"), sensor_eligible_days=("aligned_inputs_available", "sum"),
        history_eligible_days=("history_eligible", "sum"), usable_days=("supervised_eligible", "sum"))
    return pd.DataFrame(rows), participants.reset_index()


def reference_coverage(audit, links):
    #### Count eligible prediction opportunities before every exact reference start.
    references = audit.loc[audit.start_status.eq(1).fillna(False), protocol.KEY].copy()
    references = references.rename(columns={"day_in_study": "reference_day"})
    selected = links.merge(audit.loc[audit.supervised_eligible, protocol.KEY],
                          on=protocol.KEY, validate="many_to_one")
    counts = selected.groupby(["id", "study_interval", "reference_day"]).size().rename("usable_prediction_days")
    return references.join(counts, on=["id", "study_interval", "reference_day"]).fillna(
        {"usable_prediction_days": 0}).astype({"usable_prediction_days": int})


def build_report(summary, coverage, counts):
    #### Present concrete cohort sizes and the weakest fold counts in a short generated report.
    lines = ["# Three-day onset feasibility", "", f"Protocol: `{protocol.VERSION}`", "",
             "This stage constructs outcomes and folds only. No model or preprocessing was fitted.", "",
             "| Selection stage | Participants | Days | Positive | Negative | Unknown | Distinct starts |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in coverage.itertuples(index=False):
        lines.append(f"| {row.stage} | {row.participants} | {row.days} | {row.positive_days} | "
                     f"{row.negative_days} | {row.unknown_outcomes} | {row.represented_starts} |")
    lines += ["", "Distinct starts are counted once even when several preceding days are positive.", "",
              f"Feature values match all {summary['historical_feature_rows_checked']} earlier history-ready rows.",
              "", "| Cohort type | Number checked | Participants, range | Minimum positive days | Minimum negative days | Minimum distinct starts |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for stage, group in counts.groupby("stage", sort=False):
        lines.append(f"| {stage} | {len(group)} | {group.participants.min()}–{group.participants.max()} | "
                     f"{group.positive_days.min()} | {group.negative_days.min()} | {group.represented_starts.min()} |")
    lines += ["", f"All planned cohorts feasible: **{summary['feasible']}**.", "",
              "Feasible means both outcome classes, at least one represented onset and observed values",
              "for all 15 predictors. These are operational checks, not a sample-size power calculation.", "",
              "Fresh assignments balance the new outcome, using seed 20261020 and development-specific",
              "validation seeds 20261021–20261025. No model scores were used to choose folds.", "",
              "See fold_feasibility.csv for every cohort and feature_training_feasibility.csv for",
              "observed predictor counts. See reference_coverage.csv for starts with zero opportunities.", "",
              "Unknown future outcomes are excluded; the target does not include same-day onset.",
              "The six model/input combinations share one cohort and these saved assignments.", "",
              "This is a feasibility snapshot. The model evaluation stage still needs implementation",
              "and a verified evaluation freeze before test probabilities are generated.", ""]
    return "\n".join(lines)


def save_outputs(processed_output, results, tables, summary, provenance):
    #### Write fresh preparation outputs, a readable report and hashes for the saved snapshot.
    check_destinations(processed_output, results)
    processed_output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    cohort = tables["observations"]
    for name in ("outcome_audit", "observations", "reference_links"):
        tables[name].to_csv(processed_output / f"{name}.csv", index=False)
    for name, columns in protocol.record()["feature_sets"].items():
        cohort[protocol.KEY + columns].to_csv(processed_output / f"{name}_features.csv", index=False)
    cohort[protocol.KEY + [protocol.TARGET]].to_csv(processed_output / "outcomes.csv", index=False)
    for name, data in tables.items():
        if name not in ("outcome_audit", "observations", "reference_links"):
            data.to_csv(results / f"{name}.csv", index=False)
    write_json(results / "cohort_summary.json", summary)
    write_json(results / "evaluation_protocol.json", protocol.record())
    write_json(results / "provenance.json", provenance)
    (results / "REPORT.md").write_text(build_report(summary, tables["coverage_summary"],
                                                   tables["fold_feasibility"]), encoding="utf-8")
    paths = sorted(processed_output.iterdir()) + sorted(results.iterdir())
    write_json(results / "manifest.json", {"stage": "feasibility_only", "feasible": summary["feasible"],
               "models_fitted": False, "predictions_generated": False,
               "files": {str(path.resolve()): file_sha256(path) for path in paths}})


def run_feasibility(processed, preparation, history_reference, processed_output, results):
    #### Build new outcomes and participant folds from the existing audited inputs, then save counts.
    processed, preparation, history_reference, processed_output, results = map(
        Path, (processed, preparation, history_reference, processed_output, results))
    check_destinations(processed_output, results)
    before = input_hashes(processed, preparation, history_reference)
    configuration = json.loads((preparation / "run_config.json").read_text(encoding="utf-8"))
    if configuration.get("temporal_settings") != temporal_settings():
        raise ValueError("Audited preparation timing no longer matches the existing implementation")
    alignment = pd.read_csv(processed / "prediction_alignment_14_2022.csv")
    outcomes = future_outcomes(start_statuses(alignment))
    links = event_links(outcomes)
    enriched = build_features(processed, configuration, history_reference)
    audit, cohort = select_cohort(alignment, enriched, outcomes)
    test, validation, counts, observed = build_folds(cohort, links)
    coverage, participants = coverage_tables(audit, links)
    summary = make_summary(audit, cohort, links, counts, history_reference)
    tables = {"outcome_audit": audit, "observations": cohort, "reference_links": links,
              "participant_folds": test, "validation_fold_assignments": validation,
              "fold_feasibility": counts, "feature_training_feasibility": observed,
              "coverage_summary": coverage, "participant_coverage": participants,
              "participant_profile": participant_profile(cohort, links),
              "reference_coverage": reference_coverage(audit, links)}
    if before != input_hashes(processed, preparation, history_reference):
        raise ValueError("Inputs changed during preparation")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "inputs": before,
                  "sources": source_hashes(), "python": platform.python_version(),
                  "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")}}
    save_outputs(processed_output, results, tables, summary, provenance)
    return summary


def make_summary(audit, cohort, links, counts, history_reference):
    #### Summarise the usable cohort, unknown outcomes and fold checks with ordinary JSON types.
    positive = int(cohort[protocol.TARGET].sum())
    return {"protocol_version": protocol.VERSION, "participants": int(cohort.id.nunique()),
            "usable_days": len(cohort), "positive_days": positive, "negative_days": len(cohort) - positive,
            "positive_fraction": positive / len(cohort),
            "distinct_represented_starts": len(represented_events(cohort, links)),
            "full_alignment_exact_starts": int(audit.start_status.eq(1).sum()),
            "history_eligible_days": int(audit.history_eligible.sum()),
            "history_eligible_unknown_outcomes": int((audit.history_eligible & audit[protocol.TARGET].isna()).sum()),
            "historical_feature_rows_checked": len(pd.read_csv(history_reference / "observations.csv")),
            "cohorts_checked": len(counts), "feasible": bool(counts.feasible.all()),
            "models_fitted": False, "predictions_generated": False}
