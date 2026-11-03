#### Run a separate descriptive assessment of the existing onset task's measurements.
#### Read verified processed 2022 records without changing source data or trained models.
#### Save fresh aligned intermediate data, figures, numerical summaries and a readable report.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onset_assessment.analysis import (EVENT_KEY, MEASUREMENTS, OFFSETS, RESAMPLES, SEED,
                                       align_measurements, centre_measurements, select_events,
                                       trajectory_tables, window_contrasts)
from onset_assessment.plots import (plot_coverage, plot_participant_contrasts, plot_trajectories)
from onset_prediction import protocol
from onset_prediction.freezing import read_json, verify_hashes
from preprocessing import file_sha256


def write_json(path, record):
    #### Save readable reproducibility records using plain JSON-compatible values.
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_destination(path, allowed):
    #### Require a fresh folder inside the appropriate generated-output root.
    path, allowed = Path(path).resolve(), Path(allowed).resolve()
    if allowed not in path.parents:
        raise ValueError(f"Output must be below {allowed}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose a fresh output folder: {path}")


def load_inputs(processed, preparation):
    #### Verify historical file hashes and use the saved full-quality timeline and paired cohort.
    manifest = read_json(preparation / "manifest.json")
    if manifest.get("feasible") is not True:
        raise ValueError("Expected a completed feasible onset preparation")
    verify_hashes(manifest["files"])
    paths = [processed / f"{name}.csv" for name in ("outcome_audit", "observations", "reference_links")]
    for path in paths:
        if manifest["files"].get(str(path.resolve())) != file_sha256(path):
            raise ValueError("Assessment input is not the verified onset preparation")
    return tuple(pd.read_csv(path) for path in paths), {
        str(path.resolve()): file_sha256(path) for path in paths + [preparation / "manifest.json"]}


def report_table(summary, contrast):
    #### Present the size, uncertainty and participant consistency of each measurement change.
    lines = ["| Measurement | Mean change [95% interval] | Participants | Events | Increased / decreased / unchanged |",
             "| --- | ---: | ---: | ---: | ---: |"]
    selected = summary.loc[summary.contrast.eq(contrast)].set_index("measurement")
    for name, (label, unit) in MEASUREMENTS.items():
        row = selected.loc[name]
        lines.append(f"| {label} ({unit}) | {row.mean_change:+.3f} "
                     f"[{row.ci_lower:+.3f}, {row.ci_upper:+.3f}] | {int(row.participants)} | "
                     f"{int(row.events)} | {int(row.positive_participants)} / "
                     f"{int(row.negative_participants)} / {int(row.zero_participants)} |")
    return lines


def key_findings(summary):
    #### Translate the measured pre/post changes into a short interpretation tied to counts.
    pre = summary.loc[summary.contrast.eq("pre_change")].set_index("measurement")
    post = summary.loc[summary.contrast.eq("post_change")].set_index("measurement")
    heart, temperature, glucose = pre.loc["sleeping_hr_bpm"], pre.loc["skin_temperature_c"], pre.loc["glucose_median_mmol_l"]
    return ["## Main observations", "",
            f"- **Sleeping heart rate has the clearest shared advance pattern.** Its mean change is "
            f"{heart.mean_change:+.2f} beats/min during days -3 to -1 "
            f"(95% interval {heart.ci_lower:+.2f} to {heart.ci_upper:+.2f}). "
            f"It increases in {int(heart.positive_participants)} of {int(heart.participants)} participants "
            "with adequate paired coverage. The daily curve shows a gradual increase towards onset.",
            f"- **Temperature is clearer on and after onset.** The pre-onset mean change is "
            f"{temperature.mean_change:+.2f} °C, with an interval spanning zero. "
            f"The corresponding days 0 to +2 change is {post.loc['skin_temperature_c', 'mean_change']:+.2f} °C. "
            "Temperature alone offers a less consistent three-day advance signal in these comparisons.",
            f"- **Glucose median shows a small, inconsistent dip.** The pre-onset change is "
            f"{glucose.mean_change:+.2f} mmol/L, but only {int(glucose.negative_participants)} of "
            f"{int(glucose.participants)} participant means decrease; the others increase or remain unchanged.",
            "- **Glucose variability and HRV show no clear shared pre-onset change.** "
            "Their pre-onset intervals include zero and participant directions are mixed.",
            "These observations help explain why advance prediction may be difficult. A small heart-rate "
            "rise is a candidate signal, but these onset-centred plots do not establish how often the same "
            "rise happens on days without an upcoming onset.", ""]


def write_report(destination, audit, observations, events, summary, trajectories):
    #### Explain the observed patterns using saved quantities without assigning prediction accuracy.
    lines = ["# Measurements before and after reference luteal onset", "",
             "## What was assessed", "",
             f"The models' shared cohort has {observations.id.nunique()} participants and {len(observations):,} days.",
             f"This assessment uses {len(events)} distinct represented onsets from {events.id.nunique()} participants,",
             f"among {int(audit.start_status.eq(1).sum())} exact starts in the complete 2022 label timeline.",
             "An event is represented when at least one model-eligible day lies one to three days before it.",
             "Individual quality-approved signals are inspected around those events even when the",
             "other sensors or histories required for modelling are unavailable. Coverage is shown separately.", "",
             "The horizontal axis is prediction day relative to the supplied label transition (day 0).",
             "Glucose on day t summarises day t-1; wearables describe the main sleep ending on t by 14:00.",
             "Thus the glucose point at -1 describes calendar day -2, and the point at +1 describes day 0.",
             "HRV is shown as RMSSD in milliseconds by exponentiating the existing log-RMSSD measurement.",
             "The model's logged feature is unchanged. Glucose variability is its existing coefficient of variation.", ""]
    lines += key_findings(summary)
    lines += ["## Changes during the three prediction days: -3 to -1", "",
             "Every event is centred on its own mean over days -7 to -4, requiring at least three of four days.",
             "The following comparison requires at least two days in both -3 to -1 and 0 to +2,",
             "so the before/after summaries use the same events for each measurement.", ""]
    lines += report_table(summary, "pre_change")
    lines += ["", "## Changes on and after onset: 0 to +2", ""] + report_table(summary, "post_change")
    lines += ["", "## Additional change from before to after onset", ""] + report_table(summary, "post_minus_pre")
    lines += ["", "Events are averaged within each participant before averaging participants equally.",
              "Increased/decreased counts refer to participant means, not a required prediction rule.",
              "Changes within 1e-12 native units of zero count as unchanged to avoid floating-point artefacts.",
              "Intervals use 2,000 whole-participant resamples with seed 20261021.",
              "They describe uncertainty in the group mean, not the range of individual responses.", "",
              "## Changes on each prediction day", "",
              "These use all available centred observations, so their contributing events can differ from",
              "the paired-window tables above. Intervals are pointwise; no day was selected as an optimum.", "",
              "| Measurement | Day -3: change [95% interval], participants | Day -2 | Day -1 |",
              "| --- | --- | --- | --- |"]
    for name, (label, _) in MEASUREMENTS.items():
        data = trajectories.loc[trajectories.measurement.eq(name)].set_index("relative_day")
        entries = [f"{data.loc[day].mean_change:+.3f} [{data.loc[day].ci_lower:+.3f}, "
                   f"{data.loc[day].ci_upper:+.3f}], n={int(data.loc[day].participants)}" for day in (-3, -2, -1)]
        lines.append(f"| {label} | " + " | ".join(entries) + " |")
    lines += ["", "## Figures", "",
              "![Measurement changes](onset_measurement_changes.png)", "",
              "![Participant consistency](participant_consistency.png)", "",
              "![Measurement coverage](measurement_coverage.png)", "",
              "Vector PDF copies are saved alongside every PNG.", "",
              "## Interpretation and limits", "",
              "A clear change before day 0 would be consistent with a possible advance signal.",
              "A change mainly on or after day 0 is less helpful for the current future-onset target.",
              "Opposing participant changes make a shared model harder to train even if the group mean shifts.",
              "Inspect the participant heatmap alongside group averages and coverage before drawing that conclusion.",
              "These plots are conditioned on known onset dates and do not measure separation from all negative days.",
              "They therefore cannot establish prediction accuracy or independently confirm biological ovulation.",
              "The reference is the supplied luteal-label transition. Day 0 also includes a night ending on that day.",
              "Missing values are never interpolated, imputed or treated as zero. Requiring early baseline and",
              "paired-window coverage excludes some events; event_window_coverage.csv records every exclusion.",
              "The early baseline is a descriptive event-specific anchor, distinct from the model's unchanged",
              "14-day personal median and seven-day trend. No model, target, feature or fold is changed.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_assessment(processed, preparation, processed_output, results):
    #### Validate inputs, record the chosen design, then generate fresh intermediate and report files.
    processed, preparation = Path(processed).resolve(), Path(preparation).resolve()
    processed_output, results = Path(processed_output).resolve(), Path(results).resolve()
    check_destination(processed_output, protocol.ROOT / "data/processed")
    check_destination(results, protocol.ROOT.parent / "results")
    (audit, observations, links), inputs = load_inputs(processed, preparation)
    events = select_events(audit, observations, links)
    processed_output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    sources = [*Path(__file__).parent.rglob("*.py"), Path(__file__).parent / "README.md",
               protocol.ROOT / "onset_prediction/outcomes.py"]
    write_json(results / "assessment_protocol.json", {
        "version": "2026-11-03-onset-measurements-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_selection": "distinct_reference_starts_represented_by_primary_positive_days",
        "measurement_selection": "all_available_individually_quality_approved_signals",
        "relative_prediction_days": OFFSETS.tolist(), "baseline_days": [-7, -6, -5, -4],
        "baseline_minimum": 3, "pre_days": [-3, -2, -1], "post_days": [0, 1, 2],
        "paired_window_minimum_each": 2, "weighting": "events_within_person_then_equal_participants",
        "direction_zero_tolerance_native_units": 1e-12,
        "bootstrap_resamples": RESAMPLES, "bootstrap_seed": SEED + 1, "imputation": None,
        "inputs": inputs, "sources": {str(path.resolve()): file_sha256(path) for path in sources},
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "matplotlib", "scikit-learn")}}})
    aligned = align_measurements(audit, events)
    long = centre_measurements(aligned)
    participant_days, trajectories = trajectory_tables(long)
    event_windows, participants, summary = window_contrasts(long)
    for name, table in {"reference_events": events, "aligned_measurements": aligned,
                        "centred_measurements": long}.items():
        table.to_csv(processed_output / f"{name}.csv", index=False)
    tables = {"participant_trajectories": participant_days, "daily_trajectories": trajectories,
              "event_window_coverage": event_windows, "participant_window_changes": participants,
              "window_summary": summary}
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=False)
    plot_trajectories(participant_days, trajectories, results)
    plot_participant_contrasts(participants, results)
    plot_coverage(trajectories, results)
    write_report(results, audit, observations, events, summary, trajectories)
    verify_hashes(inputs)
    outputs = [path for folder in (processed_output, results) for path in folder.iterdir() if path.is_file()]
    write_json(results / "manifest.json", {"status": "complete", "events": len(events),
        "participants": int(events.id.nunique()), "files": {str(path): file_sha256(path) for path in outputs}})
    return summary


def main():
    #### Run the assessment from the thesis workspace or explicit input/output locations.
    parser = argparse.ArgumentParser(description="Inspect existing measurements around reference luteal onset.")
    parser.add_argument("--processed-dir", type=Path,
                        default=protocol.ROOT / "data/processed/onset_prediction_2022_v1")
    parser.add_argument("--preparation-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_prediction_2022_v1")
    parser.add_argument("--processed-output-dir", type=Path,
                        default=protocol.ROOT / "data/processed/onset_assessment_2022_v1")
    parser.add_argument("--results-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_assessment_2022_v1")
    args = parser.parse_args()
    summary = run_assessment(args.processed_dir, args.preparation_dir, args.processed_output_dir, args.results_dir)
    print(summary.to_string(index=False))
    print(f"Report: {(args.results_dir / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
