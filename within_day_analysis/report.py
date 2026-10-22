"""Readable results with coverage, paired uncertainty and onset counts."""

import pandas as pd

from within_day_analysis import protocol


def markdown_table(frame):
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(results, tables):
    coverage = pd.read_csv(results / "coverage_summary.csv")
    availability = pd.read_csv(results / "feature_availability.csv")
    daily = tables["model_summary"].query("metric != 'luteal_prevalence'").copy()
    daily["Estimate [95% CI]"] = daily.apply(lambda r: f"{r.estimate:.4f} [{r.ci_lower:.4f}, {r.ci_upper:.4f}]", axis=1)
    daily = daily[["variant", "input_set", "metric", "Estimate [95% CI]"]]
    effects = tables["feature_comparisons"].copy()
    effects["Difference [95% CI]"] = effects.apply(lambda r: f"{r.estimate:+.4f} [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]", axis=1)
    paired = tables["paired_comparison"].copy()
    paired["Combined minus wearable AP [95% CI]"] = paired.apply(lambda r: f"{r.estimate:+.4f} [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]", axis=1)
    lines = ["# Exploratory within-day glucose extension", "", f"Protocol: {protocol.VERSION}.", "",
             "## Coverage", "", markdown_table(coverage), "", markdown_table(availability), "",
             "Baseline and extended models were refitted on the same eligible keys. The wearable fit is reused "
             "unchanged across variants. No new glucose feature or glucose gap is filled.", "",
             "## Daily performance", "", markdown_table(daily), "",
             "## Feature addition", "", markdown_table(effects[["input_set", "metric", "Difference [95% CI]"]]), "",
             "## Added glucose value", "", markdown_table(paired[["variant", "Combined minus wearable AP [95% CI]"]]), "",
             "## Onset assessment", "", markdown_table(tables["onset_summary"].round(4)), "",
             "Thresholds and confirmation rules were selected using validation predictions only. Timing error "
             "concerns matched events and must be read with missed starts and unmatched alerts.", "",
             "## Interpretation limits", "", "These are exploratory results after earlier test inspection. "
             "Bootstrap intervals use paired participant resamples and are conditional on fitted cross-validation models; "
             "they are not adjusted for multiple comparisons. Onset outcomes have no uncertainty intervals. "
             "Clock-based night/day periods are not measured fasting or sleep. Added availability requirements "
             "can select different days. Reference labels do not independently establish biological onset. "
             "New glucose features have no extra personal-history transformations. Extra Trees is not part of this run.", ""]
    (results / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
