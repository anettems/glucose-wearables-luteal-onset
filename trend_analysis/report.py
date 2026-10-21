"""Reproducible tables, figures and interpretation of the exploratory comparison."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reporting_figures import COLOURS, save_figure
from reporting_tables import MODEL_LABELS
import primary_protocol as config
from trend_analysis.protocol import VARIANTS


LABELS = {"original": "Original", "deviations": "+ personal deviations", "deviations_trends": "+ deviations and trends"}


def markdown_table(frame):
    """Avoid adding tabulate solely for Markdown export."""
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def metric_table(summary, metric):
    rows = []
    for variant in VARIANTS:
        row = {"Features": LABELS[variant]}
        for name, label in MODEL_LABELS.items():
            item = summary.loc[summary.variant.eq(variant) & summary.input_set.eq(name) & summary.metric.eq(metric)].iloc[0]
            row[label] = f"{item.estimate:.3f} [{item.ci_lower:.3f}, {item.ci_upper:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def performance_figure(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, metric, title in zip(axes, ["average_precision", "roc_auc"], ["Pooled held-out AP", "Pooled held-out ROC-AUC"]):
        for offset, (name, label) in enumerate(MODEL_LABELS.items()):
            for index, variant in enumerate(VARIANTS):
                row = tables["model_summary"].query("variant == @variant and input_set == @name and metric == @metric").iloc[0]
                y = index + (offset - 1) * 0.18
                axis.plot([row.ci_lower, row.ci_upper], [y, y], color=COLOURS[name])
                axis.plot(row.estimate, y, "o", color=COLOURS[name], label=label if index == 0 else None)
        axis.set(yticks=range(3), yticklabels=list(LABELS.values()), xlim=(0, 1), title=title)
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Exploratory personal-history comparison on identical days\n95% participant-bootstrap intervals conditional on fitted models")
    save_figure(figure, output, "daily_performance")


def effect_figure(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    comparison_labels = ["Deviations − original", "Deviations + trends − original", "Trends added to deviations"]
    comparisons = tables["feature_comparisons"].comparison.drop_duplicates().tolist()
    for offset, (name, label) in enumerate(MODEL_LABELS.items()):
        for index, comparison in enumerate(comparisons):
            row = tables["feature_comparisons"].query("comparison == @comparison and input_set == @name and metric == 'average_precision'").iloc[0]
            y = index + (offset - 1) * 0.18
            axes[0].plot([row.ci_lower, row.ci_upper], [y, y], color=COLOURS[name])
            axes[0].plot(row.estimate, y, "o", color=COLOURS[name], label=label if index == 0 else None)
    axes[0].set(yticks=range(3), yticklabels=comparison_labels, title="Paired AP changes from feature additions")
    axes[0].legend(fontsize=8)
    for index, variant in enumerate(VARIANTS):
        row = tables["paired_comparison"].query("variant == @variant").iloc[0]
        axes[1].plot([row.ci_lower, row.ci_upper], [index, index], color=COLOURS["combined"])
        axes[1].plot(row.estimate, index, "o", color=COLOURS["combined"])
    axes[1].set(yticks=range(3), yticklabels=list(LABELS.values()), title="Combined minus wearable AP")
    for axis in axes:
        axis.axvline(0, color="grey", linestyle="--")
        axis.set_xlabel("AP difference (95% participant-bootstrap interval)")
    figure.suptitle("Exploratory paired differences on the shared history cohort")
    save_figure(figure, output, "paired_differences")


def onset_figure(tables, output):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, variant in zip(axes, VARIANTS):
        rows = tables["onset_summary"].query("variant == @variant").set_index("input_set").loc[list(MODEL_LABELS)]
        for offset, (metric, label) in enumerate([("matched_starts", "Matched"), ("missed_evaluable_starts", "Missed"), ("unmatched_alerts", "Unmatched alerts")]):
            axis.bar(np.arange(3) + (offset - 1) * 0.25, rows[metric], width=0.25, label=label)
        axis.set(xticks=range(3), xticklabels=["Glucose", "Wearable", "Combined"], title=LABELS[variant], ylabel="Events")
        axis.set_ylim(0, max(1, rows[["matched_starts", "missed_evaluable_starts", "unmatched_alerts"]].max().max() * 1.3))
    axes[0].legend(fontsize=8)
    figure.suptitle("Onset assessment using validation-selected rules; timing is reported alongside counts in REPORT.md")
    save_figure(figure, output, "onset_events")


def write_report(results, tables):
    output = results / "evaluation"
    performance_figure(tables, output)
    effect_figure(tables, output)
    onset_figure(tables, output)
    coverage = pd.read_csv(results / "coverage_summary.csv")
    onset = tables["onset_summary"].copy()
    onset["Features"] = onset.variant.map(LABELS)
    onset["Inputs"] = onset.input_set.map(MODEL_LABELS)
    onset["MAE (days)"] = onset.timing_mae_days.map(lambda x: f"{x:.2f}" if pd.notna(x) else "undefined")
    onset["Event F1"] = onset.event_f1.map(lambda x: f"{x:.3f}")
    onset = onset[["Features", "Inputs", "matched_starts", "missed_evaluable_starts", "unmatched_alerts", "MAE (days)", "Event F1"]]
    differences = tables["feature_comparisons"].query("metric == 'average_precision'").copy()
    differences["AP difference [95% CI]"] = differences.apply(lambda x: f"{x.estimate:+.4f} [{x.ci_lower:+.4f}, {x.ci_upper:+.4f}]", axis=1)
    paired = tables["paired_comparison"].copy()
    paired["Combined minus wearable AP [95% CI]"] = paired.apply(lambda x: f"{x.estimate:+.4f} [{x.ci_lower:+.4f}, {x.ci_upper:+.4f}]", axis=1)
    stability = tables["model_fold_metrics"].groupby(["variant", "input_set"])[["average_precision", "roc_auc"]].agg(["min", "max"])
    stability.columns = [f"{metric}_{bound}" for metric, bound in stability.columns]
    stability = stability.reset_index().round(4)
    lines = ["# Exploratory personal deviations and trends", "", "## Cohort and timing", "", markdown_table(coverage), "",
             "References use at least seven observed values from the preceding 14 calendar days. Slopes use at least four "
             "observed values from the preceding seven calendar days, using actual day spacing. Both exclude the current day. "
             "Glucose values remain aligned to the preceding full calendar day. Histories include valid signals on days "
             "excluded from the shared cohort and do not require historical phase labels. No imputed value enters history.", "",
             "The shared cohort requires both glucose features and at least two wearable features with current value, reference "
             "and slope. Every variant is fitted again on identical days. The original row is the refitted baseline on this "
             "reduced cohort, not the completed primary evaluation. Saved participant and validation folds are reused after "
             "filtering retained IDs and checking feasibility. Earlier unlabelled measurements from a held-out participant "
             "are allowed for online personalisation; population replacement, scaling and model fitting use training participants.", "",
             "## Pooled average precision [95% CI]", "", markdown_table(metric_table(tables["model_summary"], "average_precision")), "",
             "## Pooled ROC-AUC [95% CI]", "", markdown_table(metric_table(tables["model_summary"], "roc_auc")), "",
             "## Paired feature changes", "", markdown_table(differences[["comparison", "input_set", "AP difference [95% CI]"]]), "",
             "## Incremental glucose comparison", "", markdown_table(paired[["variant", "Combined minus wearable AP [95% CI]"]]), "",
             "## Fold stability", "", markdown_table(stability), "",
             "## Onset counts and timing", "", markdown_table(onset), "",
             "Timing errors describe matched events only; undefined values remain missing when there are no matches. "
             "Onset confidence intervals were not added. Fold metrics, selected settings, validation scores, probability "
             "summaries and calibration bins are saved in the evaluation directory.", "",
             "## Interpretation limits", "", "This extension was designed after inspecting primary held-out results and is exploratory. "
             f"Confidence intervals use {config.BOOTSTRAP_RESAMPLES:,} identical participant draws across all nine models and condition on fitted "
             "cross-validation models. Multiple feature contrasts are descriptive and have no multiplicity adjustment. "
             "A confidence interval crossing zero allows changes in either direction. Coverage conditions on having enough "
             "recent measurements; a rolling reference may contain luteal days and is not a follicular baseline. Supplied "
             "phase labels do not independently validate biological onset. Availability by 14:00 remains an assumption.", "",
             "## Figures", ""]
    for name in ["daily_performance", "paired_differences", "onset_events"]:
        lines += [f"![{name}](evaluation/{name}.png)", ""]
    lines += ["## Saved tables", ""]
    lines += [f"- [{path.name}](evaluation/{path.name})" for path in sorted(output.glob("*.csv"))]
    (results / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
