"""Export reporting figures using observed values and saved test results only."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from reporting_tables import FEATURE_LABELS, MODEL_LABELS


COLOURS = {"glucose": "#0072B2", "wearable": "#D55E00", "combined": "#009E73"}


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def coverage_figure(observations, alignment, output):
    participants = sorted(alignment.id.unique())
    days = np.arange(int(alignment.day_in_study.min()), int(alignment.day_in_study.max()) + 1)
    grid = np.full((len(participants), len(days)), np.nan)
    person_index = {person: index for index, person in enumerate(participants)}
    for row in alignment.itertuples():
        grid[person_index[row.id], int(row.day_in_study) - days[0]] = 0
    for row in observations.itertuples():
        grid[person_index[row.id], int(row.day_in_study) - days[0]] = 1 + row.luteal_status
    figure, axis = plt.subplots(figsize=(11, max(4, len(participants) * 0.18)), layout="constrained")
    image = axis.imshow(grid, aspect="auto", interpolation="none", vmin=0, vmax=2,
                        cmap=ListedColormap(["#dddddd", "#56B4E9", "#CC79A7"]))
    axis.set(xticks=np.arange(0, len(days), 10), xticklabels=days[::10],
             yticks=np.arange(len(participants)), yticklabels=[str(p) for p in participants],
             xlabel="Calendar day in study", ylabel="Participant ID", title="2022 aligned days and shared eligible cohort")
    bar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2], shrink=0.55)
    bar.ax.set_yticklabels(["Ineligible", "Eligible, non-luteal", "Eligible, luteal"])
    save_figure(figure, output, "participant_coverage")


def descriptive_figures(observations, alignment, tables, output):
    coverage_figure(observations, alignment, output)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    balance = tables["outcome_balance"]
    axes[0].bar(["Non-luteal", "Luteal"], balance.eligible_days, color=["#56B4E9", "#CC79A7"])
    axes[0].set(ylabel="Eligible days", title=f"Outcome balance (n = {len(observations):,} days)")
    missing = tables["feature_missingness"].query("id == 'all'")
    axes[1].barh([FEATURE_LABELS[f] for f in missing.feature], 100 * missing.missing_fraction, color="#0072B2")
    axes[1].set(xlabel="Missing among eligible days (%)", xlim=(0, 100), title="Before training-only replacement")
    save_figure(figure, output, "outcome_and_missingness")
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for axis, (feature, label) in zip(axes.flat, FEATURE_LABELS.items()):
        values = observations[feature].dropna()
        axis.hist(values, bins=25, color="#0072B2", edgecolor="white")
        axis.set(xlabel=label, ylabel="Eligible days", title=f"Observed n = {len(values):,}")
    axes.flat[-1].axis("off")
    figure.suptitle("Observed feature distributions in the shared eligible cohort")
    save_figure(figure, output, "feature_distributions")


def performance_figure(tables, output):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    names = list(MODEL_LABELS)
    for axis, metric in zip(axes[:2], ["average_precision", "roc_auc"]):
        for index, name in enumerate(names):
            row = tables["model_summary"].query("input_set == @name and metric == @metric").iloc[0]
            axis.plot([row.ci_lower, row.ci_upper], [index, index], color=COLOURS[name])
            axis.plot(row.estimate, index, "o", color=COLOURS[name])
        axis.set(yticks=range(3), yticklabels=list(MODEL_LABELS.values()), xlim=(0, 1),
                 title={"average_precision": "Pooled average precision", "roc_auc": "Pooled ROC-AUC"}[metric])
        axis.grid(axis="x", alpha=0.25)
    row = tables["paired_comparison"].iloc[0]
    axes[2].axvline(0, color="grey", linestyle="--")
    axes[2].plot([row.ci_lower, row.ci_upper], [0, 0], color=COLOURS["combined"])
    axes[2].plot(row.estimate, 0, "o", color=COLOURS["combined"])
    axes[2].set(yticks=[], ylim=(-1, 1), title="Combined minus wearable AP", xlabel="Paired difference")
    figure.suptitle("Held-out estimates and 95% participant-bootstrap intervals\nConditional on the fitted cross-validation models")
    save_figure(figure, output, "daily_performance")


def curve_figures(predictions, tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, label in MODEL_LABELS.items():
        group = predictions.loc[predictions.input_set.eq(name)]
        precision, recall, _ = precision_recall_curve(group.luteal_status, group.probability)
        fpr, tpr, _ = roc_curve(group.luteal_status, group.probability)
        axes[0].step(recall, precision, where="post", label=label, color=COLOURS[name])
        axes[1].plot(fpr, tpr, label=label, color=COLOURS[name])
    axes[0].axhline(group.luteal_status.mean(), color="grey", linestyle="--", label="Luteal prevalence")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    for axis, x, y in zip(axes, ["Recall", "False positive rate"], ["Precision", "True positive rate"]):
        axis.set(xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1.02))
        axis.legend(fontsize=8)
    figure.suptitle("Pooled held-out discrimination curves")
    save_figure(figure, output, "discrimination_curves")
    calibration_figure(tables, output)


def calibration_figure(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for offset, (name, label) in enumerate(MODEL_LABELS.items()):
        bins = tables["calibration_bins"].query("input_set == @name")
        # Empty bins remain gaps rather than invented observations on the curve.
        axes[0].plot(bins.mean_probability, bins.observed_luteal_fraction, "o-", label=label, color=COLOURS[name])
        axes[1].bar(bins.lower_bound + 0.015 + offset * 0.03, bins.days, width=0.028, label=label, color=COLOURS[name])
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed luteal fraction", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Predicted probability (fixed 0.1-wide bins)", ylabel="Eligible days", xlim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle("Descriptive held-out calibration; no recalibration fitted\nDays within participants are correlated; no independent-day intervals")
    save_figure(figure, output, "calibration")


def onset_figure(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    positions = np.arange(3)
    for offset, (metric, label) in enumerate([
        ("matched_starts", "Matched starts"), ("missed_evaluable_starts", "Missed evaluable starts"),
        ("unmatched_alerts", "Unmatched alerts"),
    ]):
        counts = tables["onset_summary"].set_index("input_set").loc[list(MODEL_LABELS), metric]
        axes[0].bar(positions + (offset - 1) * 0.24, counts, width=0.24, label=label)
    axes[0].set(xticks=positions, xticklabels=list(MODEL_LABELS.values()), ylabel="Events", title="Counts under validation-selected alert rules")
    axes[0].legend(fontsize=8)
    for offset, (name, label) in enumerate(MODEL_LABELS.items()):
        group = tables["matched_timing"].query("input_set == @name")
        counts = group.signed_error_days.value_counts().reindex(range(-3, 4), fill_value=0)
        mae = group.timing_error_days.mean()
        text = f"{mae:.2f} d" if len(group) else "undefined"
        axes[1].bar(counts.index + (offset - 1) * 0.24, counts, width=0.24, color=COLOURS[name],
                    label=f"{label}: n={len(group)}, MAE={text}")
    axes[1].set(xticks=range(-3, 4), xlabel="Alert day minus reference day (negative = earlier)",
                ylabel="Matched events", title="Timing among matched starts only")
    axes[1].legend(fontsize=8)
    figure.suptitle("Onset assessment against supplied phase-label starts (±3 calendar days)")
    save_figure(figure, output, "onset_counts_and_timing")


def model_figures(predictions, tables, output):
    performance_figure(tables, output)
    curve_figures(predictions, tables, output)
    onset_figure(tables, output)
