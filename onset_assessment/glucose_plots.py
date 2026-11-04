#### Plot actual glucose-feature overlap and positive-minus-negative participant differences.
#### Label-based centring is used only to display distributions; it is never a model input.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from onset_assessment.glucose_comparison import FEATURES
from onset_assessment.plots import save_figure
from onset_prediction import protocol


def plot_distributions(data, destination):
    #### Balance participants within each outcome class and display overlap on shared bins.
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (feature, (label, unit)) in zip(axes.flat, FEATURES.items()):
        selected = data.loc[data.feature.eq(feature)]
        bins = np.histogram_bin_edges(selected.display_value, bins=30)
        for outcome, colour, name in ((0, "#277c9e", "Negative days"), (1, "#d2782c", "Positive days")):
            group = selected.loc[selected[protocol.TARGET].eq(outcome)]
            axis.hist(group.display_value, bins=bins, weights=group.display_weight,
                      histtype="step", linewidth=2.1, color=colour, label=name)
        axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel(f"Value minus participant's negative-day mean\n({unit})")
        axis.set_ylabel("Weighted fraction of days per bin")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Do existing glucose features distinguish upcoming-onset days?\n"
                   "Each participant receives equal weight within each class",
                   fontsize=16, fontweight="bold")
    save_figure(figure, destination, "glucose_feature_distributions")


def plot_differences(comparisons, summary, destination):
    #### Show participant-level heterogeneity and the equally weighted mean with its interval.
    figure, axes = plt.subplots(3, 2, figsize=(13, 16), constrained_layout=True)
    people = sorted(comparisons.id.unique())
    for axis, (feature, (label, unit)) in zip(axes.flat, FEATURES.items()):
        data = comparisons.loc[comparisons.feature.eq(feature)].set_index("id").loc[people]
        row = summary.loc[summary.feature.eq(feature)].iloc[0]
        axis.axvline(0, color="#777777", linestyle=":", linewidth=0.9)
        axis.scatter(data.mean_difference, range(len(people)), s=22, color="#277c9e")
        axis.errorbar(row.mean_difference, len(people) + 1,
                      xerr=[[max(0, row.mean_difference - row.mean_difference_ci_lower)],
                            [max(0, row.mean_difference_ci_upper - row.mean_difference)]],
                      fmt="D", color="#b65324", capsize=4, linewidth=2)
        axis.set_yticks(list(range(len(people))) + [len(people) + 1],
                        [str(person) for person in people] + ["Mean"])
        axis.invert_yaxis()
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel(f"Positive-day mean minus negative-day mean ({unit})")
        axis.set_ylabel("Participant ID")
        axis.grid(axis="x", alpha=0.15)
    figure.suptitle("Consistency of glucose-feature differences within participants\n"
                   "Orange: equally weighted mean and 95% participant-bootstrap interval",
                   fontsize=16, fontweight="bold")
    save_figure(figure, destination, "within_participant_differences")
