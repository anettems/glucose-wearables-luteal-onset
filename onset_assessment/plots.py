#### Draw onset-aligned measurement changes, participant contrasts and measurement coverage.
#### Export PNG previews and vector PDF figures using identical underlying summary tables.
#### The horizontal axis always refers to the prediction day, including the prior-day CGM lag.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from onset_assessment.analysis import MEASUREMENTS, OFFSETS


COLOUR = "#186a88"
PRE_COLOUR = "#e8ad39"


def save_figure(figure, destination, name):
    #### Save readable raster previews and scalable thesis-ready copies, then release memory.
    figure.savefig(destination / f"{name}.png", dpi=170, facecolor="white")
    figure.savefig(destination / f"{name}.pdf", facecolor="white")
    plt.close(figure)


def time_axis(axis):
    #### Mark the actual prediction horizon and the reference transition consistently.
    axis.axvspan(-3.5, -0.5, color=PRE_COLOUR, alpha=0.17)
    axis.axvline(0, color="#444444", linestyle=":", linewidth=1.2)
    axis.axhline(0, color="#aaaaaa", linewidth=0.7)
    axis.set_xlim(-7, 7)
    axis.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    axis.grid(alpha=0.15)


def plot_trajectories(participants, summaries, destination):
    #### Show participant variation behind the equally weighted group mean and its uncertainty.
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for axis, (name, (label, unit)) in zip(axes.flat, MEASUREMENTS.items()):
        data = participants.loc[participants.measurement.eq(name)]
        for _, person in data.groupby("id"):
            axis.plot(person.relative_day, person.change, color="#999999", alpha=0.16, linewidth=0.7)
        group = summaries.loc[summaries.measurement.eq(name)].sort_values("relative_day")
        axis.fill_between(group.relative_day, group.ci_lower, group.ci_upper, color=COLOUR, alpha=0.23)
        axis.plot(group.relative_day, group.mean_change, color=COLOUR, linewidth=2.4, marker="o", markersize=3)
        time_axis(axis)
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_ylabel(f"Change from days -7 to -4 ({unit})")
        axis.set_xlabel("Prediction day relative to reference onset")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.95,
        "Blue: mean across participants.\n"
        "Band: 95% participant-bootstrap interval.\n\n"
        "Grey: participants, averaging their events first.\n\n"
        "Yellow: days -3 to -1, when onset is\n"
        "within the next three days.\n\n"
        "Day 0: first supplied luteal label after\n"
        "a known non-luteal day.\n\n"
        "CGM at t summarises calendar day t-1.\n"
        "Wearables: night ending on t by 14:00.\n\n"
        f"Quality-approved signals around {int(summaries.total_events.max())} represented starts.\n"
        "Baseline: at least 3 days among -7 to -4.\n"
        "Gaps stay missing; coverage varies by day.",
        va="top", fontsize=9, linespacing=1.25, transform=axes.flat[-1].transAxes)
    figure.suptitle("Do measurements change before luteal onset?", fontsize=18, fontweight="bold")
    save_figure(figure, destination, "onset_measurement_changes")


def plot_participant_contrasts(participants, destination):
    #### Compare each participant's pre/post changes on identical events using native units.
    people = sorted(participants.id.unique())
    figure, axes = plt.subplots(1, 5, figsize=(15, 12), constrained_layout=True)
    for axis, (name, (label, unit)) in zip(axes, MEASUREMENTS.items()):
        data = participants.loc[participants.measurement.eq(name)].set_index("id").reindex(people)
        matrix = data[["pre_change", "post_change"]].to_numpy()
        limit = max(float(np.nanmax(np.abs(matrix))), 0.0001)
        colour_map = plt.get_cmap("RdBu_r").with_extremes(bad="#d0d0d0")
        graphic = axis.imshow(matrix, aspect="auto", cmap=colour_map,
                              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        axis.set_xticks([0, 1], ["Before\n-3 to -1", "After\n0 to +2"])
        axis.set_yticks(range(len(people)), [str(person) for person in people])
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_ylabel("Participant ID")
        figure.colorbar(graphic, ax=axis, location="bottom", label=f"Change ({unit})", shrink=0.85)
    figure.suptitle("Consistency across participants\nSame events before and after; relative to days -7 to -4\n"
                   "Red: increase; blue: decrease; grey: insufficient coverage",
                   fontsize=15, fontweight="bold")
    save_figure(figure, destination, "participant_consistency")


def plot_coverage(summaries, destination):
    #### Contrast available measurements and the subset that passed the actual modelling cohort.
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (name, (label, _)) in zip(axes.flat, MEASUREMENTS.items()):
        data = summaries.loc[summaries.measurement.eq(name)].sort_values("relative_day")
        axis.plot(data.relative_day, data.observed_events, color=COLOUR, marker="o", label="Quality-approved signal")
        axis.plot(data.relative_day, data.centred_events, color="#659b59", linestyle="--", label="Signal + early baseline")
        axis.plot(data.relative_day, data.model_eligible_observed_events, color="#a95064", linestyle=":",
                  marker=".", label="Signal on model-eligible day")
        time_axis(axis)
        axis.set_ylim(0, data.total_events.max() + 3)
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_ylabel("Events with a measurement")
        axis.set_xlabel("Prediction day relative to reference onset")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    figure.suptitle(f"Coverage around the {int(summaries.total_events.max())} represented onset events",
                   fontsize=17, fontweight="bold")
    save_figure(figure, destination, "measurement_coverage")
