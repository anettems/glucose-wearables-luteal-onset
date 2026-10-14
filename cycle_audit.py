"""Audit individual supplied-label menstrual cycles and sensor coverage.

This is a descriptive quality-control tool.  It does not validate the supplied
phase labels against an independent clinical reference and does not train a
model.  The raw mcPHASES export is read only; derived tables and figures are
written below ``results/mcphases/cycle_audit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PHASES = ("Menstrual", "Follicular", "Fertility", "Luteal")
HORMONES = ("lh", "estrogen", "pdg")
WINDOW_DAYS = 3
PHASE_COLOURS = {
    "Menstrual": "#d55e00",
    "Follicular": "#56b4e9",
    "Fertility": "#e69f00",
    "Luteal": "#009e73",
}


def parse_arguments() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=workspace / "code" / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=workspace / "results" / "mcphases" / "cycle_audit")
    parser.add_argument(
        "--max-plots",
        type=int,
        default=5,
        help="Number of cycle PNGs to create (default: 5; use 192 to request all current cycles).",
    )
    parser.add_argument(
        "--participant-id",
        type=int,
        action="append",
        help="Restrict plots and eligibility tables to one or more participant IDs; repeat the option for multiple IDs.",
    )
    return parser.parse_args()


def load_daily_data(data_dir: Path) -> pd.DataFrame:
    hormones = pd.read_csv(data_dir / "hormones_and_selfreport.csv")
    rhr = pd.read_csv(data_dir / "resting_heart_rate.csv")
    temperature = pd.read_csv(data_dir / "computed_temperature.csv")
    required = {"id", "study_interval", "day_in_study", "phase", *HORMONES}
    missing = required.difference(hormones.columns)
    if missing:
        raise ValueError(f"hormones_and_selfreport.csv is missing: {sorted(missing)}")

    rhr_daily = (
        rhr.loc[rhr["value"] > 0, ["id", "study_interval", "day_in_study", "value"]]
        .groupby(["id", "study_interval", "day_in_study"], as_index=False)
        .median()
        .rename(columns={"value": "resting_heart_rate"})
    )
    temperature_daily = (
        temperature.rename(columns={"sleep_end_day_in_study": "day_in_study"})
        [["id", "study_interval", "day_in_study", "nightly_temperature"]]
        .groupby(["id", "study_interval", "day_in_study"], as_index=False)
        .median()
    )
    return (
        hormones.merge(rhr_daily, on=["id", "study_interval", "day_in_study"], how="left")
        .merge(temperature_daily, on=["id", "study_interval", "day_in_study"], how="left")
        .sort_values(["id", "study_interval", "day_in_study"])
        .reset_index(drop=True)
    )


def assign_cycles(data: pd.DataFrame) -> pd.DataFrame:
    """Start a cycle at each observed transition into a supplied menstrual label."""
    data = data.copy()
    keys = ["id", "study_interval"]
    previous_phase = data.groupby(keys)["phase"].shift()
    data["cycle_number"] = (data["phase"].eq("Menstrual") & ~previous_phase.eq("Menstrual")).groupby(
        [data["id"], data["study_interval"]]
    ).cumsum()
    data = data[data["cycle_number"] > 0].copy()
    data["cycle_day"] = data.groupby([*keys, "cycle_number"]).cumcount() + 1
    return data


def consecutive_days(days: pd.Series) -> bool:
    values = np.sort(days.dropna().unique())
    return len(values) > 0 and np.array_equal(values, np.arange(values[0], values[-1] + 1))


def cycle_record(group: pd.DataFrame) -> dict[str, object]:
    phases_present = set(group["phase"].dropna())
    onset_rows = group.index[group["phase"].eq("Luteal") & ~group["phase"].shift().eq("Luteal")]
    luteal_onset_day = group.loc[onset_rows[0], "cycle_day"] if len(onset_rows) else np.nan
    expected_window = np.arange(luteal_onset_day - WINDOW_DAYS, luteal_onset_day + WINDOW_DAYS + 1) if pd.notna(luteal_onset_day) else np.array([])
    valid_window = expected_window[(expected_window >= group["cycle_day"].min()) & (expected_window <= group["cycle_day"].max())]
    window = group[group["cycle_day"].isin(valid_window)]
    record = {
        "id": group["id"].iat[0],
        "study_interval": group["study_interval"].iat[0],
        "cycle_number": group["cycle_number"].iat[0],
        "cycle_start_day_in_study": group["day_in_study"].min(),
        "cycle_end_day_in_study": group["day_in_study"].max(),
        "cycle_days": len(group),
        "has_all_phase_labels": set(PHASES).issubset(phases_present),
        "has_every_calendar_day": consecutive_days(group["day_in_study"]),
        "luteal_onset_cycle_day": luteal_onset_day,
        "onset_window_days_available": len(valid_window),
        "onset_window_is_full_seven_days": len(valid_window) == 2 * WINDOW_DAYS + 1,
    }
    for column in (*HORMONES, "resting_heart_rate", "nightly_temperature"):
        record[f"{column}_days"] = int(group[column].notna().sum())
        record[f"{column}_complete_cycle"] = bool(group[column].notna().all())
        record[f"{column}_onset_window_days"] = int(window[column].notna().sum())
        record[f"{column}_complete_onset_window"] = bool(
            len(valid_window) == 2 * WINDOW_DAYS + 1 and window[column].notna().all()
        )
    record["all_hormones_complete_cycle"] = bool(group[list(HORMONES)].notna().all().all())
    record["both_wearables_complete_cycle"] = bool(group[["resting_heart_rate", "nightly_temperature"]].notna().all().all())
    record["all_hormones_complete_onset_window"] = bool(
        len(valid_window) == 2 * WINDOW_DAYS + 1 and window[list(HORMONES)].notna().all().all()
    )
    record["both_wearables_complete_onset_window"] = bool(
        len(valid_window) == 2 * WINDOW_DAYS + 1
        and window[["resting_heart_rate", "nightly_temperature"]].notna().all().all()
    )
    return record


def summarise_eligibility(audit: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "Observed menstrual-start cycle": pd.Series(True, index=audit.index),
        "All four supplied phase labels": audit["has_all_phase_labels"],
        "Phase labels plus every calendar day": audit["has_all_phase_labels"] & audit["has_every_calendar_day"],
        "Phase labels plus RHR and temperature every cycle day": audit["has_all_phase_labels"] & audit["both_wearables_complete_cycle"],
        "Phase labels plus all three hormones every cycle day": audit["has_all_phase_labels"] & audit["all_hormones_complete_cycle"],
        "Phase labels plus RHR and temperature for the full ±3-day luteal window": audit["has_all_phase_labels"] & audit["both_wearables_complete_onset_window"],
        "Phase labels plus all three hormones for the full ±3-day luteal window": audit["has_all_phase_labels"] & audit["all_hormones_complete_onset_window"],
    }
    return pd.DataFrame(
        {
            "definition": definitions.keys(),
            "eligible_cycles": [int(values.sum()) for values in definitions.values()],
            "eligible_participants": [int(audit.loc[values, "id"].nunique()) for values in definitions.values()],
        }
    )


def shade_phases(axis: plt.Axes, group: pd.DataFrame) -> None:
    for phase, phase_rows in group.groupby("phase", dropna=True):
        colour = PHASE_COLOURS.get(phase, "#999999")
        for _, block in phase_rows.groupby(phase_rows["cycle_day"].diff().ne(1).cumsum()):
            axis.axvspan(block["cycle_day"].min() - 0.5, block["cycle_day"].max() + 0.5, color=colour, alpha=0.12)


def plot_cycle(group: pd.DataFrame, output_path: Path) -> None:
    signals = [("lh", "LH (mIU/mL)"), ("estrogen", "E3G (ng/mL)"), ("pdg", "PdG (mcg/mL)"), ("resting_heart_rate", "Resting HR (bpm)"), ("nightly_temperature", "Nightly temperature (°C)")]
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 10), sharex=True)
    for axis, (column, label) in zip(axes, signals):
        shade_phases(axis, group)
        axis.plot(group["cycle_day"], group[column], marker="o", markersize=3, linewidth=1.3, color="#1f2937")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    onset = group.loc[group["phase"].eq("Luteal") & ~group["phase"].shift().eq("Luteal"), "cycle_day"]
    if len(onset):
        for axis in axes:
            axis.axvline(onset.iat[0], color=PHASE_COLOURS["Luteal"], linestyle="--", linewidth=1.3)
    axes[-1].set_xlabel("Day since observed menstrual onset")
    legend = [Patch(facecolor=PHASE_COLOURS[phase], alpha=0.25, label=phase) for phase in PHASES]
    axes[0].legend(handles=legend, loc="upper right", ncol=4, fontsize=8)
    fig.suptitle(f"Participant {group['id'].iat[0]} | interval {group['study_interval'].iat[0]} | cycle {group['cycle_number'].iat[0]}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cycles = assign_cycles(load_daily_data(args.data_dir))
    if args.participant_id:
        cycles = cycles[cycles["id"].isin(args.participant_id)].copy()
    if cycles.empty:
        raise ValueError("No menstrual-start cycles matched the requested participant selection.")
    cycle_keys = ["id", "study_interval", "cycle_number"]
    audit = pd.DataFrame(cycle_record(group) for _, group in cycles.groupby(cycle_keys))
    audit.to_csv(args.output_dir / "cycle_eligibility.csv", index=False)
    summary = summarise_eligibility(audit)
    summary.to_csv(args.output_dir / "eligibility_summary.csv", index=False)
    for index, (_, group) in enumerate(cycles.groupby(cycle_keys), start=1):
        if index > args.max_plots:
            break
        name = f"participant_{group['id'].iat[0]}_interval_{group['study_interval'].iat[0]}_cycle_{group['cycle_number'].iat[0]}.png"
        plot_cycle(group, figures_dir / name)
    print(summary.to_string(index=False))
    print(f"Cycle-level audit: {args.output_dir / 'cycle_eligibility.csv'}")
    print(f"Individual figures: {figures_dir}")


if __name__ == "__main__":
    main()
