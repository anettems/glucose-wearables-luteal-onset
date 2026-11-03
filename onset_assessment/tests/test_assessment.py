#### Check calendar alignment, event selection, missingness, equal participant weighting
#### and uncertainty summaries using small synthetic records with known measurements.

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_assessment.analysis import (MEASUREMENTS, align_measurements, bootstrap_means, direction_counts,
                                       centre_measurements, select_events, trajectory_tables,
                                       window_contrasts)
from onset_assessment.main import check_destination
from onset_assessment.plots import plot_coverage, plot_participant_contrasts, plot_trajectories
from onset_prediction import protocol
from onset_prediction.outcomes import event_links, future_outcomes, start_statuses


def synthetic_audit():
    #### Create two complete timelines with a known start on day 10 and linearly rising signals.
    rows = []
    for person in (1, 2):
        for day in range(1, 19):
            value = person + day
            row = {"id": person, "study_interval": 2022, "day_in_study": day,
                   "phase": "follicular" if day < 10 else "luteal", "supervised_eligible": True,
                   "prior_day_source_day": day - 1, "prediction_cutoff_seconds": day * 86400 + 50400,
                   "main_sleep_end_seconds": day * 86400 + 30000,
                   "log_median_rmssd_ms": np.log(value)}
            row.update({name: value for name in MEASUREMENTS if name != "rmssd_ms"})
            rows.append(row)
    return future_outcomes(start_statuses(pd.DataFrame(rows))).merge(
        pd.DataFrame(rows).drop(columns="phase"), on=protocol.KEY, validate="one_to_one")


class AssessmentTests(unittest.TestCase):
    def test_rounding_does_not_create_a_direction(self):
        #### Negligible arithmetic differences must not be counted as physiological changes.
        counts = direction_counts([1e-14, -1e-14, 0, 1, -1])
        self.assertEqual(counts, {"positive_participants": 1, "negative_participants": 1,
                                  "zero_participants": 3})

    def test_reference_selection_uses_only_represented_exact_starts(self):
        #### Multiple positive days count as one event; a start without an eligible day is excluded.
        audit = synthetic_audit()
        observations = audit.loc[audit.id.eq(1) & audit.day_in_study.isin([7, 8, 9, 13])]
        events = select_events(audit, observations, event_links(audit))
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].reference_day, 10)
        self.assertEqual(events.iloc[0].id, 1)
        bad = audit.copy()
        bad.loc[bad.day_in_study.eq(10), "start_status"] = 0
        with self.assertRaises(ValueError):
            select_events(bad, observations, event_links(audit))

    def test_actual_calendar_gaps_rmssd_conversion_and_timing(self):
        #### Missing day 9 stays missing at offset -1, while earlier records retain their offsets.
        audit = synthetic_audit()
        events = pd.DataFrame({"id": [1], "study_interval": [2022], "reference_day": [10]})
        audit = audit.loc[~(audit.id.eq(1) & audit.day_in_study.eq(9))]
        aligned = align_measurements(audit, events).set_index("relative_day")
        self.assertEqual(len(aligned), 15)
        self.assertFalse(aligned.loc[-1].record_available)
        self.assertTrue(np.isnan(aligned.loc[-1].glucose_median_mmol_l))
        self.assertEqual(aligned.loc[-2].day_in_study, 8)
        self.assertAlmostEqual(aligned.loc[-2].rmssd_ms, 9)
        bad = audit.copy()
        bad.loc[bad.id.eq(1), "prior_day_source_day"] += 1
        with self.assertRaises(ValueError):
            align_measurements(bad, events)
        bad = audit.copy()
        bad.loc[bad.id.eq(1), "main_sleep_end_seconds"] = bad.loc[bad.id.eq(1), "prediction_cutoff_seconds"] + 1
        with self.assertRaises(ValueError):
            align_measurements(bad, events)

    def test_baseline_and_paired_window_changes_have_known_values(self):
        #### Known slopes give before=3.5, after=6.5 and after-minus-before=3 in native units.
        audit = synthetic_audit()
        events = pd.DataFrame({"id": [1, 2], "study_interval": [2022, 2022], "reference_day": [10, 10]})
        aligned = align_measurements(audit, events)
        long = centre_measurements(aligned)
        event_windows, participants, summary = window_contrasts(long)
        self.assertTrue(event_windows.paired_eligible.all())
        np.testing.assert_allclose(participants.pre_change, 3.5)
        np.testing.assert_allclose(participants.post_change, 6.5)
        np.testing.assert_allclose(participants.post_minus_pre, 3)
        self.assertTrue(summary.participants.eq(2).all())
        #### Losing one baseline day is allowed; losing two removes only that signal/event.
        mask = aligned.id.eq(1) & aligned.relative_day.isin([-7, -6])
        aligned.loc[mask, "skin_temperature_c"] = np.nan
        long = centre_measurements(aligned)
        affected = long.loc[long.id.eq(1) & long.measurement.eq("skin_temperature_c")]
        self.assertTrue(affected.change.isna().all())
        windows, _, _ = window_contrasts(long)
        self.assertFalse(windows.loc[windows.id.eq(1) & windows.measurement.eq("skin_temperature_c"),
                                     "paired_eligible"].iloc[0])
        #### One missing post-onset day is allowed; two make the paired contrast unavailable.
        aligned.loc[aligned.id.eq(2) & aligned.relative_day.eq(0), "sleeping_hr_bpm"] = np.nan
        windows, _, _ = window_contrasts(centre_measurements(aligned))
        affected = windows.loc[windows.id.eq(2) & windows.measurement.eq("sleeping_hr_bpm")]
        self.assertTrue(affected.paired_eligible.iloc[0])
        aligned.loc[aligned.id.eq(2) & aligned.relative_day.eq(1), "sleeping_hr_bpm"] = np.nan
        windows, _, _ = window_contrasts(centre_measurements(aligned))
        affected = windows.loc[windows.id.eq(2) & windows.measurement.eq("sleeping_hr_bpm")]
        self.assertFalse(affected.paired_eligible.iloc[0])

    def test_participants_have_equal_weight_despite_different_event_counts(self):
        #### Ten events from one person must not outweigh one event from another person.
        rows = []
        for name in MEASUREMENTS:
            for person, events, value in ((1, 10, 1.0), (2, 1, 9.0)):
                for event in range(events):
                    for day in range(-7, 8):
                        rows.append({"id": person, "study_interval": 2022, "reference_day": 20 + event * 30,
                                     "measurement": name, "relative_day": day, "change": value,
                                     "value": value, "supervised_eligible": True})
        long = pd.DataFrame(rows)
        _, daily = trajectory_tables(long)
        np.testing.assert_allclose(daily.mean_change, 5)
        _, people, summary = window_contrasts(long)
        np.testing.assert_allclose(summary.loc[summary.contrast.eq("pre_change"), "mean_change"], 5)
        self.assertEqual(people.groupby("id").events.first().to_dict(), {1: 10, 2: 1})
        first = bootstrap_means(np.array([[1, np.nan], [9, 3]]))
        pd.testing.assert_frame_equal(first, bootstrap_means(np.array([[1, np.nan], [9, 3]])))
        self.assertTrue(first.ci_lower.le(first.ci_upper).all())

    def test_fresh_destinations_and_plot_outputs(self):
        #### Protect existing files and exercise every figure with the same tested numeric tables.
        audit = synthetic_audit()
        events = pd.DataFrame({"id": [1, 2], "study_interval": [2022, 2022], "reference_day": [10, 10]})
        long = centre_measurements(align_measurements(audit, events))
        days, daily = trajectory_tables(long)
        _, participants, _ = window_contrasts(long)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                check_destination(root, root / "allowed")
            destination = root / "outputs"
            check_destination(destination, root)
            destination.mkdir()
            plot_trajectories(days, daily, destination)
            plot_participant_contrasts(participants, destination)
            plot_coverage(daily, destination)
            self.assertEqual(len(list(destination.glob("*.png"))), 3)
            self.assertEqual(len(list(destination.glob("*.pdf"))), 3)
            with self.assertRaises(FileExistsError):
                check_destination(destination, root)


if __name__ == "__main__":
    unittest.main()
