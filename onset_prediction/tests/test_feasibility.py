#### Test calendar outcomes, unchanged histories and participant-fold feasibility.
#### Small synthetic timelines cover missing dates, uncertain starts and recording boundaries.
#### No tests fit models, alter source exports or generate held-out predictions.

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_prediction import protocol
from onset_prediction.features import select_cohort
from onset_prediction.folds import audit_folds, build_folds, represented_events
from onset_prediction.outcomes import event_links, future_outcomes, start_statuses
from onset_prediction.workflow import check_destinations
from trend_analysis.features import window_statistics


def timeline(days, phases, participant=1):
    #### Make a labelled 2022 timeline without silently filling missing dates.
    return pd.DataFrame({"id": participant, "study_interval": 2022,
                         "day_in_study": days, "phase": phases})


def synthetic_cohort():
    #### Provide twenty participants with two starts each for deterministic fold checks.
    rows, links = [], []
    for person in range(1, 21):
        for day in range(1, 13):
            positive = day in (3, 4, 5, 9, 10, 11)
            row = {"id": person, "study_interval": 2022, "day_in_study": day,
                   protocol.TARGET: int(positive)}
            row.update({name: person + day / 100 for name in protocol.feature_columns("combined")})
            rows.append(row)
            if positive:
                links.append({"id": person, "study_interval": 2022, "day_in_study": day,
                              "reference_day": 6 if day < 6 else 12})
    return pd.DataFrame(rows), pd.DataFrame(links)


class OutcomeTests(unittest.TestCase):
    def test_three_calendar_days_and_no_same_day_onset(self):
        #### A start on day six makes only days three, four and five positive.
        data = timeline(range(1, 10), ["follicular"] * 5 + ["luteal"] * 4)
        outcomes = future_outcomes(start_statuses(data)).set_index("day_in_study")
        self.assertEqual(outcomes.loc[[3, 4, 5], protocol.TARGET].tolist(), [1, 1, 1])
        self.assertEqual(outcomes.loc[2, protocol.TARGET], 0)
        self.assertEqual(outcomes.loc[6, protocol.TARGET], 0)
        self.assertTrue(pd.isna(outcomes.loc[7, protocol.TARGET]))
        links = event_links(outcomes.reset_index())
        self.assertEqual(len(links), 3)
        self.assertEqual(links.reference_day.unique().tolist(), [6])

    def test_missing_dates_do_not_extend_horizon_or_confirm_start(self):
        #### A missing prior label makes the first observed luteal day uncertain.
        data = timeline([1, 3, 4, 5, 6], ["follicular", "luteal", "luteal", "luteal", "menstrual"])
        outcomes = future_outcomes(start_statuses(data)).set_index("day_in_study")
        self.assertTrue(pd.isna(outcomes.loc[3, "start_status"]))
        self.assertTrue(pd.isna(outcomes.loc[1, protocol.TARGET]))
        self.assertEqual(outcomes.loc[3, protocol.TARGET], 0)
        self.assertTrue(event_links(outcomes.reset_index()).empty)

    def test_known_start_overrides_other_unknown_days(self):
        #### One exact start is enough for a positive despite another missing future record.
        data = timeline([1, 2, 4], ["follicular", "luteal", "luteal"])
        outcomes = future_outcomes(start_statuses(data)).set_index("day_in_study")
        self.assertEqual(outcomes.loc[1, protocol.TARGET], 1)
        self.assertEqual(outcomes.loc[1, "future_records_missing"], 1)

    def test_missing_labels_and_non_luteal_after_gap(self):
        #### Missing labels leave uncertain starts unknown but a non-luteal day is no start.
        data = timeline([1, 2, 3, 4, 5], ["follicular", None, "luteal", "menstrual", "follicular"])
        statuses = start_statuses(data).set_index("day_in_study")
        self.assertTrue(pd.isna(statuses.loc[2, "start_status"]))
        self.assertTrue(pd.isna(statuses.loc[3, "start_status"]))
        self.assertEqual(statuses.loc[4, "start_status"], 0)

    def test_participant_boundaries_and_bad_keys(self):
        #### A different participant cannot supply the previous day's label.
        data = pd.concat([timeline([1, 2], ["follicular", "follicular"]),
                          timeline([3, 4], ["luteal", "luteal"], participant=2)])
        statuses = start_statuses(data)
        self.assertTrue(pd.isna(statuses.loc[statuses.id.eq(2), "start_status"].iloc[0]))
        for invalid in (pd.concat([data, data.iloc[[0]]]), data.assign(study_interval=2024),
                        data.assign(day_in_study=1.5), data.assign(phase="unknown_label")):
            with self.assertRaises(ValueError):
                start_statuses(invalid)


class FeatureTests(unittest.TestCase):
    def test_history_windows_are_calendar_based_and_past_only(self):
        #### Future and current extremes cannot affect the earlier reference or trend.
        days = np.arange(1, 22)
        values = days.astype(float)
        expected = window_statistics(days, values, 15)
        values[days >= 15] = 10000
        self.assertEqual(expected, window_statistics(days, values, 15))
        self.assertAlmostEqual(expected["slope"], 1)
        self.assertEqual(expected["last_reference_day"], 14)
        sparse = window_statistics(np.array([1, 2, 3, 30]), np.ones(4), 31)
        self.assertEqual(sparse["reference_count"], 1)
        self.assertTrue(np.isnan(sparse["slope"]))

    def test_current_phase_is_not_an_eligibility_requirement(self):
        #### Preserve a known future outcome when today's supplied phase is absent.
        alignment = timeline([10], [None]).assign(
            aligned_inputs_available=True, prior_day_source_day=9, prediction_cutoff_seconds=14 * 3600,
            main_sleep_end_seconds=8 * 3600, valid_wearable_count=3)
        enriched = alignment[protocol.KEY].copy()
        for name in protocol.feature_columns("combined"):
            enriched[name] = 1.0
        for name in ("glucose_median_mmol_l", "glucose_cv_percent", "sleeping_hr_bpm",
                     "log_median_rmssd_ms", "skin_temperature_c"):
            enriched[f"{name}__last_reference_day"] = 9
            enriched[f"{name}__last_trend_day"] = 9
        outcomes = alignment[protocol.KEY + ["phase"]].assign(**{protocol.TARGET: 0})
        _, cohort = select_cohort(alignment, enriched, outcomes)
        self.assertEqual(len(cohort), 1)
        self.assertTrue(pd.isna(cohort.phase.iloc[0]))

    def test_protected_outputs_are_rejected(self):
        #### New outputs must not target raw data or overwrite an existing preparation.
        with self.assertRaises(ValueError):
            check_destinations(protocol.ROOT / "data/raw/new", protocol.ROOT.parent / "results/new")
        with self.assertRaises(FileExistsError):
            check_destinations(protocol.ROOT / "data/processed/primary_1400_2022",
                               protocol.ROOT.parent / "results/new")


class FoldTests(unittest.TestCase):
    def test_reproducible_disjoint_balanced_folds_and_event_counts(self):
        #### Every planned cohort has both classes, distinct events and observed predictors.
        data, links = synthetic_cohort()
        test, validation, counts, observed = build_folds(data, links)
        repeat = build_folds(data.sample(frac=1, random_state=7), links)
        pd.testing.assert_frame_equal(test, repeat[0])
        pd.testing.assert_frame_equal(validation, repeat[1])
        self.assertEqual(len(counts), 40)
        self.assertTrue(counts.feasible.all())
        self.assertTrue(observed.observed_days.gt(0).all())
        self.assertEqual(len(represented_events(data, links)), 40)
        self.assertEqual(int(data[protocol.TARGET].sum()), 120)
        self.assertEqual(counts.loc[counts.stage.eq("test"), "participants"].tolist(), [4] * 5)
        for name in ("glucose", "wearable", "combined"):
            table = data[protocol.KEY + protocol.feature_columns(name)]
            pd.testing.assert_frame_equal(table[protocol.KEY], data[protocol.KEY])

    def test_leakage_missing_assignments_and_infeasible_cohorts(self):
        #### Reject invalid assignments and flag absent classes or trainable predictor values.
        data, links = synthetic_cohort()
        test, validation, _, _ = build_folds(data, links)
        for broken in (pd.concat([test, test.iloc[[0]]]), test.iloc[1:]):
            with self.assertRaises(ValueError):
                audit_folds(data, links, broken, validation)
        leaked = validation.copy()
        leaked.loc[0, "id"] = test.loc[test.test_fold.eq(1), "id"].iloc[0]
        with self.assertRaises(ValueError):
            audit_folds(data, links, test, leaked)
        missing = data.assign(**{protocol.TARGET: 0, protocol.feature_columns("wearable")[0]: np.nan})
        counts, _ = audit_folds(missing, links.iloc[:0], test, validation)
        self.assertFalse(counts.feasible.any())


if __name__ == "__main__":
    unittest.main()
