#### Verify calendar boundaries, source-day alignment, paired phase coverage and repeatability.
#### Synthetic values make expected contrasts explicit without relying on project outcomes.

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from onset_assessment.cycle_analysis import (SIGNALS, adjacent_pairs, cycle_contrasts,
    effect_summary, identify_cycles, participant_effects, phase_summaries,
    repeat_summary, source_day_data)


def synthetic_days():
    #### Build two complete cycles between three exactly observed menstrual starts.
    phases = ['luteal'] + (['menstrual'] * 3 + ['follicular'] * 4
                           + ['fertility'] * 3 + ['luteal'] * 5) * 2 + ['menstrual']
    data = pd.DataFrame({'id': 1, 'study_interval': 2022,
                         'day_in_study': np.arange(1, len(phases) + 1), 'phase': phases})
    for name in SIGNALS:
        data[name] = np.where(data.phase.eq('luteal'), 7., 5.)
    return data


class CycleTests(unittest.TestCase):
    #### Check the calculations that determine which days and cycles support the findings.

    def test_source_day(self):
        #### Ensure day t glucose is compared with day t-1 labels and preserve missing values.
        audit = synthetic_days()
        audit['prior_day_source_day'] = audit.day_in_study - 1
        audit['glucose_median_mmol_l'] = audit.day_in_study.astype(float)
        shifted = source_day_data(audit)
        self.assertEqual(shifted.iloc[0].glucose_median_mmol_l, 2.)
        self.assertEqual(shifted.iloc[0].phase, 'luteal')
        self.assertTrue(pd.isna(shifted.iloc[-1].glucose_median_mmol_l))
        audit.loc[3, 'prior_day_source_day'] = 99
        with self.assertRaises(ValueError):
            source_day_data(audit)

    def test_boundaries(self):
        #### Keep next menstruation in the next cycle and exclude incomplete record edges.
        cycles, assigned = identify_cycles(synthetic_days())
        self.assertEqual(cycles.cycle_start.tolist(), [2, 17])
        self.assertEqual(cycles.next_start.tolist(), [17, 32])
        self.assertTrue(cycles.accepted.all())
        self.assertEqual(len(assigned), 30)
        self.assertEqual(assigned.day_in_study.min(), 2)
        self.assertEqual(assigned.day_in_study.max(), 31)

    def test_unknown_gap_reversal(self):
        #### Reject missing dates, unknown labels and reverse phase order within bounded cycles.
        for variant in ['gap', 'unknown', 'reversal']:
            data = synthetic_days()
            if variant == 'gap':
                data = data.loc[data.day_in_study.ne(10)]
            else:
                data.loc[data.day_in_study.eq(10), 'phase'] = (
                    np.nan if variant == 'unknown' else 'menstrual')
            cycles, _ = identify_cycles(data)
            self.assertFalse(cycles.accepted.all())
        data = synthetic_days()
        data.loc[data.day_in_study.eq(1), 'phase'] = np.nan
        cycles, _ = identify_cycles(data)
        self.assertEqual(len(cycles), 1)

    def test_coverage_and_contrasts(self):
        #### Require both absolute and proportional phase coverage before paired subtraction.
        _, assigned = identify_cycles(synthetic_days())
        phases = phase_summaries(assigned)
        contrasts = cycle_contrasts(phases)
        luteal = contrasts.loc[contrasts.phase.eq('luteal')]
        np.testing.assert_allclose(luteal.difference, 2)
        self.assertEqual(len(luteal), 6)
        assigned.loc[assigned.day_in_study.isin([5, 6]), 'glucose_median_mmol_l'] = np.nan
        contrasts = cycle_contrasts(phase_summaries(assigned))
        self.assertFalse(((contrasts.cycle_start == 2)
                          & contrasts.measurement.eq('glucose_median_mmol_l')).any())
        strict = cycle_contrasts(phase_summaries(assigned, minimum=5))
        self.assertTrue(strict.empty)

    def test_equal_participant_weight_and_intervals(self):
        #### Extra cycles must not increase a participant's contribution to the group mean.
        _, assigned = identify_cycles(synthetic_days())
        contrasts = cycle_contrasts(phase_summaries(assigned))
        other = contrasts.loc[contrasts.cycle_number.eq(1)].assign(id=2, difference=-2.,
                                                                 median_difference=-2.)
        people = participant_effects(pd.concat([contrasts, other]))
        summary = effect_summary(people)
        luteal = summary.loc[summary.phase.eq('luteal')]
        np.testing.assert_allclose(luteal.difference, 0)
        np.testing.assert_allclose(luteal.difference_ci_lower, -2)
        np.testing.assert_allclose(luteal.difference_ci_upper, 2)

    def test_half_phase_coverage(self):
        #### Three values cannot represent a phase of more than six labelled days.
        _, assigned = identify_cycles(synthetic_days())
        extra = assigned.loc[assigned.cycle_number.eq(1) & assigned.phase.eq('follicular')].copy()
        extra['day_in_study'] += 100
        assigned = pd.concat([assigned, extra], ignore_index=True)
        mask = assigned.cycle_number.eq(1) & assigned.phase.eq('follicular')
        indices = assigned.index[mask]
        assigned.loc[indices[3:], 'glucose_median_mmol_l'] = np.nan
        phases = phase_summaries(assigned)
        row = phases.loc[phases.cycle_number.eq(1) & phases.phase.eq('follicular')
                         & phases.measurement.eq('glucose_median_mmol_l')].iloc[0]
        self.assertEqual(row.observed_days, 3)
        self.assertEqual(row.labelled_days, 8)
        self.assertFalse(row.eligible)

    def test_adjacent_and_direction(self):
        #### Reverse a later effect and never join cycles across a missing ordinal.
        _, assigned = identify_cycles(synthetic_days())
        contrasts = cycle_contrasts(phase_summaries(assigned))
        contrasts.loc[contrasts.cycle_number.eq(2), 'difference'] *= -1
        pairs = adjacent_pairs(contrasts)
        luteal = pairs.loc[pairs.phase.eq('luteal')]
        self.assertEqual(len(luteal), 3)
        np.testing.assert_allclose(luteal.same_direction, 0)
        np.testing.assert_allclose(luteal.absolute_change, 4)
        summary = repeat_summary(luteal)
        np.testing.assert_allclose(summary.agreement, 0)
        contrasts.loc[contrasts.cycle_number.eq(2), 'cycle_number'] = 3
        self.assertTrue(adjacent_pairs(contrasts).empty)

    def test_figures(self):
        #### Exercise every figure export using a complete synthetic paired example.
        from onset_assessment.cycle_plots import plot_effects, plot_repeatability
        _, assigned = identify_cycles(synthetic_days())
        contrasts = cycle_contrasts(phase_summaries(assigned))
        people = participant_effects(contrasts)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            plot_effects(people, effect_summary(people), destination)
            plot_repeatability(adjacent_pairs(contrasts), destination)
            self.assertEqual(len(list(destination.iterdir())), 4)
            self.assertTrue(all(path.stat().st_size > 1000 for path in destination.iterdir()))


if __name__ == '__main__':
    unittest.main()
