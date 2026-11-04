#### Check that thesis plots preserve saved estimates and reject inconsistent source summaries.

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from onset_assessment.cycle_analysis import (identify_cycles, phase_summaries,
    cycle_contrasts, participant_effects, effect_summary, adjacent_pairs)
from onset_assessment.tests.test_cycles import synthetic_days
from onset_assessment.thesis_figures import draw_figures, effect_panel, repeat_panel


class ThesisFigureTests(unittest.TestCase):
    #### Exercise exports and the guards that protect numerical figure labels.

    def setUp(self):
        #### Prepare known synthetic contrasts without depending on project results.
        _, assigned = identify_cycles(synthetic_days())
        contrasts = cycle_contrasts(phase_summaries(assigned))
        self.people = participant_effects(contrasts)
        self.summary = effect_summary(self.people)
        self.pairs = adjacent_pairs(contrasts)

    def test_export_and_immutability(self):
        #### Save both figures and verify input data are unchanged by plotting.
        originals = [table.copy(deep=True) for table in [self.people, self.summary, self.pairs]]
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            draw_figures(self.people, self.summary, self.pairs, destination)
            self.assertEqual(len(list(destination.iterdir())), 4)
            self.assertTrue(all(path.stat().st_size > 1000 for path in destination.iterdir()))
            with self.assertRaises(FileExistsError):
                draw_figures(self.people, self.summary, self.pairs, destination)
        for actual, expected in zip([self.people, self.summary, self.pairs], originals):
            pd.testing.assert_frame_equal(actual, expected)
        plt.close('all')

    def test_inconsistent_summary_rejected(self):
        #### Prevent a changed source count or estimate from silently appearing in a caption.
        fig, ax = plt.subplots()
        self.summary.loc[self.summary.measurement.eq('glucose_median_mmol_l'), 'difference'] += 1
        with self.assertRaises(ValueError):
            effect_panel(ax, self.people, self.summary, 'glucose_median_mmol_l', 'Test', 'mmol/L')
        plt.close(fig)

    def test_empty_pairs_rejected(self):
        #### Avoid exporting a misleading blank repeatability panel.
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            repeat_panel(ax, self.pairs.iloc[:0], 'glucose_median_mmol_l', 'Test', 'mmol/L')
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
