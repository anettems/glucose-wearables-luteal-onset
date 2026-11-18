#### Check that thesis plotting preserves paired results and rejects inconsistent inputs.
#### Synthetic probabilities exercise saved summaries, bootstrap pairing and empty bins.
#### Figure exports are checked without fitting models or reading the source dataset.

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from onset_prediction.figure_data import validate_design, validate_estimates
from onset_prediction.metrics import (bootstrap_metrics, calibration_bins, participant_draws,
                                      summary_tables)
from onset_prediction.thesis_figures import (calibration_figure, check_destinations,
                                           export_figure, performance_figure)
from onset_prediction.figure_diagrams import evaluation_diagram, prediction_timeline


def synthetic_results():
    #### Create three paired input sets with known outcomes, including probabilities zero and one.
    observations = pd.DataFrame({'id': [1, 1, 2, 2], 'study_interval': [2022] * 4,
                                 'day_in_study': [10, 11, 10, 11],
                                 'onset_within_3_days': [0, 1, 0, 1]})
    assignments = pd.DataFrame({'id': [1, 2], 'test_fold': [1, 2]})
    rows, bins = [], []
    for name, scores in [('glucose', [0, .2, .3, 1]), ('wearable', [.1, .3, .4, .8]),
                          ('combined', [.1, .5, .2, .7])]:
        frame = observations.merge(assignments, on='id').assign(
            probability=scores, input_set=name, model_family='logistic_regression')
        rows.append(frame)
        bins.append(calibration_bins(frame).assign(input_set=name, model_family='logistic_regression'))
    predictions = pd.concat(rows, ignore_index=True)
    draws = participant_draws([1, 2], resamples=5)
    bootstrap = bootstrap_metrics(predictions, draws)
    summary, paired = summary_tables(predictions, bootstrap)
    return {'test_predictions': predictions, 'observations': observations,
            'participant_folds': assignments, 'bootstrap_participants': draws,
            'bootstrap_metrics': bootstrap, 'model_summary': summary,
            'paired_comparison': paired, 'calibration_bins': pd.concat(bins, ignore_index=True)}


class FigureDataTests(unittest.TestCase):
    #### Protect figure values against plausible but misleading changes in saved tables.

    def test_valid_saved_values_and_immutability(self):
        #### Accept consistent results without modifying the source tables.
        tables = synthetic_results()
        original = deepcopy(tables)
        validate_estimates(tables, 'logistic_regression')
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], original[name])

    def test_unpaired_or_invalid_predictions(self):
        #### Reject missing keys, changed outcomes, foreign intervals and invalid probabilities.
        for column, value in [('id', 99), ('onset_within_3_days', 1),
                              ('study_interval', 2024), ('probability', 1.1)]:
            tables = synthetic_results()
            tables['test_predictions'].loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_estimates(tables, 'logistic_regression')

    def test_changed_summary_interval_or_bin_rejected(self):
        #### Catch corrupted estimates, confidence limits, counts and empty-bin summaries.
        changes = [('model_summary', 'estimate', .99), ('model_summary', 'ci_upper', .99),
                   ('calibration_bins', 'days', 100), ('paired_comparison', 'estimate', .99)]
        for name, column, value in changes:
            tables = synthetic_results()
            tables[name].loc[0, column] = value
            with self.subTest(table=name, column=column), self.assertRaises(ValueError):
                validate_estimates(tables, 'logistic_regression')
        tables = synthetic_results()
        empty = tables['calibration_bins'].days.eq(0).idxmax()
        tables['calibration_bins'].loc[empty, 'observed_frequency'] = 0
        with self.assertRaises(ValueError):
            validate_estimates(tables, 'logistic_regression')

    def test_unpaired_bootstrap_difference_rejected(self):
        #### Do not allow independently altered differences to appear as paired intervals.
        tables = synthetic_results()
        tables['bootstrap_metrics'].loc[0, 'combined_minus_wearable_average_precision'] += .1
        with self.assertRaises(ValueError):
            validate_estimates(tables, 'logistic_regression')

    def test_design_mismatch_rejected(self):
        #### Prevent a different horizon from being described by the fixed three-day diagram.
        with self.assertRaises(ValueError):
            validate_design({'horizon_days': [1, 2, 3, 4]})


class FigureExportTests(unittest.TestCase):
    #### Verify plotted numerical coordinates, complete bin counts and safe exports.

    def test_calibration_points_and_counts(self):
        #### Empty bins have no points, and count rows retain every day including probability one.
        tables = synthetic_results()
        fig = calibration_figure({name: tables for name in ('logistic_regression', 'random_forest')})
        ax, counts = fig.axes[:2]
        for collection, name in zip(ax.collections, ('glucose', 'wearable', 'combined')):
            bins = tables['calibration_bins'].query('input_set == @name and days > 0')
            np.testing.assert_allclose(collection.get_offsets(),
                                       bins[['mean_probability', 'observed_frequency']])
        self.assertEqual(len(counts.texts), 30)
        self.assertEqual(sum(int(text.get_text()) for text in counts.texts), 12)
        plt.close(fig)

    def test_four_exports_and_no_overwrite(self):
        #### Export every diagram and result figure, retaining existing outputs on a repeat call.
        tables = synthetic_results()
        families = {name: tables for name in ('logistic_regression', 'random_forest')}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            figures = [prediction_timeline(), evaluation_diagram(),
                       performance_figure(families), calibration_figure(families)]
            for index, fig in enumerate(figures):
                export_figure(fig, destination, f'figure_{index}')
            self.assertEqual(len(list(destination.iterdir())), 8)
            self.assertTrue(all(path.stat().st_size > 1000 for path in destination.iterdir()))
            before = (destination / 'figure_0.pdf').read_bytes()
            fig = prediction_timeline()
            with self.assertRaises(FileExistsError):
                export_figure(fig, destination, 'figure_0')
            self.assertEqual((destination / 'figure_0.pdf').read_bytes(), before)
            plt.close(fig)

    def test_output_location_rejected(self):
        #### Keep generated output away from source data and unrelated user folders.
        with self.assertRaises(ValueError):
            check_destinations(Path(__file__).parent, None)


if __name__ == '__main__':
    unittest.main()
