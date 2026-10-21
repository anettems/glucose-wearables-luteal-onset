"""Training separation, paired bootstrap and complete synthetic workflow checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "tests")]
from daily_models import fit_and_predict as primary_fit
from participant_folds import build_fold_assignments
import primary_protocol as primary_config
from trend_analysis import protocol, workflow
from trend_analysis.features import FEATURES
from trend_analysis.models import fit_and_predict
from trend_analysis.statistics import feature_comparisons, validate_all_predictions
from test_history import long_observations, history_from


class TrendEvaluationTests(unittest.TestCase):
    def test_replacement_scaling_and_coefficients_use_training_participants(self):
        data = long_observations(3)
        data["derived"] = data.sleeping_hr_bpm - 60
        training = data.loc[data.id.le(2)].copy()
        test = data.loc[data.id.eq(3)].copy()
        training.loc[training.index[0], "derived"] = np.nan
        test["derived"] = np.nan
        columns = FEATURES + ["derived"]
        _, parameters = fit_and_predict(training, test, columns, 0.01)
        self.assertEqual(parameters["replacement_medians"]["derived"], training.derived.median())
        self.assertAlmostEqual(parameters["feature_means"]["derived"], training.derived.fillna(training.derived.median()).mean())
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_and_predict(training, training.iloc[:1], columns, 0.01)
        with self.assertRaisesRegex(ValueError, "no observed training"):
            fit_and_predict(training.assign(derived=np.nan), test, columns, 0.01)

    def test_original_variant_exactly_matches_primary_model_logic(self):
        data = long_observations(3)
        training, test = data.loc[data.id.le(2)], data.loc[data.id.eq(3)]
        expected, _ = primary_fit(training, test, FEATURES, 0.1)
        actual, _ = fit_and_predict(training, test, FEATURES, 0.1)
        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)

    def test_paired_feature_intervals_use_difference_of_same_draws(self):
        summaries, draws = [], []
        for variant, offset in zip(protocol.VARIANTS, [0.0, 0.1, 0.2]):
            for name in ["glucose", "wearable", "combined"]:
                for metric in ["average_precision", "roc_auc"]:
                    summaries.append({"variant": variant, "input_set": name, "metric": metric, "estimate": 0.5 + offset})
            for replicate, noise in enumerate([-0.1, 0.0, 0.1]):
                row = {"variant": variant, "replicate": replicate, "participant_draw": "[1, 2]"}
                row.update({f"{name}_{metric}": 0.5 + offset + noise for name in ["glucose", "wearable", "combined"]
                            for metric in ["average_precision", "roc_auc"]})
                draws.append(row)
        table = feature_comparisons(pd.DataFrame(summaries), pd.DataFrame(draws))
        selected = table.loc[table.comparison.eq("deviations_minus_original")]
        self.assertTrue(np.allclose(selected.estimate, 0.1))
        self.assertTrue(np.allclose(selected.ci_lower, 0.1))
        self.assertTrue(np.allclose(selected.ci_upper, 0.1))
        changed = pd.DataFrame(draws)
        changed.loc[0, "participant_draw"] = "[2, 2]"
        with self.assertRaisesRegex(ValueError, "paired participant"):
            feature_comparisons(pd.DataFrame(summaries), changed)

    def test_destinations_protect_raw_and_existing_directories(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            with patch.object(protocol, "ROOT", root / "code"):
                processed = root / "code/data/processed/new"
                for results in [root / "code/data/raw/new", processed / "nested", root]:
                    with self.assertRaises(ValueError):
                        workflow.check_destinations(processed, results, [])
                results = root / "results"
                results.mkdir()
                (results / "keep.txt").write_text("original")
                with self.assertRaises(FileExistsError):
                    workflow.check_destinations(processed, results, [])
                self.assertEqual((results / "keep.txt").read_text(), "original")

    def test_full_synthetic_freeze_evaluation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, \
                patch.object(primary_config, "BOOTSTRAP_RESAMPLES", 8):
            root = Path(directory).resolve()
            data = long_observations(10)
            references = data.loc[data.exact_luteal_onset.eq(1), primary_config.KEY]
            test, validation, _ = build_fold_assignments(data, references)
            source, preparation, frozen = root / "source", root / "preparation", root / "primary_freeze"
            for path in [source, preparation, frozen]:
                path.mkdir()
            for name in protocol.HISTORY_INPUTS:
                (source / name).write_text("synthetic fixture\n")
            (preparation / "run_config.json").write_text("{}")
            (frozen / "freeze_manifest.json").write_text("{}")
            processed, results = root / "code/data/processed/history", root / "results/trend"
            with patch.object(protocol, "ROOT", root / "code"), \
                    patch.object(workflow.primary, "load_primary_inputs", return_value=(data, references, {})), \
                    patch.object(workflow.primary, "load_frozen_protocol", return_value=(test, validation, None)), \
                    patch.object(workflow, "load_quality_history", return_value=history_from(data)), \
                    patch.object(workflow, "evaluate_variants", side_effect=AssertionError("No fitting before freeze")):
                workflow.prepare(source, preparation, frozen, processed, results)
            self.assertFalse((results / "evaluation").exists())
            workflow.load_frozen(processed, results)
            output = workflow.run(processed, results)
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "complete")
            cohort = pd.read_csv(processed / "observations.csv")
            predictions = pd.read_csv(output / "model_predictions.csv")
            self.assertEqual(len(predictions), len(cohort) * 9)
            validate_all_predictions(predictions, cohort)
            bootstraps = pd.read_csv(output / "bootstrap_metrics.csv")
            self.assertEqual(len(bootstraps), 24)
            for _, group in bootstraps.groupby("variant"):
                self.assertEqual(group.participant_draw.tolist(), bootstraps.loc[bootstraps.variant.eq("original"), "participant_draw"].tolist())
            self.assertEqual(len(pd.read_csv(output / "feature_comparisons.csv")), 18)
            self.assertEqual(len(pd.read_csv(output / "model_fold_metrics.csv")), 45)
            self.assertEqual(len(pd.read_csv(output / "onset_summary.csv")), 9)
            self.assertTrue((results / "REPORT.md").exists())
            for name in ["daily_performance", "paired_differences", "onset_events"]:
                self.assertGreater((output / f"{name}.png").stat().st_size, 1000)
                self.assertGreater((output / f"{name}.pdf").stat().st_size, 1000)
            changed = predictions.copy()
            changed.loc[changed.variant.eq("deviations") & changed.input_set.eq("combined"), "outer_test_fold"] = 99
            with self.assertRaises(ValueError):
                validate_all_predictions(changed, cohort)
            with patch.object(workflow, "source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "source files changed"):
                    workflow.load_frozen(processed, results)
            with (processed / "observations.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen input or output changed"):
                workflow.load_frozen(processed, results)


if __name__ == "__main__":
    unittest.main()
