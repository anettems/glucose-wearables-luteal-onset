"""Paired cohorts, training-only fitting and a frozen end-to-end evaluation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE), str(CODE / "tests"), str(CODE / "trend_analysis/tests")]
import primary_protocol as config
from daily_metrics import validate_paired_predictions
from participant_folds import build_fold_assignments
from test_history import long_observations, history_from
from trend_analysis.features import add_personal_features, select_history_cohort
from trend_analysis.models import fit_and_predict
from within_day_analysis import protocol, workflow
from within_day_analysis.evaluation import feature_effects, validate_predictions


def fixture():
    data = long_observations(10)
    _, cohort = select_history_cohort(data, add_personal_features(history_from(data)))
    references = data.loc[data.exact_luteal_onset.eq(1), config.KEY]
    test, validation, _ = build_fold_assignments(cohort, references)
    parts = []
    for row in cohort.itertuples():
        seconds = np.arange(288) * 300
        values = 5 + row.id / 100 + np.sin(np.arange(288) / 10) / 10
        parts.append(pd.DataFrame({"id": row.id, "study_interval": 2022, "day_in_study": row.day_in_study - 1,
                                   "audit_end_seconds": (row.day_in_study - 1) * 86400 + seconds,
                                   "glucose_value": values, "audit_structurally_usable": True}))
    glucose = pd.concat(parts, ignore_index=True)
    # One day fails night coverage; every model must exclude its next-day key.
    remove = glucose.id.eq(1) & glucose.day_in_study.eq(11) & glucose.audit_end_seconds.lt(11 * 86400 + 21600)
    return cohort, references, test, validation, glucose.loc[~remove]


class WithinDayWorkflowTests(unittest.TestCase):
    def test_added_columns_use_training_only_scaling(self):
        data = long_observations(3)
        for feature in protocol.PATTERN_FEATURES:
            data[feature] = data.id.astype(float)
        training, testing = data.loc[data.id.le(2)], data.loc[data.id.eq(3)].copy()
        testing[protocol.PATTERN_FEATURES] = 10000
        columns = list(protocol.PATTERN_FEATURES)
        _, parameters = fit_and_predict(training, testing, columns, 0.01)
        self.assertEqual(parameters["feature_means"][columns[0]], 1.5)
        self.assertEqual(parameters["replacement_medians"][columns[0]], 1.5)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_and_predict(training, training.iloc[:1], columns, 0.01)

    def test_feature_effects_use_paired_draws(self):
        summaries, draws = [], []
        for variant, offset in [("baseline", 0), ("within_day", 0.1)]:
            for name in ["glucose", "wearable", "combined"]:
                for metric in ["average_precision", "roc_auc"]:
                    summaries.append({"variant": variant, "input_set": name, "metric": metric, "estimate": 0.5 + offset})
            for replicate, noise in enumerate([-0.1, 0, 0.1]):
                row = {"variant": variant, "replicate": replicate, "participant_draw": "[1, 2]"}
                row.update({f"{name}_{metric}": 0.5 + offset + noise for name in ["glucose", "wearable", "combined"]
                            for metric in ["average_precision", "roc_auc"]})
                draws.append(row)
        actual = feature_effects(pd.DataFrame(summaries), pd.DataFrame(draws))
        np.testing.assert_allclose(actual[["estimate", "ci_lower", "ci_upper"]], 0.1)
        draws[-1]["participant_draw"] = "[2, 2]"
        with self.assertRaisesRegex(ValueError, "identical participant"):
            feature_effects(pd.DataFrame(summaries), pd.DataFrame(draws))

    def test_freeze_before_fit_same_cohorts_and_tamper_detection(self):
        baseline, references, test, validation, glucose = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, patch.object(config, "BOOTSTRAP_RESAMPLES", 8):
            root = Path(directory).resolve()
            baseline_processed, baseline_results = root / "baseline_processed", root / "baseline_results"
            baseline_processed.mkdir()
            baseline_results.mkdir()
            glucose_path = baseline_processed / "glucose.csv"
            glucose.to_csv(glucose_path, index=False)
            old_manifest = {"inputs": {str(glucose_path): "verified by baseline loader"}, "files": {}}
            (baseline_results / "freeze_manifest.json").write_text(json.dumps(old_manifest))
            original_bytes = glucose_path.read_bytes()
            processed, results = root / "code/data/processed/within_day", root / "results/within_day"
            with patch.object(workflow.history_workflow.protocol, "ROOT", root / "code"), \
                    patch.object(workflow.history_workflow, "load_frozen", return_value=(baseline, references, test, validation)), \
                    patch.object(workflow, "evaluate_models", side_effect=AssertionError("Fit before freeze")):
                workflow.prepare(baseline_processed, baseline_results, glucose_path, processed, results)
            self.assertFalse((results / "evaluation").exists())
            data, starts, saved_test, saved_validation = workflow.load_frozen(processed, results)
            self.assertEqual(len(data), len(baseline) - 1)
            self.assertFalse((data.id.eq(1) & data.day_in_study.eq(12)).any())
            pd.testing.assert_frame_equal(test, saved_test)
            pd.testing.assert_frame_equal(validation, saved_validation)
            output = workflow.run(processed, results)
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "complete")
            self.assertEqual(glucose_path.read_bytes(), original_bytes)
            predicted = pd.read_csv(output / "model_predictions.csv")
            self.assertEqual(len(predicted), len(data) * 6)
            validate_predictions(predicted, data)
            for _, group in predicted.groupby("variant"):
                validate_paired_predictions(group, data)
            bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
            self.assertEqual(len(bootstrap), 16)
            self.assertEqual(bootstrap.loc[bootstrap.variant.eq("baseline"), "participant_draw"].tolist(),
                             bootstrap.loc[bootstrap.variant.eq("within_day"), "participant_draw"].tolist())
            self.assertEqual(len(pd.read_csv(output / "onset_summary.csv")), 6)
            self.assertTrue((results / "REPORT.md").exists())
            changed = predicted.copy()
            mask = changed.variant.eq("within_day") & changed.input_set.eq("combined")
            changed.loc[mask, "outer_test_fold"] = 99
            with self.assertRaises(ValueError):
                validate_predictions(changed, data)
            with patch.object(workflow, "source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "source files changed"):
                    workflow.load_frozen(processed, results)
            with (processed / "observations.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen input or output changed"):
                workflow.load_frozen(processed, results)


if __name__ == "__main__":
    unittest.main()
