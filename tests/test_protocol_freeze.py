"""Verify that the final protocol can be frozen without model predictions."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import primary_evaluation as evaluation
import primary_protocol as config
from test_primary_evaluation import freeze_synthetic_protocol, synthetic_observations


class ProtocolFreezeTests(unittest.TestCase):
    def prepare(self, root):
        processed, report = root / "processed", root / "report"
        processed.mkdir()
        report.mkdir()
        data = synthetic_observations()
        data.to_csv(processed / "observations_2022.csv", index=False)
        data[config.KEY + ["luteal_status", "exact_luteal_onset"]].to_csv(
            processed / "prediction_alignment_14_2022.csv", index=False,
        )
        (report / "run_config.json").write_text(json.dumps({"features_ready": True, "prediction_cutoff": "14:00:00"}))
        return processed, report

    def test_freeze_creates_complete_evidence_and_loads_saved_assignments_without_fitting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, report = self.prepare(root)
            with patch.object(evaluation, "evaluate_participant_folds", side_effect=AssertionError("No fitting")):
                frozen = freeze_synthetic_protocol(root, processed, report)
            manifest = json.loads((frozen / "freeze_manifest.json").read_text())
            self.assertEqual(manifest["status"], "frozen")
            self.assertFalse(manifest["predictions_generated"])
            self.assertEqual(set(manifest["files"]), set(evaluation.FREEZE_FILES))
            self.assertEqual(json.loads((frozen / "software_versions.json").read_text()), evaluation.software_versions())
            self.assertFalse((frozen / "model_predictions.csv").exists())
            observations, references, _ = evaluation.load_primary_inputs(processed, report)
            with patch.object(evaluation, "build_fold_assignments", side_effect=AssertionError("No reallocation")):
                test, validation, feasibility = evaluation.load_frozen_protocol(frozen, processed, report, observations, references)
            self.assertEqual(len(test), 10)
            self.assertEqual(len(validation), 40)
            self.assertTrue(feasibility.evaluable_reference_starts.gt(0).all())
            pd.testing.assert_frame_equal(test, pd.read_csv(frozen / "participant_folds.csv"))

    def test_freeze_rejects_changed_settings_sources_inputs_and_saved_payloads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            processed, report = self.prepare(root)
            frozen = freeze_synthetic_protocol(root, processed, report)
            observations, references, _ = evaluation.load_primary_inputs(processed, report)
            def load():
                return evaluation.load_frozen_protocol(frozen, processed, report, observations, references)
            manifest_path = frozen / "freeze_manifest.json"
            original = manifest_path.read_text()
            changed_manifest = json.loads(original)
            changed_manifest["protocol_version"] = "different-version"
            manifest_path.write_text(json.dumps(changed_manifest))
            with self.assertRaisesRegex(ValueError, "manifest"):
                load()
            manifest_path.write_text(original)
            with patch.object(config, "LOGISTIC_C_VALUES", (0.1,)):
                with self.assertRaisesRegex(ValueError, "protocol or software"):
                    load()
            changed = evaluation.provenance_record(processed, report)
            changed["sources"]["daily_models.py"] = "changed"
            with patch.object(evaluation, "provenance_record", return_value=changed):
                with self.assertRaisesRegex(ValueError, "inputs or analysis sources"):
                    load()
            with (processed / "observations_2022.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "inputs or analysis sources"):
                load()
            with (frozen / "participant_folds.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "file changed"):
                load()
            with self.assertRaises(FileExistsError):
                evaluation.freeze_primary_protocol(processed, report, frozen)


if __name__ == "__main__":
    unittest.main()
