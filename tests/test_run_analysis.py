#### Check the one-command launcher without fitting the full real-data analysis.
#### Verify fresh destinations, protocol/input/fold protection, execution order,
#### failure propagation and automatic selection of the project Python environment.
#### Temporary protocol fixtures use small participant tables and are removed by unittest.

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_analysis as launcher
from preprocessing import file_sha256


class RunAnalysisTests(unittest.TestCase):
    def write_protocol(self, folder):
        #### Create a minimal frozen protocol with hashed inputs and participant tables.
        folder.mkdir()
        (folder / "evaluation_protocol.json").write_text(json.dumps({"seed": 20261020}))
        (folder / "provenance.json").write_text(json.dumps({
            "inputs": {"observations": "input_hash"}, "preparation_config_sha256": "config_hash",
            "sources": {"example.py": "source_hash"},
        }))
        for name in ["participant_folds.csv", "validation_fold_assignments.csv"]:
            pd.DataFrame({"id": [1, 2], "fold": [1, 2]}).to_csv(folder / name, index=False)
        files = {path.name: file_sha256(path) for path in folder.iterdir()}
        (folder / "freeze_manifest.json").write_text(json.dumps({"files": files}))

    def test_protocol_checks_keep_inputs_settings_and_folds_but_record_new_sources(self):
        #### Accept fresh source provenance and reject substantive changes or tampered folds.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            original, fresh = Path(directory) / "original", Path(directory) / "fresh"
            self.write_protocol(original)
            self.write_protocol(fresh)
            path = fresh / "provenance.json"
            record = json.loads(path.read_text())
            record["sources"] = {"example.py": "new_source_hash"}
            path.write_text(json.dumps(record))
            launcher.verify_original_protocol(original, fresh)

            for name, message in [("evaluation_protocol.json", "settings or software"),
                                  ("provenance.json", "Prepared inputs")]:
                path = fresh / name
                previous = path.read_text()
                record = json.loads(previous)
                if name == "provenance.json":
                    record["inputs"] = {}
                else:
                    record["seed"] = 1
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, message):
                    launcher.verify_original_protocol(original, fresh)
                path.write_text(previous)

            for name in ["participant_folds.csv", "validation_fold_assignments.csv"]:
                path = fresh / name
                previous = path.read_text()
                path.write_text("id,fold\n1,2\n2,1\n")
                with self.assertRaisesRegex(ValueError, "assignments differ"):
                    launcher.verify_original_protocol(original, fresh)
                path.write_text(previous)
            (original / "participant_folds.csv").write_text("id,fold\n1,2\n")
            with self.assertRaisesRegex(ValueError, "Original frozen file changed"):
                launcher.verify_original_protocol(original, fresh)

    def test_output_paths_avoid_collision_in_any_destination(self):
        #### Keep a repeated timestamp from overwriting any previous run directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, \
                patch.object(launcher, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "fixed"
            code_dir = Path(directory) / "code"
            first = launcher.output_paths(code_dir)
            first[2].mkdir(parents=True)
            second = launcher.output_paths(code_dir)
            self.assertTrue(all(a != b for a, b in zip(first, second)))
            self.assertTrue(all(not path.exists() for path in second))
            self.assertTrue(first[2].exists())

    def test_pipeline_verifies_and_freezes_before_evaluation_and_stops_on_failure(self):
        #### Require protocol verification and preparation before any model evaluation.
        events = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            code_dir = Path(directory) / "code"
            def prepare(*args):
                #### Stand in for preparation while recording its place in the pipeline.
                events.append("prepare")
                return pd.DataFrame({"participants": [42]})

            with patch("primary_evaluation.freeze_primary_protocol", side_effect=lambda *a: events.append("freeze")), \
                    patch.object(launcher, "verify_original_protocol", side_effect=lambda *a: events.append("verify")) as verify, \
                    patch("trend_analysis.workflow.prepare", side_effect=prepare), \
                    patch("trend_analysis.workflow.run", side_effect=lambda *a: events.append("evaluate")) as evaluate, \
                    patch("primary_evaluation.write_json") as write, redirect_stdout(io.StringIO()):
                results = launcher.run_analysis(code_dir)
                self.assertEqual(events, ["freeze", "verify", "prepare", "evaluate"])
                self.assertEqual(results.parent, code_dir.parent / "results")
                self.assertEqual(write.call_args.args[0], results / "entry_point.json")
                self.assertTrue(write.call_args.args[1]["original_settings_inputs_and_folds_verified"])
                events.clear()
                evaluate.reset_mock()
                verify.side_effect = ValueError("changed folds")
                with self.assertRaisesRegex(ValueError, "changed folds"):
                    launcher.run_analysis(code_dir)
                self.assertEqual(events, ["freeze"])
                evaluate.assert_not_called()

    def test_main_selects_project_interpreter_and_preserves_failure_status(self):
        #### Rerun through the project Python when launched from a different environment.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            code_dir = Path(directory)
            interpreter = code_dir / ".venv/Scripts/python.exe"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            with patch.object(launcher, "CODE_DIR", code_dir), \
                    patch.object(launcher.subprocess, "run", return_value=Mock(returncode=7)) as process, \
                    patch.object(launcher, "run_analysis") as analysis:
                self.assertEqual(launcher.main([]), 7)
                self.assertEqual(process.call_args.args[0][0], str(interpreter))
                analysis.assert_not_called()
                with patch.object(launcher.sys, "executable", str(interpreter)):
                    self.assertEqual(launcher.main([]), 0)
                    analysis.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
