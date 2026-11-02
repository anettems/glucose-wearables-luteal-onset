#### Run the thesis personal-history analysis with one command and no required options.
#### Use the existing project Python environment and prepared 2022 inputs. Save a fresh
#### primary protocol, verify the original settings, inputs and participant assignments,
#### then prepare, freeze and evaluate the exploratory personal-history comparison.
#### Each run writes new processed data and results, including the report and figures.
#### Existing results and raw data are preserved. Earlier results have already been
#### inspected, so this remains an exploratory analysis rather than a new primary test.

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CODE_DIR = Path(__file__).resolve().parent


def output_paths(code_dir):
    #### Choose matching, unused output folders for this run without replacing old files.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    sequence = 1
    while True:
        suffix = stamp if sequence == 1 else f"{stamp}_{sequence}"
        paths = (
            code_dir / "data/processed" / f"trend_analysis_{suffix}",
            code_dir.parent / "results" / f"trend_analysis_{suffix}",
            code_dir.parent / "results" / f"primary_protocol_{suffix}",
        )
        if not any(path.exists() for path in paths):
            return paths
        sequence += 1


def verify_original_protocol(original, refreshed):
    #### Keep the original model settings, prepared inputs and saved participant folds.
    import pandas as pd
    from preprocessing import file_sha256

    manifest = json.loads((original / "freeze_manifest.json").read_text(encoding="utf-8"))
    names = ["evaluation_protocol.json", "provenance.json", "participant_folds.csv",
             "validation_fold_assignments.csv"]
    for name in names:
        if file_sha256(original / name) != manifest["files"][name]:
            raise ValueError(f"Original frozen file changed: {name}")

    #### Source comments can change; record current sources in the fresh freeze instead.
    saved = json.loads((original / names[0]).read_text(encoding="utf-8"))
    current = json.loads((refreshed / names[0]).read_text(encoding="utf-8"))
    if current != saved:
        raise ValueError("Model settings or software differ from the original protocol")
    saved = json.loads((original / names[1]).read_text(encoding="utf-8"))
    current = json.loads((refreshed / names[1]).read_text(encoding="utf-8"))
    for key in ["inputs", "preparation_config_sha256"]:
        if current[key] != saved[key]:
            raise ValueError("Prepared inputs differ from the original protocol")

    #### Allocation is deterministic, but explicitly reject any changed participant folds.
    for name in names[2:]:
        expected, actual = pd.read_csv(original / name), pd.read_csv(refreshed / name)
        if not actual.equals(expected):
            raise ValueError(f"Participant assignments differ from the original: {name}")


def run_analysis(code_dir=CODE_DIR):
    #### Freeze current sources, then run the existing preparation and evaluation workflow.
    from primary_evaluation import freeze_primary_protocol, write_json
    from preprocessing import file_sha256
    from trend_analysis.workflow import prepare, run

    code_dir = Path(code_dir).resolve()
    processed = code_dir / "data/processed/primary_1400_2022"
    preparation = code_dir.parent / "results/primary_1400_2022"
    original = code_dir.parent / "results/primary_protocol_2022_v4"
    processed_output, results, frozen = output_paths(code_dir)

    print("1/3 Saving a fresh protocol and checking the original folds...", flush=True)
    freeze_primary_protocol(processed, preparation, frozen)
    verify_original_protocol(original, frozen)

    print("2/3 Preparing and freezing personal deviations and trends...", flush=True)
    coverage = prepare(processed, preparation, frozen, processed_output, results)
    print(coverage.to_string(index=False), flush=True)
    write_json(results / "entry_point.json", {
        "entry_point_sha256": file_sha256(Path(__file__)),
        "original_protocol": str(original), "refreshed_protocol": str(frozen),
        "original_settings_inputs_and_folds_verified": True,
        "analysis_status": "exploratory_after_primary_test_inspection",
    })

    print("3/3 Evaluating models and generating the report...", flush=True)
    run(processed_output, results)
    print(f"Analysis complete. Report: {results / 'REPORT.md'}", flush=True)
    return results


def main(argv=None):
    #### Use the project environment automatically, then run the complete analysis.
    parser = argparse.ArgumentParser(
        description="Run the thesis personal-history analysis and report in fresh folders."
    )
    parser.parse_args(argv)
    interpreter = CODE_DIR / ".venv/Scripts/python.exe"
    if not interpreter.is_file():
        parser.exit(1, f"Project Python environment is missing: {interpreter}\n")
    if Path(sys.executable).resolve() != interpreter.resolve():
        return subprocess.run([str(interpreter), "-u", str(Path(__file__).resolve())],
                              check=False).returncode
    try:
        run_analysis()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Analysis stopped: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
