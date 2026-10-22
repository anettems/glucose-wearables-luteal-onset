"""Prepare/freeze or evaluate the exploratory within-day glucose extension."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from within_day_analysis import protocol
from within_day_analysis.workflow import prepare, run


def main(argv=None):
    root = protocol.ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--baseline-processed", type=Path, default=root / "data/processed/trend_analysis_2022_v1")
    parser.add_argument("--baseline-results", type=Path, default=root.parent / "results/trend_analysis_2022_v1")
    parser.add_argument("--glucose-path", type=Path, default=root / "data/processed/primary_1400_2022/glucose_2022_audited.csv")
    parser.add_argument("--processed-output", type=Path, default=root / "data/processed/within_day_glucose_2022_v1")
    parser.add_argument("--results-dir", type=Path, default=root.parent / "results/within_day_glucose_2022_v1")
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(args.baseline_processed, args.baseline_results, args.glucose_path,
                      args.processed_output, args.results_dir).to_string(index=False))
        print(f"Within-day protocol and inputs frozen: {args.results_dir.resolve()}")
    else:
        print(f"Within-day evaluation complete: {run(args.processed_output, args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
