"""Prepare/freeze or evaluate the exploratory personal-history comparison."""

import argparse
from pathlib import Path
import sys

# Permit the requested standalone folder entry point from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trend_analysis.workflow import prepare, run


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/primary_1400_2022")
    parser.add_argument("--preparation-dir", type=Path, default=root.parent / "results/primary_1400_2022")
    parser.add_argument("--primary-freeze", type=Path, default=root.parent / "results/primary_protocol_2022_v4")
    parser.add_argument("--processed-output-dir", type=Path, default=root / "data/processed/trend_analysis_2022_v1")
    parser.add_argument("--results-dir", type=Path, default=root.parent / "results/trend_analysis_2022_v1")
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(args.processed_dir, args.preparation_dir, args.primary_freeze,
                      args.processed_output_dir, args.results_dir).to_string(index=False))
        print(f"Exploratory protocol and inputs frozen: {args.results_dir.resolve()}")
    else:
        print(f"Exploratory evaluation complete: {run(args.processed_output_dir, args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
