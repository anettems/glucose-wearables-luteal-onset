#### Prepare the new three-day onset outcome and feasibility report from existing audited data.
#### This separate entry point saves paired inputs and participant folds without fitting models.

import argparse
from pathlib import Path
import sys


#### Allow this folder's entry point to be run directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onset_prediction import protocol
from onset_prediction.workflow import run_feasibility


def main(argv=None):
    #### Read preparation paths and run only the requested outcome/fold feasibility stage.
    parser = argparse.ArgumentParser(description="Prepare three-day onset outcomes and audit participant folds.")
    parser.add_argument("--processed-dir", type=Path, default=protocol.ROOT / "data/processed/primary_1400_2022")
    parser.add_argument("--preparation-dir", type=Path, default=protocol.ROOT.parent / "results/primary_1400_2022")
    parser.add_argument("--history-reference-dir", type=Path,
                        default=protocol.ROOT / "data/processed/trend_analysis_2022_v1")
    parser.add_argument("--processed-output-dir", type=Path,
                        default=protocol.ROOT / "data/processed/onset_prediction_2022_v1")
    parser.add_argument("--results-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_prediction_2022_v1")
    args = parser.parse_args(argv)
    summary = run_feasibility(args.processed_dir, args.preparation_dir, args.history_reference_dir,
                              args.processed_output_dir, args.results_dir)
    print(f"Feasible: {summary['feasible']}; {summary['participants']} participants, "
          f"{summary['usable_days']} days, {summary['positive_days']} positive days, "
          f"{summary['distinct_represented_starts']} distinct starts.")
    print(f"Report: {(args.results_dir / 'REPORT.md').resolve()}")
    if not summary["feasible"]:
        raise SystemExit("One or more planned cohorts failed feasibility; see saved counts.")
    return summary


if __name__ == "__main__":
    main()
