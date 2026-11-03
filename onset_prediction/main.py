#### Prepare three-day onset inputs, freeze logistic settings, or evaluate logistic probabilities.
#### Preparation retains the existing features; freezing verifies and saves the cohort and folds.
#### Evaluation fits only logistic regression from a verified freeze and saves paired results.

import argparse
from pathlib import Path
import sys


#### Allow this folder's entry point to be run directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onset_prediction import protocol
from onset_prediction.workflow import run_feasibility


def main(argv=None):
    #### Read stage-specific paths and run preparation, freezing or logistic evaluation.
    parser = argparse.ArgumentParser(description="Prepare or evaluate three-day onset probabilities.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze-logistic", action="store_true")
    modes.add_argument("--evaluate-logistic", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=protocol.ROOT / "data/processed/primary_1400_2022")
    parser.add_argument("--preparation-dir", type=Path, default=protocol.ROOT.parent / "results/primary_1400_2022")
    parser.add_argument("--history-reference-dir", type=Path,
                        default=protocol.ROOT / "data/processed/trend_analysis_2022_v1")
    parser.add_argument("--processed-output-dir", type=Path,
                        default=protocol.ROOT / "data/processed/onset_prediction_2022_v1")
    parser.add_argument("--results-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_prediction_2022_v1")
    parser.add_argument("--frozen-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_logistic_2022_v1/frozen")
    parser.add_argument("--evaluation-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_logistic_2022_v1/evaluation")
    args = parser.parse_args(argv)

    if args.freeze_logistic or args.evaluate_logistic:
        return run_logistic_stage(args)
    return run_preparation(args)


def run_logistic_stage(args):
    #### Import modelling code only for the explicitly requested frozen logistic stage.
    if args.freeze_logistic:
        from onset_prediction.freezing import freeze_logistic
        result = freeze_logistic(args.processed_output_dir, args.results_dir, args.frozen_dir)
        print(f"Logistic protocol frozen: {result['participants']} participants, {result['days']} days.")
        print(f"Frozen inputs: {result['directory']}")
        return result
    from onset_prediction.evaluation import run_logistic
    result = run_logistic(args.frozen_dir, args.evaluation_dir)
    print(f"Logistic evaluation complete: {result.resolve()}")
    return result


def run_preparation(args):
    #### Preserve the original default preparation command and never overwrite earlier outputs.
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
