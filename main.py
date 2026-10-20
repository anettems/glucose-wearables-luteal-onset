"""Prepare the 2022 inputs or run the locked primary model evaluation.

The default command rebuilds the one-day 14:00 analysis inputs. Use
--evaluate-primary to run only the pre-specified participant-held-out
logistic-regression evaluation from existing prepared inputs.
"""

from pathlib import Path

import preprocessing
from model_inputs import prepare_datasets
from primary_evaluation import run_primary_evaluation


def parse_main_arguments(argv=None):
    """Expose preparation and evaluation options together, including in help."""
    parser = preprocessing.create_argument_parser(prepare=True)
    parser.description = __doc__
    parser.add_argument("--evaluate-primary", action="store_true",
                        help="Evaluate the fixed models using existing prepared inputs")
    workspace = Path(__file__).resolve().parents[1]
    parser.add_argument("--evaluation-dir", type=Path,
                        default=workspace / "results/primary_evaluation_2022_v4",
                        help="Empty directory for protocol, folds, predictions and metrics")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_main_arguments(argv)
    if args.evaluate_primary:
        result = run_primary_evaluation(
            args.processed_dir, args.report_dir, args.evaluation_dir, raw_dir=args.raw_dir,
        )
        print(f"Primary evaluation complete: {result['participants']} participants, "
              f"{result['days']} held-out days.")
        print(f"Results: {args.evaluation_dir.resolve()}")
        return result

    args.temporal_review = True
    preprocessing.run_audit(args)
    data = prepare_datasets(args.processed_dir, args.report_dir)
    print(f"Features ready: {data.id.nunique()} participants, {len(data)} days, "
          f"{int(data.exact_luteal_onset.eq(1).sum())} luteal starts.")
    print(f"Dataset: {args.processed_dir / 'observations_2022.csv'}")
    print("Wearable replacement must be fitted separately on each training split.")
    return None


if __name__ == "__main__":
    main()
