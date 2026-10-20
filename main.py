"""Prepare the 2022 inputs or run the locked primary model evaluation.

The default command rebuilds the one-day 14:00 analysis inputs. Use
--freeze-primary to save the protocol and participant assignments without fitting,
then --evaluate-primary to evaluate from that verified freeze.
"""

from pathlib import Path

import preprocessing
from model_inputs import prepare_datasets
from primary_evaluation import freeze_primary_protocol, run_primary_evaluation


def parse_main_arguments(argv=None):
    """Expose preparation and evaluation options together, including in help."""
    parser = preprocessing.create_argument_parser(prepare=True)
    parser.description = __doc__
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze-primary", action="store_true",
                       help="Save the final protocol, software versions and feasible folds without fitting")
    modes.add_argument("--evaluate-primary", action="store_true",
                       help="Evaluate the fixed models using the verified frozen protocol")
    workspace = Path(__file__).resolve().parents[1]
    parser.add_argument("--evaluation-dir", type=Path,
                        default=workspace / "results/primary_evaluation_2022_v4",
                        help="Empty directory for protocol, folds, predictions and metrics")
    parser.add_argument("--frozen-protocol-dir", type=Path,
                        default=workspace / "results/primary_protocol_2022_v4",
                        help="Empty destination for freezing, or existing frozen protocol for evaluation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_main_arguments(argv)
    if args.freeze_primary:
        result = freeze_primary_protocol(args.processed_dir, args.report_dir, args.frozen_protocol_dir,
                                         raw_dir=args.raw_dir)
        print(f"Protocol frozen: {result['participants']} participants, {result['days']} eligible days.")
        print(f"Frozen protocol: {result['output_dir']}")
        return result
    if args.evaluate_primary:
        result = run_primary_evaluation(
            args.processed_dir, args.report_dir, args.evaluation_dir,
            frozen_protocol_dir=args.frozen_protocol_dir, raw_dir=args.raw_dir,
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
