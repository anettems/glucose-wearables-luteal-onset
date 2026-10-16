"""Regenerate the agreed 2022 analysis inputs; no models or global imputation."""

import preprocessing
from model_inputs import prepare_datasets


def main(argv=None):
    args = preprocessing.parse_arguments(argv, prepare=True)
    args.temporal_review = True
    preprocessing.run_audit(args)
    data = prepare_datasets(args.processed_dir, args.report_dir)
    print(f"Features ready: {data.id.nunique()} participants, {len(data)} days, "
          f"{int(data.exact_luteal_onset.eq(1).sum())} luteal starts.")
    print(f"Dataset: {args.processed_dir / 'observations_2022.csv'}")
    print("Wearable replacement must be fitted separately on each training split.")


if __name__ == "__main__":
    main()
