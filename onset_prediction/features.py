#### Reuse the implemented sensor summaries and personal-history calculations unchanged.
#### Separate sensor/history eligibility from current phase labels and future outcomes.
#### Check reconstructed values against the earlier prepared features before saving new data.

import numpy as np
import pandas as pd

from model_inputs import FEATURE_SETS, GLUCOSE_FEATURES, WEARABLE_FEATURES
from onset_prediction.protocol import KEY, TARGET, feature_columns
from trend_analysis.features import add_personal_features, load_quality_history
from trend_analysis import protocol as history_protocol


def compare_features(actual, expected, columns):
    #### Require identical feature values for all previously prepared observation keys.
    joined = expected[KEY + columns].merge(actual[KEY + columns], on=KEY, how="left",
                                           validate="one_to_one", suffixes=("_old", "_new"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Earlier feature keys are absent from reconstructed histories")
    for column in columns:
        if not np.allclose(joined[f"{column}_old"], joined[f"{column}_new"],
                           equal_nan=True, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Reconstructed feature differs from earlier analysis: {column}")


def build_features(processed, configuration, history_reference):
    #### Reconstruct quality-approved histories and confirm that their definitions are unchanged.
    settings = (history_protocol.REFERENCE_DAYS, history_protocol.REFERENCE_MINIMUM,
                history_protocol.TREND_DAYS, history_protocol.TREND_MINIMUM)
    if settings != (14, 7, 7, 4):
        raise ValueError("Existing personal-history definitions changed")
    if configuration.get("prediction_cutoff") != "14:00:00":
        raise ValueError("Expected the existing 14:00 preparation")
    history = load_quality_history(processed, configuration)
    enriched = add_personal_features(history)
    compare_features(enriched, pd.read_csv(processed / "observations_2022.csv"), FEATURE_SETS["combined"])
    compare_features(enriched, pd.read_csv(history_reference / "observations.csv"), feature_columns("combined"))
    return enriched


def select_cohort(alignment, enriched, outcomes):
    #### Apply existing sensor/history requirements before excluding unknown future outcomes.
    timing = ["aligned_inputs_available", "prior_day_source_day", "prediction_cutoff_seconds",
              "main_sleep_end_seconds", "valid_wearable_count"]
    audit = outcomes.merge(alignment[KEY + timing], on=KEY, validate="one_to_one")
    audit = audit.merge(enriched, on=KEY, validate="one_to_one")
    ready = {}
    for feature in FEATURE_SETS["combined"]:
        columns = [feature, f"{feature}__deviation", f"{feature}__slope"]
        ready[feature] = audit[columns].notna().all(axis=1)
        audit[f"{feature}__history_ready"] = ready[feature]
    audit["glucose_history_ready"] = np.logical_and.reduce([ready[name] for name in GLUCOSE_FEATURES])
    audit["wearable_histories_ready"] = sum(ready[name].astype(int) for name in WEARABLE_FEATURES)
    audit["history_eligible"] = (audit.aligned_inputs_available & audit.glucose_history_ready
                                  & audit.wearable_histories_ready.ge(2))
    audit["supervised_eligible"] = audit.history_eligible & audit[TARGET].notna()
    cohort = audit.loc[audit.supervised_eligible].copy().reset_index(drop=True)
    validate_cohort(cohort)
    return audit, cohort


def validate_cohort(data):
    #### Reject missing required inputs, unknown outcomes and timing errors without imputing.
    if data.empty or not data[TARGET].isin([0, 1]).all():
        raise ValueError("No eligible cohort or an unknown supervised outcome")
    if np.isinf(data[feature_columns("combined")].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature value")
    if data[feature_columns("glucose")].isna().any().any():
        raise ValueError("Missing required glucose inputs")
    if not data.prior_day_source_day.eq(data.day_in_study - 1).all():
        raise ValueError("Glucose timing differs from the preceding calendar day")
    if not data.main_sleep_end_seconds.le(data.prediction_cutoff_seconds).all():
        raise ValueError("A selected sleep ends after the prediction cutoff")
    for name in FEATURE_SETS["combined"]:
        for kind in ("reference", "trend"):
            last = data[f"{name}__last_{kind}_day"].dropna()
            if not last.lt(data.loc[last.index, "day_in_study"]).all():
                raise ValueError("History includes the current day or future data")
