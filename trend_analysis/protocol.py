"""Fixed exploratory choices, recorded before extension test predictions."""

from pathlib import Path

from model_inputs import FEATURE_SETS
import primary_evaluation


VERSION = "2026-10-21-personal-history-v1"
REFERENCE_DAYS = 14
REFERENCE_MINIMUM = 7
TREND_DAYS = 7
TREND_MINIMUM = 4
VARIANTS = ("original", "deviations", "deviations_trends")
ROOT = Path(__file__).resolve().parents[1]
HISTORY_INPUTS = ["prediction_alignment_14_2022.csv", "glucose_2022_audited.csv",
                  "hrv_sleep_alignment_2022.csv", "computed_temperature_2022_audited.csv",
                  "sleep_2022_audited.csv", "observations_2022.csv"]


def feature_columns(input_set, variant):
    base = list(FEATURE_SETS[input_set])
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    columns = base.copy()
    if variant != "original":
        columns += [f"{name}__deviation" for name in base]
    if variant == "deviations_trends":
        columns += [f"{name}__slope" for name in base]
    return columns


def record():
    return {
        "version": VERSION, "analysis_status": "exploratory_after_primary_test_inspection",
        "primary_evaluation_rules": primary_evaluation.protocol_record(),
        "reference": {"window_days": REFERENCE_DAYS, "minimum_observed_days": REFERENCE_MINIMUM,
                      "statistic": "median", "bounds": "t-14 <= aligned_prediction_day < t"},
        "trend": {"window_days": TREND_DAYS, "minimum_observed_days": TREND_MINIMUM,
                  "statistic": "ordinary_least_squares_slope_per_calendar_day",
                  "bounds": "t-7 <= aligned_prediction_day < t"},
        "history_source": "quality-approved aligned feature values before common-cohort or label filtering",
        "history_replacement": "none; observed finite values only",
        "personalisation": "earlier unlabelled observations of the assessed participant allowed; no future data",
        "glucose_timing": "aligned feature on day s uses CGM from calendar day s-1",
        "cohort": "primary eligible days; both glucose features and >=2 wearables have current value, reference and slope",
        "missing_predictors": "training-participant median for each selected raw or derived predictor; no missingness indicators",
        "folds": "restrict saved primary participant/test/validation assignments to retained participants; verify feasibility",
        "feature_sets": {v: {name: feature_columns(name, v) for name in FEATURE_SETS} for v in VARIANTS},
        "paired_bootstrap": "same participant draw across all nine models and all feature differences",
        "feature_comparisons": ["deviations_minus_original", "deviations_trends_minus_original",
                                "deviations_trends_minus_deviations"],
        "limitations": "history-conditioned cohort; previously inspected participants; intervals conditional on fitted models",
    }
