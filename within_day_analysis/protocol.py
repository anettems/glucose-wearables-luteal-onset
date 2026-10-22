"""Fixed choices for the within-day glucose extension, saved before prediction."""

from pathlib import Path

from trend_analysis.protocol import feature_columns as history_columns
from primary_evaluation import protocol_record


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-10-22-within-day-glucose-v1"
VARIANTS = ("baseline", "within_day")
PAIR_SECONDS = 300
PAIR_TOLERANCE_SECONDS = 5
MINIMUM_PAIRS = 24
NIGHT_END_SECONDS = 6 * 3600
MINIMUM_PERIOD_COVERAGE = 0.70
PATTERN_FEATURES = [
    "glucose_iqr_mmol_l", "glucose_p90_minus_median_mmol_l",
    "glucose_mean_absolute_rate_mmol_l_min", "glucose_lag5min_correlation",
    "glucose_night_minus_day_mmol_l",
]
DIAGNOSTICS = ["glucose_pattern_samples", "glucose_pattern_pairs",
               "glucose_night_coverage", "glucose_day_coverage"]


def feature_columns(input_set, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown within-day variant: {variant}")
    columns = history_columns(input_set, "deviations_trends")
    if variant == "within_day" and input_set != "wearable":
        columns += PATTERN_FEATURES
    return columns


def record():
    return {
        "version": VERSION, "status": "exploratory_after_test_inspection",
        "evaluation": protocol_record(), "variants": VARIANTS,
        "baseline": "existing current values, personal deviations and trends; refit on the same narrowed cohort",
        "feature_sets": {v: {s: feature_columns(s, v) for s in ["glucose", "wearable", "combined"]} for v in VARIANTS},
        "glucose_day": "previous calendar day only; retain existing 14:00 cutoff and quality eligibility",
        "distribution": "linear sample quantiles: IQR=P75-P25; upper spread=P90-median; equal reading weights",
        "pairs": {"adjacent_readings_only": True, "target_seconds": PAIR_SECONDS,
                  "tolerance_seconds": PAIR_TOLERANCE_SECONDS, "minimum_pairs": MINIMUM_PAIRS,
                  "rate": "mean(abs(delta glucose)/(elapsed seconds/60)); mmol/L/min",
                  "correlation": "Pearson correlation of earlier/later values in eligible pairs; undefined for constant vectors"},
        "periods": {"night_seconds": [0, NIGHT_END_SECONDS], "day_seconds": [NIGHT_END_SECONDS, 86400],
                    "minimum_coverage_each": MINIMUM_PERIOD_COVERAGE,
                    "coverage": "union of nominal five-minute intervals clipped to each period",
                    "contrast": "night median minus day median; clock periods, not fasting or sleep"},
        "missing_glucose": "no interpolation or replacement; all five pattern features required for every comparison",
        "history": "no additional references or trends for new pattern features",
        "cohort": "subset of frozen personal-history cohort; all variants and input sets use identical keys",
        "folds": "reuse saved participant and validation assignments, restricting retained IDs and rechecking feasibility",
        "bootstrap": "same 2000 participant draws across variants and modalities; intervals conditional on fitted models",
        "limitations": "previous test inspection; observational reference labels; clock periods and availability selection",
    }
