#### Record the agreed onset probability design without fitting or selecting any model.
#### Inputs are the existing five measurements and their unchanged personal histories.
#### This stage constructs outcomes and participant folds; modelling is a later stage.

from pathlib import Path

from model_inputs import FEATURE_SETS


VERSION = "2026-11-03-onset-3d-v1"
ROOT = Path(__file__).resolve().parents[1]
KEY = ["id", "study_interval", "day_in_study"]
TARGET = "onset_within_3_days"
SEED = 20261020
TEST_FOLDS = 5
VALIDATION_FOLDS = 3
MAX_SWAP_PASSES = 20
HORIZON = 3
HISTORY_FILES = ["prediction_alignment_14_2022.csv", "glucose_2022_audited.csv",
                 "hrv_sleep_alignment_2022.csv", "computed_temperature_2022_audited.csv",
                 "sleep_2022_audited.csv", "observations_2022.csv"]


def feature_columns(input_set):
    #### Return only the existing current values, personal deviations and trends.
    base = list(FEATURE_SETS[input_set])
    return base + [f"{name}__deviation" for name in base] + [f"{name}__slope" for name in base]


def record():
    #### Save machine-readable choices for this preparation stage and later model development.
    return {
        "version": VERSION, "stage": "feasibility_only", "study_interval": 2022,
        "target": TARGET, "horizon_days": [1, 2, 3], "prediction_cutoff": "14:00:00",
        "reference_start": "luteal_after_recognised_non_luteal_on_previous_calendar_day",
        "positive": "any_exact_start_on_t_plus_1_through_3_even_if_other_days_unknown",
        "negative": "all_three_future_start_statuses_known_zero",
        "unknown": "no_exact_start_and_at_least_one_unknown_future_start_status",
        "cohort": "shared_sensor_history_eligible_days_with_known_future_outcome; no_phase_restriction",
        "reference": {"days": 14, "minimum_observed_days": 7, "statistic": "median"},
        "trend": {"days": 7, "minimum_observed_days": 4, "statistic": "calendar_day_OLS_slope"},
        "history": "past_only; no_interpolation_or_imputation; current_aligned_value_excluded",
        "feature_sets": {name: feature_columns(name) for name in FEATURE_SETS},
        "wearable_replacement": "training_participant_feature_median; not_fitted_in_feasibility",
        "scaling": "training_only_standardisation_for_logistic_regression; none_for_random_forest",
        "models": {
            "logistic_regression": {"C": [0.01, 0.1, 1, 10], "penalty": "l2",
                                    "solver": "lbfgs", "max_iter": 1000, "tol": 1e-10,
                                    "class_weight": None, "tie_break": "smaller_C"},
            "random_forest": {"n_estimators": 100, "max_depth": 5, "max_features": "sqrt",
                              "min_samples_leaf": [1, 5, 10], "criterion": "gini",
                              "min_samples_split": 2, "bootstrap": True, "max_samples": None,
                              "class_weight": None, "random_state": SEED,
                              "tie_break": "larger_min_samples_leaf"}},
        "selection": "mean_participant_disjoint_validation_average_precision",
        "test_folds": TEST_FOLDS, "validation_folds": VALIDATION_FOLDS, "seed": SEED,
        "validation_seeds": [SEED + fold for fold in range(1, TEST_FOLDS + 1)],
        "allocation": "seeded_capacity_balanced_greedy_then_at_most_20_improving_swap_passes",
        "balance": ["participants", "eligible_days", "positive_days", "represented_starts"],
        "minimum_per_cohort": {"positive_days": 1, "negative_days": 1, "represented_starts": 1},
        "primary_metric": "pooled_out_of_fold_average_precision",
        "complementary_metric": "pooled_out_of_fold_ROC_AUC",
        "calibration": "10_fixed_equal_width_probability_bins_and_Brier_score; no_recalibration",
        "primary_effect": "combined_minus_wearable_AP_separately_for_each_model_family",
        "bootstrap": {"resamples": 2000, "seed": SEED + 1, "unit": "participant",
                      "pairing": "same_draw_for_all_six_configurations", "interval": "95_percentile"},
        "alerts": False, "models_fitted": False, "predictions_generated": False,
    }
