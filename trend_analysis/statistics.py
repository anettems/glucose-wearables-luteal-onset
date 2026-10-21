"""Participant-paired uncertainty across modalities and feature variants."""

import pandas as pd

from daily_metrics import bootstrap_metrics, daily_metrics, interval_record, summary_tables, validate_paired_predictions
from model_inputs import FEATURE_SETS
import primary_protocol as config
from trend_analysis.protocol import VARIANTS


def validate_all_predictions(predictions, observations):
    if set(predictions.variant) != set(VARIANTS):
        raise ValueError("Expected all three feature variants")
    for variant in VARIANTS:
        group = predictions.loc[predictions.variant.eq(variant)]
        validate_paired_predictions(group, observations)
    expected = predictions.loc[predictions.variant.eq("original") & predictions.input_set.eq("wearable")]
    expected = expected[config.KEY + ["outer_test_fold"]].sort_values(config.KEY).reset_index(drop=True)
    for _, group in predictions.groupby(["variant", "input_set"]):
        actual = group[config.KEY + ["outer_test_fold"]].sort_values(config.KEY).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError("Feature variants use different held-out keys or participant folds")


def bootstrap_tables(predictions, observations):
    validate_all_predictions(predictions, observations)
    bootstraps, summaries, paired = [], [], []
    draws = None
    for variant in VARIANTS:
        print(f"Bootstrap: {variant}, {config.BOOTSTRAP_RESAMPLES} participant resamples", flush=True)
        predicted = predictions.loc[predictions.variant.eq(variant)]
        bootstrap = bootstrap_metrics(predicted)
        if draws is not None and not bootstrap.participant_draw.equals(draws):
            raise ValueError("Bootstrap draws differ across feature variants")
        draws = bootstrap.participant_draw.copy()
        summary, effect = summary_tables(predicted, bootstrap)
        bootstraps.append(bootstrap.assign(variant=variant))
        summaries.append(summary.assign(variant=variant))
        paired.append(effect.assign(variant=variant))
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    return {"bootstrap_metrics": bootstrap, "model_summary": summary,
            "paired_comparison": pd.concat(paired, ignore_index=True),
            "feature_comparisons": feature_comparisons(summary, bootstrap),
            "feature_fold_differences": fold_differences(predictions)}


def fold_differences(predictions):
    estimates = {}
    for key, group in predictions.groupby(["variant", "input_set", "outer_test_fold"]):
        estimates[key] = daily_metrics(group)
    rows = []
    for added, reference in [("deviations", "original"), ("deviations_trends", "original"),
                             ("deviations_trends", "deviations")]:
        for input_set in FEATURE_SETS:
            for fold in range(1, config.OUTER_FOLDS + 1):
                differences = {metric: estimates[added, input_set, fold][metric] - estimates[reference, input_set, fold][metric]
                               for metric in ["average_precision", "roc_auc"]}
                rows.append({"comparison": f"{added}_minus_{reference}", "input_set": input_set,
                             "outer_test_fold": fold, **differences})
    return pd.DataFrame(rows)


def feature_comparisons(summary, bootstrap):
    rows = []
    estimates = summary.set_index(["variant", "input_set", "metric"]).estimate
    comparisons = [("deviations", "original"), ("deviations_trends", "original"),
                   ("deviations_trends", "deviations")]
    for added, reference in comparisons:
        first = bootstrap.loc[bootstrap.variant.eq(added)].sort_values("replicate")
        second = bootstrap.loc[bootstrap.variant.eq(reference)].sort_values("replicate")
        if not first.participant_draw.reset_index(drop=True).equals(second.participant_draw.reset_index(drop=True)):
            raise ValueError("Feature differences require paired participant draws")
        for input_set in FEATURE_SETS:
            for metric in ["average_precision", "roc_auc"]:
                differences = first[f"{input_set}_{metric}"].to_numpy() - second[f"{input_set}_{metric}"].to_numpy()
                rows.append({"comparison": f"{added}_minus_{reference}", "input_set": input_set, "metric": metric,
                             "estimate": float(estimates[added, input_set, metric] - estimates[reference, input_set, metric]),
                             **interval_record(differences)})
    return pd.DataFrame(rows)
