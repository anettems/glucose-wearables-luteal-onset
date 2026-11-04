#### Compare the six existing glucose predictors on positive and negative days within people.
#### Read the unchanged shared 2022 model cohort and verify it against frozen model inputs.
#### Summarise each person's class means, medians and feature-ranking overlap before giving
#### participants equal weight. Save descriptive plots, counts and participant-bootstrap
#### intervals in fresh folders. Labels are used only for this assessment, never new predictors.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onset_assessment.analysis import RESAMPLES, SEED, bootstrap_means, direction_counts
from onset_assessment.main import check_destination, write_json
from onset_prediction import protocol
from onset_prediction.freezing import read_json, verify_hashes
from onset_prediction.outcomes import validate_keys
from preprocessing import file_sha256


FEATURES = {
    "glucose_median_mmol_l": ("Glucose median: current value", "mmol/L"),
    "glucose_cv_percent": ("Glucose variability: current value", "percentage points"),
    "glucose_median_mmol_l__deviation": ("Glucose median: personal deviation", "mmol/L"),
    "glucose_cv_percent__deviation": ("Glucose variability: personal deviation", "percentage points"),
    "glucose_median_mmol_l__slope": ("Glucose median: seven-day trend", "mmol/L per day"),
    "glucose_cv_percent__slope": ("Glucose variability: seven-day trend", "percentage points per day"),
}


def validate_data(data):
    #### Require the exact complete glucose feature set and valid binary 2022 observation keys.
    validate_keys(data)
    if set(FEATURES) != set(protocol.feature_columns("glucose")):
        raise ValueError("The agreed glucose feature definition has changed")
    if not data[protocol.TARGET].isin([0, 1]).all():
        raise ValueError("Expected known binary outcomes")
    if not np.isfinite(data[list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Expected all six glucose predictors to be observed and finite")


def participant_coverage(data):
    #### Record every model participant, including people without both outcome classes.
    validate_data(data)
    counts = data.groupby("id")[protocol.TARGET].agg(days="size", positive_days="sum").reset_index()
    counts["negative_days"] = counts.days - counts.positive_days
    counts["paired_eligible"] = counts.positive_days.gt(0) & counts.negative_days.gt(0)
    counts["reason"] = np.select([counts.positive_days.eq(0), counts.negative_days.eq(0)],
                                 ["no_positive_days", "no_negative_days"], default="both_classes")
    return counts


def participant_comparisons(data, coverage):
    #### Compare classes within each person; retain one fixed higher-value ranking direction.
    rows = []
    eligible = data.loc[data.id.isin(coverage.loc[coverage.paired_eligible, "id"])]
    for person, group in eligible.groupby("id", sort=True):
        positive, negative = group.loc[group[protocol.TARGET].eq(1)], group.loc[group[protocol.TARGET].eq(0)]
        for feature in FEATURES:
            pos, neg = positive[feature], negative[feature]
            rows.append({"id": person, "feature": feature, "positive_days": len(pos),
                         "negative_days": len(neg), "positive_mean": pos.mean(), "negative_mean": neg.mean(),
                         "mean_difference": pos.mean() - neg.mean(),
                         "positive_median": pos.median(), "negative_median": neg.median(),
                         "median_difference": pos.median() - neg.median(),
                         "higher_value_auc": roc_auc_score(group[protocol.TARGET], group[feature])})
    if not rows:
        raise ValueError("No participant has both positive and negative days")
    return eligible, pd.DataFrame(rows)


def summary_table(comparisons):
    #### Average participant differences and ranking overlap, with equal participant weights.
    rows = []
    columns = ["mean_difference", "median_difference", "higher_value_auc"]
    for feature in FEATURES:
        data = comparisons.loc[comparisons.feature.eq(feature)].sort_values("id")
        intervals = bootstrap_means(data[columns].to_numpy())
        row = {"feature": feature, "participants": len(data),
               "positive_days": int(data.positive_days.sum()), "negative_days": int(data.negative_days.sum()),
               "positive_mean": data.positive_mean.mean(), "negative_mean": data.negative_mean.mean(),
               **direction_counts(data.mean_difference)}
        for index, column in enumerate(columns):
            row.update({column: data[column].mean(),
                        f"{column}_ci_lower": intervals.iloc[index].ci_lower,
                        f"{column}_ci_upper": intervals.iloc[index].ci_upper})
        rows.append(row)
    return pd.DataFrame(rows)


def centred_distributions(data, comparisons):
    #### Remove each person's negative-day mean for display and balance people within each class.
    long = data.melt(id_vars=protocol.KEY + [protocol.TARGET], value_vars=list(FEATURES),
                     var_name="feature", value_name="value")
    columns = ["id", "feature", "negative_mean", "positive_days", "negative_days"]
    long = long.merge(comparisons[columns], on=["id", "feature"], how="left", validate="many_to_one")
    long["display_value"] = long.value - long.negative_mean
    counts = np.where(long[protocol.TARGET].eq(1), long.positive_days, long.negative_days)
    long["display_weight"] = 1 / (comparisons.id.nunique() * counts)
    return long


def load_data(processed, preparation, frozen):
    #### Check the saved cohort's hashes, labels and predictors against the original model freeze.
    manifest = read_json(preparation / "manifest.json")
    if manifest.get("feasible") is not True:
        raise ValueError("Expected a completed feasible onset preparation")
    verify_hashes(manifest["files"])
    verify_hashes(read_json(frozen / "freeze_manifest.json")["files"], frozen)
    source, reference = processed / "observations.csv", frozen / "observations.csv"
    if manifest["files"].get(str(source)) != file_sha256(source):
        raise ValueError("The input cohort is not the verified prepared cohort")
    data, original = pd.read_csv(source), pd.read_csv(reference)
    validate_data(data)
    keys = protocol.KEY + [protocol.TARGET]
    first, second = [frame.sort_values(protocol.KEY).reset_index(drop=True) for frame in (data, original)]
    if not first[keys].equals(second[keys]) or not np.allclose(
            first[list(FEATURES)], second[list(FEATURES)], atol=1e-12, rtol=1e-12):
        raise ValueError("Assessment observations differ from the actual model cohort or predictors")
    paths = [source, reference, preparation / "manifest.json", frozen / "freeze_manifest.json"]
    return data, {str(path): file_sha256(path) for path in paths}


def main_findings(summary):
    #### Describe the observed contrast sizes without equating feature ranking with model accuracy.
    mean_crosses_zero = summary.mean_difference_ci_lower.le(0) & summary.mean_difference_ci_upper.ge(0)
    auc_crosses_half = summary.higher_value_auc_ci_lower.le(0.5) & summary.higher_value_auc_ci_upper.ge(0.5)
    current = summary.loc[summary.feature.eq("glucose_median_mmol_l")].iloc[0]
    deviation = summary.loc[summary.feature.eq("glucose_median_mmol_l__deviation")].iloc[0]
    trend = summary.loc[summary.feature.eq("glucose_median_mmol_l__slope")].iloc[0]
    lines = ["## Main findings", "",
             f"- {int(mean_crosses_zero.sum())} of six mean-difference intervals include zero. "
             f"{int(auc_crosses_half.sum())} of six within-participant feature-AUC intervals include 0.50.",
             f"- Current median glucose is lower on positive days in {int(current.negative_participants)} "
             f"of {int(current.participants)} participants, but the average difference is only "
             f"{current.mean_difference:+.3f} mmol/L and its interval spans both directions.",
             f"- Personal glucose deviation is lower in {int(deviation.negative_participants)} "
             f"participants and higher in {int(deviation.positive_participants)}. "
             "Subtracting the existing 14-day reference therefore does not reveal a universal direction.",
             f"- The seven-day glucose-median trend has mean within-participant AUC "
             f"{trend.higher_value_auc:.3f}. Its direction alone provides little separation in these data."]
    median_signal = summary.loc[(summary.median_difference_ci_lower > 0) | (summary.median_difference_ci_upper < 0)]
    for row in median_signal.itertuples(index=False):
        lines.append(f"- A secondary median-based comparison for {FEATURES[row.feature][0].lower()} "
                     f"gives {row.median_difference:+.3f} [{row.median_difference_ci_lower:+.3f}, "
                     f"{row.median_difference_ci_upper:+.3f}]. Its interval excludes zero; "
                     "this isolated result among related summaries does not establish reliable prediction.")
    return lines + ["", "The distribution plots show extensive class overlap. These results help explain weak",
                    "glucose-only performance, while leaving possible joint/non-monotonic information unresolved.", ""]


def write_report(results, coverage, summary):
    #### Explain native-unit differences, participant consistency and feature-value overlap.
    included = coverage.loc[coverage.paired_eligible]
    excluded = coverage.loc[~coverage.paired_eligible]
    lines = ["# Glucose features on positive and negative days within participants", "",
             f"The unchanged model cohort contains {len(coverage)} participants and {int(coverage.days.sum()):,} days.",
             f"The within-participant comparison includes {len(included)} participants, "
             f"{int(included.positive_days.sum())} positive days and {int(included.negative_days.sum())} negative days.",
             f"{len(excluded)} participants lack one outcome class and cannot contribute a paired difference.",
             "Their original days remain in the modelling cohort; exclusions here concern this comparison only.", "",
             "Positive: a supplied reference luteal start occurs on one of the next three calendar days.",
             "Negative: all three future days have known no-start status. Negative days can fall anywhere",
             "in the cycle, including an established luteal phase. No retrospectively selected control phase is used.", ""]
    lines += main_findings(summary)
    lines += ["## Mean differences", "",
             "First calculate positive and negative means separately within each participant; subtract",
             "negative from positive, then average the participant differences with equal weights.",
             "Intervals are 95% percentile intervals from 2,000 whole-participant resamples (seed 20261021).", "",
             "| Feature | Positive mean | Negative mean | Difference [95% interval] | Lower / higher / unchanged participants |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        label, unit = FEATURES[row.feature]
        lines.append(f"| {label} ({unit}) | {row.positive_mean:.3f} | {row.negative_mean:.3f} | "
                     f"{row.mean_difference:+.3f} [{row.mean_difference_ci_lower:+.3f}, "
                     f"{row.mean_difference_ci_upper:+.3f}] | {row.negative_participants} / "
                     f"{row.positive_participants} / {row.zero_participants} |")
    lines += ["", "Current glucose summarises the completed preceding calendar day. Personal deviation is",
              "current value minus the prior 14-calendar-day median, requiring seven observed days.",
              "Trend is the previous seven-calendar-day regression slope, requiring four observed days",
              "and excluding the current aligned value. Model features are used exactly as saved.", "",
              "## Separation and overlap", "",
              "Within each participant, AUC is the probability that a randomly selected positive day",
              "has a higher feature value than a randomly selected negative day, with half credit for ties.",
              "Average these AUCs equally across participants. 0.50 means no directional ranking;",
              "below 0.50 means lower values tend to occur on positive days. The direction is never",
              "flipped separately per participant. This is a descriptive single-feature comparison,",
              "not cross-validated model performance or a test of joint/nonlinear feature patterns.", "",
              "| Feature | Mean within-participant AUC [95% interval] | Mean within-participant median difference |",
              "| --- | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {FEATURES[row.feature][0]} | {row.higher_value_auc:.3f} "
                     f"[{row.higher_value_auc_ci_lower:.3f}, {row.higher_value_auc_ci_upper:.3f}] | "
                     f"{row.median_difference:+.3f} |")
    lines += ["", "## Figures", "",
              "![Feature distributions](glucose_feature_distributions.png)", "",
              "For display only, subtract each participant's negative-day mean. Each participant",
              "has equal total weight within each class; no participant dominates through more days.",
              "This label-based centring is not used in fitting or predicting, and does not change saved model inputs.", "",
              "![Participant differences](within_participant_differences.png)", "",
              "## Coverage and interpretation", "",
              f"Included positive-day counts range from {int(included.positive_days.min())} to "
              f"{int(included.positive_days.max())} per participant; neighbouring positive days may represent the same onset.",
              "Whole-participant resampling retains this dependence. Intervals describe uncertainty in",
              "the average participant effect, and do not give the range of individual effects.",
              "Mean and median differences provide complementary checks against outlying days.",
              "Near-zero direction differences (within 1e-12 native units) count as unchanged.",
              "Six related features are described; intervals are not adjusted for multiple comparisons.",
              "A small group difference can coexist with extensive overlap. A near-0.50 single-feature",
              "AUC cannot rule out a joint or non-monotonic relationship with upcoming onset.", "",
              "Participant exclusions and day counts: participant_coverage.csv. Every participant's",
              "feature means, medians, differences and AUCs: participant_comparisons.csv.",
              "No models, participant folds, feature definitions or probability thresholds are changed.", ""]
    (results / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_comparison(processed, preparation, frozen, processed_output, results):
    #### Save the comparison design, compute paired summaries, and record all generated outputs.
    paths = [Path(path).resolve() for path in (processed, preparation, frozen, processed_output, results)]
    processed, preparation, frozen, processed_output, results = paths
    check_destination(processed_output, protocol.ROOT / "data/processed")
    check_destination(results, protocol.ROOT.parent / "results")
    data, inputs = load_data(processed, preparation, frozen)
    coverage = participant_coverage(data)
    eligible, comparisons = participant_comparisons(data, coverage)
    processed_output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name("glucose_plots.py"),
                    Path(__file__).with_name("analysis.py"), Path(__file__).with_name("main.py"),
                    Path(__file__).with_name("README.md"),
                    Path(__file__).parent / "tests/test_glucose_comparison.py"]
    write_json(results / "assessment_protocol.json", {
        "version": "2026-11-04-within-participant-glucose-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "target": protocol.TARGET,
        "features": list(FEATURES), "cohort": "unchanged_model_cohort; paired_summaries_require_both_classes",
        "effect": "mean_of_participant_positive_mean_minus_negative_mean",
        "ranking": "mean_participant_auc; fixed_higher_value_direction; half_credit_for_ties",
        "bootstrap_resamples": RESAMPLES, "bootstrap_seed": SEED + 1,
        "direction_zero_tolerance": 1e-12, "imputation": None, "inputs": inputs,
        "sources": {str(path): file_sha256(path) for path in source_files},
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "matplotlib", "scikit-learn")}}})
    summary = summary_table(comparisons)
    distributions = centred_distributions(eligible, comparisons)
    distributions.to_csv(processed_output / "display_distributions.csv", index=False)
    eligible[protocol.KEY + [protocol.TARGET] + list(FEATURES)].to_csv(
        processed_output / "paired_observations.csv", index=False)
    for name, table in {"participant_coverage": coverage, "participant_comparisons": comparisons,
                        "feature_summary": summary}.items():
        table.to_csv(results / f"{name}.csv", index=False)
    from onset_assessment.glucose_plots import plot_distributions, plot_differences
    plot_distributions(distributions, results)
    plot_differences(comparisons, summary, results)
    write_report(results, coverage, summary)
    verify_hashes(inputs)
    outputs = [path for folder in (processed_output, results) for path in folder.iterdir() if path.is_file()]
    write_json(results / "manifest.json", {"status": "complete", "participants": len(coverage),
        "paired_participants": int(coverage.paired_eligible.sum()),
        "files": {str(path): file_sha256(path) for path in outputs}})
    return summary


def main():
    #### Run this comparison separately from the earlier onset-aligned measurement assessment.
    parser = argparse.ArgumentParser(description="Compare actual glucose predictors within participants.")
    parser.add_argument("--processed-dir", type=Path,
                        default=protocol.ROOT / "data/processed/onset_prediction_2022_v1")
    parser.add_argument("--preparation-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_prediction_2022_v1")
    parser.add_argument("--frozen-dir", type=Path,
                        default=protocol.ROOT.parent / "results/onset_logistic_2022_v1/frozen")
    parser.add_argument("--processed-output-dir", type=Path,
                        default=protocol.ROOT / "data/processed/glucose_feature_assessment_2022_v1")
    parser.add_argument("--results-dir", type=Path,
                        default=protocol.ROOT.parent / "results/glucose_feature_assessment_2022_v1")
    args = parser.parse_args()
    result = run_comparison(args.processed_dir, args.preparation_dir, args.frozen_dir,
                            args.processed_output_dir, args.results_dir)
    print(result.to_string(index=False))
    print(f"Report: {(args.results_dir / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
