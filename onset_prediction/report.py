#### Write a concise report from logistic probability outputs, without changing thesis files.
#### Present pooled estimates and paired confidence intervals beside calibration summaries.
#### The report concerns the three-day outcome, with no alert or threshold-based analysis.


def write_report(output, tables):
    #### Save the cohort, selected settings, pooled metrics and paired glucose comparison.
    summary, paired = tables["model_summary"], tables["paired_comparison"].iloc[0]
    probability = tables["probability_summary"]
    first = probability.iloc[0]
    lines = ["# Logistic regression: three-day luteal-onset probabilities", "",
             f"Cohort: {int(first.participants)} participants and {int(first.days)} identical held-out days; "
             f"{int(first.positive_days)} positive days ({first.prevalence:.2%}).", "",
             "All models use current values, personal deviations and seven-day trends.",
             "C was selected by mean validation AP, with preprocessing fitted on training participants only.", "",
             "| Input set | Average precision [95% CI] | ROC-AUC [95% CI] | Brier score [95% CI] |",
             "| --- | --- | --- | --- |"]
    for name in ("glucose", "wearable", "combined"):
        formatted = []
        for metric in ("average_precision", "roc_auc", "brier_score"):
            row = summary.loc[summary.input_set.eq(name) & summary.metric.eq(metric)].iloc[0]
            formatted.append(f"{row.estimate:.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]")
        lines.append(f"| {name} | " + " | ".join(formatted) + " |")
    lines += ["", f"Combined minus wearable-only AP: **{paired.estimate:.4f}** "
              f"(95% CI {paired.ci_lower:.4f} to {paired.ci_upper:.4f}).", "",
              "AP is compared with the positive-day proportion above. Lower Brier scores are better.",
              "Intervals use 2,000 identical participant draws across input sets and are conditional",
              "on the fitted cross-validation models. The saved draws can be reused for random forest.", "",
              "| Input set | Mean predicted probability | Observed positive frequency |",
              "| --- | ---: | ---: |"]
    for row in probability.itertuples(index=False):
        lines.append(f"| {row.input_set} | {row.mean_probability:.4f} | {row.prevalence:.4f} |")
    lines += ["", "Overall agreement of means does not establish calibration across probability ranges.",
              "See calibration_bins.csv for all ten fixed bins and their observation counts.", "",
              "| Test fold | Glucose C | Wearable C | Combined C |", "| --- | ---: | ---: | ---: |"]
    settings = tables["selected_settings"].pivot(index="test_fold", columns="input_set", values="C")
    for fold, row in settings.iterrows():
        lines.append(f"| {fold} | {row.glucose:g} | {row.wearable:g} | {row.combined:g} |")
    lines += ["", "Validation probabilities for every candidate C, fitted medians/scales/coefficients,",
              "15 fitted development models and all test probabilities are saved alongside this report.",
              "No probability thresholds, alerts or random forests were evaluated.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
