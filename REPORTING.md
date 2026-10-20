# Descriptive outputs and model figures

Reporting is a separate, read-only stage after preparation or primary evaluation.
It does not fit models, alter the frozen protocol, allocate folds, replace feature
values, recalibrate predictions, or select settings from test results. The requested
reporting extension adds no new predictors or primary outcome measures.

## Install and run

The optional plotting dependency is Matplotlib. `requirements-reporting.txt` pins
it and its runtime dependencies and includes the existing modelling requirements.
Use the same environment as the frozen evaluation; do not upgrade the frozen
modelling packages. Reporting records all its software versions separately.

```powershell
python -m pip install -r code/requirements-reporting.txt

# Descriptive cohort report, without loading test predictions or fitting models.
code/.venv/Scripts/python.exe code/reporting.py --descriptive-only `
    --output-dir results/descriptive_2022_v4

# After the frozen primary evaluation has completed:
code/.venv/Scripts/python.exe code/reporting.py `
    --evaluation-dir results/primary_evaluation_2022_v4 `
    --output-dir results/model_report_2022_v4
```

The primary evaluation command remains in `MODELLING.md`. Reporting is invoked
explicitly so it can also be rerun from saved results. Use a fresh output directory
for every report. Inputs, raw data, preparation reports, the frozen protocol and
evaluation directories are protected from overlap and overwriting.

## Outputs

Every figure is exported as a PNG and a vector PDF. A Markdown index explains the
outputs and embeds the figures. CSVs contain the numerical values used in them.

| Output | Description |
| --- | --- |
| `participant_coverage.csv`, coverage figure | Aligned and eligible days, calendar spans and gaps, outcome counts and eligibility fractions, including excluded participants |
| `outcome_balance.csv`, outcome figure | Counts and fractions of luteal and non-luteal eligible days |
| `feature_distributions.csv`, histograms | Counts, mean, standard deviation, quartiles and range of observed feature values |
| `feature_missingness.csv`, missingness figure | Observed/missing counts and fractions overall and by participant, before replacement |
| `model_summary.csv`, daily performance figure | Pooled held-out AP and ROC-AUC with saved participant-bootstrap intervals |
| `paired_comparison.csv`, daily performance figure | Combined minus wearable-only AP with its paired interval |
| `model_fold_metrics.csv` | Daily and onset results for checking stability across test folds |
| Discrimination curves | Pooled precision-recall and ROC curves; luteal prevalence is shown for context |
| `calibration_bins.csv`, calibration figure | Ten fixed 0.1-wide probability bins, day and participant counts, mean probability and observed luteal fraction |
| `probability_summary.csv` | Brier score, mean predicted probability and observed prevalence, alongside daily metrics |
| `onset_summary.csv`, onset figure | Matched starts, missed evaluable starts, unmatched alerts, event F1 and mean absolute timing error |
| `onset_reference_events.csv`, `onset_alert_events.csv` | Every saved reference start and alert, including unmatched and non-evaluable events |
| `matched_timing.csv`, timing figure | Absolute timing error and supplementary signed error for matched events |
| `selected_settings.csv`, `onset_fold_metrics.csv` | Saved validation-selected model/alert settings and fold event results |
| `report_manifest.json`, `run_status.json` | Reporting settings, input/source/output hashes, versions, timestamp and completion status |

The descriptive-only mode produces the first four rows, the index and provenance.
It clearly marks model results as pending. Complete model reports require a saved
evaluation with status `complete`, the matching frozen manifest, paired predictions
on exactly the eligible keys, and the saved participant folds. Daily estimates and
intervals are checked against predictions and saved bootstrap draws. Alert events,
counts and timing are checked by applying the saved rules again, without tuning.

## Interpretation and fixed reporting choices

- Descriptive feature summaries are conditional on eligibility: complete glucose
  and at least two wearable features. They do not estimate source-population
  missingness. Source audits and exclusion tables remain in the preparation report.
- Coverage uses calendar days, not consecutive rows. Aligned days are the observed
  prepared alignment rows, not an assumed recording period. White cells in the
  coverage figure lie outside that participant's alignment; grey cells are ineligible.
- Model summaries pool held-out days. Fold results are stability checks. The saved
  95% intervals resample participants and condition on the fitted cross-validation
  models; reporting does not run another bootstrap.
- Calibration bins are fixed before inspecting test outputs: `[0, 0.1)`, ..., `[0.9, 1]`.
  Empty bins have missing means and remain gaps in the plot. No calibration curve,
  bin count or threshold is selected by test performance. The Brier score is the
  mean squared probability error. It measures overall probability accuracy and
  does not isolate calibration. All these summaries weight eligible days equally;
  repeated days within participants are correlated. No independent-day confidence
  intervals or clinical calibration claims are provided.
- Timing is measured only for matched starts and is undefined without matches.
  Misses and unmatched alerts must accompany timing. The signed error (alert day
  minus reference day) describes earlier/later alerts; the protocol's absolute
  timing error remains the principal timing measure. The matching window and
  alert rules are unchanged, including fold-specific validation selection.
- Agreement is with supplied phase labels. These outputs do not establish
  independent hormonal validation, causation or clinical performance.

Implementation: `reporting.py` handles CLI, verification, provenance and output;
`reporting_tables.py` constructs descriptive/calibration tables;
`reporting_figures.py` draws figures. `tests/test_reporting.py` checks these stages
using synthetic cohorts and a synthetic completed evaluation.
