# Exploratory personal deviations and trends

This extension evaluates whether personal deviations and recent trends help identify
daily luteal status and its onset. It was requested after the completed primary
evaluation had been inspected. Results must be described as exploratory.

## Commands

From the thesis workspace, using the existing project environment:

```powershell
code/.venv/Scripts/python.exe -m unittest discover -s code/trend_analysis/tests -v

# Prepare histories, audit availability, verify reused folds and save the freeze.
code/.venv/Scripts/python.exe code/trend_analysis/main.py --prepare

# Verify the freeze, fit nine models through the saved folds and generate a report.
code/.venv/Scripts/python.exe code/trend_analysis/main.py --evaluate
```

Preparation writes fresh outputs to `code/data/processed/trend_analysis_2022_v1`
and `results/trend_analysis_2022_v1`. Evaluation writes its fresh `evaluation/`
subdirectory and `REPORT.md` within the latter. Output directories must be empty
when created. CLI options support different destinations for subsequent versions.
Raw data and the primary source, protocol and outputs are protected.

No additional dependencies are needed. The existing modelling and reporting
packages are used, with their versions saved before evaluation.

## Feature and history definitions

The same five raw features and 14:00 timing rules are retained. Historical values
are reconstructed from the quality-approved processed exports before common-cohort
or label filtering. A valid wearable measurement can therefore enter its history
even when glucose was unavailable that day. Current-day reconstructed values are
checked against all original eligible observations.

For aligned prediction day `t` and each biomarker:

- Reference: median of finite observed values with `t-14 <= day < t`, requiring
  at least seven values on distinct days.
- Deviation: current aligned value minus that reference, in the feature's original
  units. HRV remains on its natural-log scale.
- Trend: ordinary least-squares slope of observed values with `t-7 <= day < t`,
  requiring at least four distinct days and using actual calendar-day spacing.
- Both histories exclude the current aligned value. An aligned glucose value on
  day `s` uses glucose from `s-1`, so the reference never includes the current
  predictor's CGM day. No interpolation, forward filling or median replacement
  occurs within histories. Absent observations remain absent.

The deployment assumption permits a new participant's earlier unlabelled sensor
measurements to inform their later predictions. Reference and trend calculations
never use phase labels, future observations, other participants, or a full-recording
personal average. This differs from the primary model's population-only features.
The rolling reference describes recent levels and can include luteal days.

## Shared cohort and model comparison

A day must satisfy primary eligibility, and both glucose features plus at least two
wearables must have a current value, reference and slope. This common rule is applied
to all nine models, including the original-feature controls. A third wearable with
insufficient current/history data may have missing derived predictors; its available
raw values and slopes are retained. Remaining missing predictors are replaced with
the corresponding training-participant median. No missingness flags are predictors.

Three feature variants are evaluated for glucose-only, wearable-only and combined:

1. Original raw features (2/3/5 predictors), refitted on the shared history cohort.
2. Original features plus personal deviations (4/6/10 predictors).
3. Original features, deviations and trends (6/9/15 predictors).

The primary saved participant and validation assignments are restricted to retained
participants, then verified for outcome classes, evaluable events and observed
training predictors. All variants use identical training, validation and test keys.
Logistic regression, training-only scaling, the C grid, validation AP selection,
alert tuning/matching and percentile participant-bootstrap rules are inherited from
the frozen primary protocol. No folds or hyperparameters are selected using test
performance. Histories are not restarted at missing prediction days or fold boundaries.

## Reproducibility and outputs

`--prepare` saves the exploratory protocol and timestamped freeze before extension
predictions, including software versions, source/input/processed-file hashes,
restricted participant assignments, fold feasibility and availability audits.
`--evaluate` refuses changed sources, parameters, versions or saved inputs.

The evaluation saves out-of-fold and validation predictions, tuning scores, selected
settings, fitted replacement/scaling/coefficient parameters, daily and event fold
metrics, event rows, calibration bins, probability summaries, and 2,000 bootstrap
draws per variant. Draws are identical across all nine models. Paired AP and AUC
changes compare each feature addition, and combined versus wearable AP is reported
for each variant. Confidence intervals condition on the fitted cross-validation
models and are not adjusted for multiple exploratory comparisons.

`REPORT.md` and PNG/vector-PDF figures present coverage, daily performance, paired
differences and onset counts with matched-event timing. A status file distinguishes
complete from failed runs. The final output manifest hashes the report and outputs.

Modules: `features.py` builds histories and audits; `models.py` handles generic
training-only preprocessing and tuning; `evaluation.py` evaluates saved folds;
`statistics.py` handles paired bootstrap comparisons; `workflow.py` manages the
freeze and files; `report.py` generates the report; `main.py` is the entry point.
