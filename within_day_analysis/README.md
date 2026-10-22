# Exploratory within-day glucose extension

This extension adds glucose distribution, dynamics and time-of-day information to
the completed personal-history analysis. It uses existing logistic regression and
paired daily/onset evaluation. It does not implement Extra Trees or select a model
separately for each participant. No additional dependencies are needed.

From the thesis workspace:

```powershell
code/.venv/Scripts/python.exe -m unittest discover -s code/within_day_analysis/tests -v
code/.venv/Scripts/python.exe code/within_day_analysis/main.py --prepare
code/.venv/Scripts/python.exe code/within_day_analysis/main.py --evaluate
```

`--prepare` verifies the frozen personal-history inputs, then writes new outputs
under `code/data/processed/within_day_glucose_2022_v1` and
`results/within_day_glucose_2022_v1`. Existing directories must be empty. CLI path
options allow a new output version. The old source and generated outputs are not
edited. `--evaluate` verifies saved hashes, software, protocol and folds before
fitting. It creates `evaluation/` and `REPORT.md`. Refitting the baseline on the
same eligible days isolates the feature addition from changes in availability.

## Fixed feature definitions

Each prediction day uses only the preceding calendar day's quality-approved 2022
CGM values in mmol/L. Existing unit, coverage, gap, sleep and 14:00 timing rules
remain required. Each participant/day is processed independently and sorted by
actual time. No values are interpolated, carried forward or replaced.

The five added predictors are:

- IQR: P75 minus P25 using linear sample quantiles.
- Upper spread: P90 minus median, using the same quantile convention.
- Mean absolute rate: mean absolute change divided by elapsed minutes across
  adjacent observed readings 295 to 305 seconds apart (inclusive), in mmol/L/min.
- Five-minute correlation: Pearson correlation of earlier/later values in those
  same pairs. Require at least 24 pairs for both dynamic features. Constant
  vectors have undefined correlation and remain unavailable.
- Night minus day: median during 00:00-06:00 minus median during 06:00-24:00.
  Require at least 70% nominal coverage in each period and readings in both.
  Coverage is the union of intervals from each reading to five minutes later,
  clipped to the period. A reading before 06:00 can contribute nominal coverage
  after 06:00, but its value belongs only to the night median. These clock periods
  do not identify fasting or actual sleep.

The timing tolerance accommodates small timestamp variations seen in the export,
while excluding skipped readings and long gaps. Pairs never cross midnight,
participants or collection intervals. Distribution summaries give each valid
reading equal weight, as in the existing median/CV implementation. Unusual dense
sampling can therefore influence these summaries. The coverage calculation is a
nominal sensor-availability estimate, not proof of continuous valid measurement.

The existing current values, personal deviations and trends remain the baseline.
New predictors are appended directly, with no additional history requirements.
Glucose/combined predictor counts are 6/15 at baseline and 11/20 with the additions;
wearable inputs remain 9 predictors. Both variants require all five new features
so they share the same cohort, including the wearable-only comparison. The audit
saves each feature's availability, sample/pair counts and period coverage. A
constant or incomplete day can be excluded from all comparisons by this rule.

## Evaluation and reproducibility

The saved five participant test folds and three validation folds are reused,
restricted only to retained participants and rechecked for both outcome classes
and evaluable starts. Logistic C values, training-only wearable median replacement
and scaling, validation AP selection and tie rules are inherited. No class weights
or SMOTE are introduced. Wearable fits are reused exactly across the two variants.

The freeze saves protocol, source/input hashes, software versions, feature
availability, common cohort and fold feasibility before any new test prediction.
Evaluation saves predictions, selected settings, coefficients, fitted preprocessing,
daily/onset fold metrics, calibration and event records. All six model/variant
combinations use the same 2,000 participant bootstrap draws. Summaries report
combined-minus-wearable AP and within-day-minus-baseline AP/ROC-AUC on paired days.
Onset thresholds are chosen again using each model's validation predictions.

The report is exploratory because prior results from these participants have
already informed development. Confidence intervals condition on the fitted models,
are not corrected for multiple comparisons, and do not remove that limitation.
Onset metrics do not have uncertainty intervals. Reference labels do not establish
independently confirmed biological onset. Availability requirements can select
different days; compare new results against the refitted same-cohort baseline.
