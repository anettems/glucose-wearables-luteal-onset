# Three-day onset probability analysis

This folder implements preparation, logistic regression and random forest for protocol
`2026-11-03-onset-3d-v1`, described in the workspace's `EVALUATION_PROTOCOL.md`.
The preparation and freeze stages do not fit models or generate predictions.
Each model family has a separate freeze and evaluation stage, using the same cohort and folds.

From the thesis workspace, using the existing environment:

```powershell
code/.venv/Scripts/python.exe -m unittest discover -s code/onset_prediction/tests -v
code/.venv/Scripts/python.exe code/onset_prediction/main.py

# Reuse the saved feasible cohort and folds; freeze before any model fitting.
code/.venv/Scripts/python.exe code/onset_prediction/main.py --freeze-logistic

# Verify the freeze, tune C using validation participants, and evaluate held-out probabilities.
code/.venv/Scripts/python.exe code/onset_prediction/main.py --evaluate-logistic

# Copy the verified logistic cohort, folds and participant draws; freeze forest settings.
code/.venv/Scripts/python.exe code/onset_prediction/main.py --freeze-forest

# Tune minimum leaf size and evaluate random-forest probabilities.
code/.venv/Scripts/python.exe code/onset_prediction/main.py --evaluate-forest
```

The entry point reads audited data from `code/data/processed/primary_1400_2022`
and its configuration from `results/primary_1400_2022`. It reuses the current
sensor-summary and personal-history calculations, checking all current values
against the earlier primary observations and all 15 predictors against the earlier
history-ready observations in `code/data/processed/trend_analysis_2022_v1`.

The target is 1 when an exact supplied reference start occurs on t+1, t+2 or t+3.
It is 0 only when every candidate day has a known no-start status. Other windows
remain unknown. Missing dates never extend the horizon. Outcomes use the full
label timeline; sensors need not be available on the onset day. Current-day phase
does not restrict prediction eligibility, and future measurements are not inputs.

The unchanged sensor/history requirements and a determinable outcome define one
paired cohort for all six model/input configurations. Missing wearable predictors
remain blank: no replacement, scaling or other model fitting happens here.

New datasets are written under `code/data/processed/onset_prediction_2022_v1`:

- `outcome_audit.csv`: the full alignment, future start statuses and eligibility.
- `observations.csv`: the usable shared cohort with history and timing diagnostics.
- `outcomes.csv` and the three `*_features.csv` files: identical keys and order.
- `reference_links.csv`: all prediction-to-reference links for counting distinct
  starts; these are outcome provenance, not alerts or event matching.

New reports are written under `results/onset_prediction_2022_v1`: cohort coverage,
participant profiles, reference coverage, saved test/validation assignments,
all 40 cohort checks, predictor availability, protocol settings, source/input
hashes, software versions, a manifest and `REPORT.md`.

Fold allocation balances participant count, eligible days, positive days and
distinct starts represented by positive days. Five test folds and three validation
folds within each development set use seed 20261020 and validation seeds
20261021–20261025. New assignments are generated because the outcome and usable
cohort have changed. Every test, development, validation and training cohort must
have both classes, at least one represented start and observed values for every
predictor. Report actual counts; these checks are not a statistical power analysis.

Existing outputs are never overwritten. For a new run, pass fresh destinations
with `--processed-output-dir` and `--results-dir`; the paths must remain under
the designated processed-data and results roots. Raw data are never changed.
No new packages are required.

## Logistic regression

The freeze saves the same 1,932-day, 41-participant cohort and assignments from
the feasibility stage. It verifies saved preparation/input hashes and unchanged
feature/outcome construction sources. Only the preparation entry point and its
README may differ from the earlier feasibility source record when adding this
evaluation mode. The new freeze records all current source hashes, settings,
software versions, fold counts and 2,000 participant bootstrap draws.

The default freeze is `results/onset_logistic_2022_v1/frozen`; evaluation outputs
are written to its sibling `evaluation` directory. Override these paths with
`--frozen-dir` and `--evaluation-dir` for a fresh version. Evaluation rejects any
source, input, frozen settings or numerical-version change after freezing.

For each of five test folds and each input set, compare C = 0.01, 0.1, 1 and 10
using the mean average precision across the three saved validation folds. Exact
ties select smaller C. Refit on all development participants before predicting
test participants. Missing wearable inputs use training medians; glucose inputs
must be observed. StandardScaler uses training means and population standard
deviations, with scale one for constant features. The lbfgs model uses an
unpenalised intercept, no class weights, 1,000 maximum iterations and tolerance
1e-10. In scikit-learn 1.9, `l1_ratio=0` specifies the agreed L2 penalty without
using the deprecated `penalty` argument. Convergence warnings stop evaluation.

Outputs include candidate validation probabilities and scores, selected settings,
validation/final medians and scaling coefficients, 15 fitted model records
(`joblib`), test probabilities, fold metrics, pooled AP/ROC-AUC/Brier estimates,
ten fixed calibration bins, paired participant bootstrap metrics and the
combined-minus-wearable AP difference. The report and all outputs have a manifest.
No probability threshold, alert rule or learned recalibration is used.

Bootstrap draws are frozen separately as `bootstrap_participants.csv`. Each draw
retains all days of each selected participant, including repeated selections,
and is identical across input sets. Use the same draws for random forest later.
Percentile intervals are conditional on the fitted cross-validation models.

## Random forest

The forest uses 100 trees, maximum depth 5, square-root feature selection, Gini
splits, bootstrap samples of the training-set size and no class weighting.
Only minimum leaf size (1, 5 or 10) is selected, using mean validation average
precision across the same three folds. Exact ties select the larger leaf size.
Replacement medians are fitted on training participants only and shared wearable
predictors use identical medians across input sets and model families. Forest
inputs are not scaled. The random seed is 20261020 and n_jobs=1 is fixed.

`--freeze-forest` reads the existing logistic input freeze as a historical data
snapshot: it verifies its files, original inputs, numerical versions and agreed
shared design. It also verifies the original feasibility preparation and checks
the two cohorts against each other. It does not require historical evaluation
code to equal the newly extended code, and does not read logistic performance
results. The new forest freeze records all current sources and settings before
fitting. Each evaluation still requires its own current-source freeze; rerunning
logistic regression with changed evaluation sources requires a fresh freeze and
new output destinations. Existing snapshots and results are never overwritten.

The forest freeze copies the logistic observations, reference links, test and
validation assignments, feasibility tables and all 2,000 bootstrap participant
draws byte for byte. Both loaders and the fitting loop check the paired cohort,
participant separation and preprocessing feasibility.

Default paths are `results/onset_forest_2022_v1/frozen` and its sibling
`evaluation` directory. Override them with `--forest-frozen-dir` and
`--forest-evaluation-dir`; select a different shared logistic snapshot with
`--logistic-frozen-dir`. Output records include all candidate validation
probabilities/scores, selected leaf sizes, replacement medians, complete fitted
forests, test probabilities, calibration bins, bootstrap intervals and the
within-forest combined-minus-wearable AP difference. No thresholds or alerts
are used and no other forest settings are tuned.

Files: `protocol.py` records settings; `outcomes.py` constructs labels;
`features.py` reuses and checks features; `folds.py` allocates and audits folds;
`workflow.py` saves the report and provenance; `main.py` provides the entry point.
`logistic.py` and `forest.py` fit/tune their model families; `metrics.py` assesses probabilities and paired
uncertainty; `freezing.py` protects the saved design and inputs; `evaluation.py`
runs either model comparison; `forest_freezing.py` saves/verifies the forest freeze;
`report.py` writes the generated model report.
