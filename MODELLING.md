# Primary modelling and evaluation

The primary comparison uses the prepared 2022 cohort and the same daily keys and
participant splits for glucose-only, wearable-only and combined logistic regression.
The implementation follows `../AGENTS.md` and the methods in `../latex/thesis.tex`.

## Structure

| File | Responsibility |
| --- | --- |
| `main.py` | Command line and preparation/evaluation dispatch |
| `primary_protocol.py` | Fixed seeds, fold counts, tuning grids and protocol version |
| `participant_folds.py` | Balanced participant allocation and training/validation/test feasibility |
| `daily_models.py` | Training-only wearable replacement, scaling and logistic regression; validation tuning |
| `daily_metrics.py` | Paired observation checks, pooled AP/ROC-AUC and participant bootstrap |
| `onset_alerts.py` | Calendar-day alert episodes, shared event matching and validation rule selection |
| `primary_evaluation.py` | Input agreement checks, execution, provenance and saved outputs |

## Commands

From the thesis workspace, install the versions in `code/requirements.txt` into
the chosen Python environment. A project interpreter is available at
`code/.venv/Scripts/python.exe`; the environment directory is excluded from Git.

```powershell
# Show preparation and evaluation options.
code/.venv/Scripts/python.exe code/main.py --help

# Run focused tests on synthetic data.
code/.venv/Scripts/python.exe -m unittest discover -s code/tests -v

# Freeze the methods, versions and feasible folds without fitting a model.
code/.venv/Scripts/python.exe code/main.py --freeze-primary `
    --frozen-protocol-dir results/primary_protocol_2022_v4

# Run the primary evaluation using the saved assignments.
code/.venv/Scripts/python.exe code/main.py --evaluate-primary `
    --frozen-protocol-dir results/primary_protocol_2022_v4 `
    --evaluation-dir results/primary_evaluation_2022_v4
```

The evaluation directory must be empty. Existing outputs, raw data, and preparation
directories are protected. Calling `main.py` without `--evaluate-primary` prepares
the feature tables unless `--freeze-primary` is selected; preparation requires fresh
output directories. Freezing also requires an empty destination. Evaluation checks
the freeze manifest, file hashes, current settings, software versions and source/input
hashes, and loads the saved folds. It refuses to proceed when they differ.

## Model definition and validation

Protocol v4 uses scikit-learn's standard definition of the penalty parameter:

`sum(binary log loss) + sum(non-intercept coefficients squared) / (2 * C)`.

The candidate values of C are 0.01, 0.1, 1 and 10, passed directly to
`LogisticRegression`. Smaller C imposes stronger regularisation. Training size
is not used to convert C. This changes the objective from protocol v3, which
divided C by the number of training days. Any earlier evaluation outputs use a
different model definition; a v4 evaluation must be saved in a fresh directory.
The intercept is unpenalised, no class weights are used, and convergence warnings
stop the run. A focused test checks the fitted objective's gradient.

Each training split fits wearable replacement medians and a `StandardScaler` /
`LogisticRegression` pipeline. Validation selects C by mean AP, breaking ties
towards the smaller C. A new pipeline is fitted on all development participants
before predicting the reserved test fold. Alert settings use validation
predictions only. Both validation and test alerts use the same event matcher.

Fold feasibility checks both daily outcome classes, at least one *evaluable*
reference start in each assessment fold, and observed wearable values in every
training complement. Full-alignment reference starts can be evaluable even if
their exact day is not an eligible prediction day, provided an eligible day
exists within the three-day matching window.

## Saved evidence

The freeze saves the versioned protocol, participant and validation fold assignments,
feasibility counts, input and analysis-source hashes, software versions, a methods
snapshot and a thesis/code consistency review. A timestamped manifest records file
hashes. The evaluation copies this verified evidence before making test predictions.
Completed runs contain:

- Test and selected-setting validation predictions on the original keys.
- Model and alert tuning scores, selected settings and fitted parameters.
- Pooled daily estimates, fold metrics and the paired combined-minus-wearable AP difference.
- Bootstrap metrics and the participant draw for each of 2,000 resamples.
- Alert and reference event tables, fold event metrics and pooled event counts/timing.
- A run configuration and a status file distinguishing complete and failed runs.

Bootstrap draws retain every day of each selected participant, including repeated
participant selections, and use the identical draw for all input sets. A draw with
no positive outcomes has undefined AP; a draw with only one outcome class has
undefined ROC-AUC. Such draws are retained with missing metrics, and summaries
report how many finite replicates contributed to each percentile interval.
Intervals are conditional on the fitted cross-validation models.

These analyses assess agreement with supplied phase labels and assume completed
measurements were available by 14:00. Actual upload times are unavailable. They do
not independently validate ovulation, establish clinical performance or support
causal conclusions.
