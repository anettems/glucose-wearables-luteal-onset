# Three-day onset probability analysis

This new folder implements the feasibility stage of protocol
`2026-11-03-onset-3d-v1`, described in the workspace's `EVALUATION_PROTOCOL.md`.
It does not fit models or generate predictions.

From the thesis workspace, using the existing environment:

```powershell
code/.venv/Scripts/python.exe -m unittest discover -s code/onset_prediction/tests -v
code/.venv/Scripts/python.exe code/onset_prediction/main.py
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

Files: `protocol.py` records settings; `outcomes.py` constructs labels;
`features.py` reuses and checks features; `folds.py` allocates and audits folds;
`workflow.py` saves the report and provenance; `main.py` provides the entry point.
The probability modelling and evaluation stage is not implemented yet.
