# 21-day glucose-history experiment

This separate comparison tests one extra predictor for three-day luteal onset.
The baseline is the existing combined glucose and wearable logistic model.
The extended model adds recent minus oldest weekly median glucose, in mmol/L.
The aligned weeks are t−7 through t−1 and t−21 through t−15. Each aligned glucose
value comes from the completed calendar day before its aligned day. The feature
uses at least four observed days in each end week and 15 in the full 21-day
window. Missing days keep their calendar positions. The calculation precedes
cohort and outcome filtering.

Both variants train, validate and test on exactly the same subset of the
existing shared cohort, using its saved participant folds. Validation chooses
the logistic penalty separately for each variant by mean average precision.
Fitted preprocessing uses only training participants. The main estimate is the
extended model's pooled held-out average precision minus the baseline's. A
paired participant bootstrap quantifies uncertainty conditional on the fits.

Run from the thesis workspace:

```powershell
$env:PYTHONPATH='code'
code/.venv/Scripts/python.exe -m unittest discover -s code/history_experiment/tests -v
code/.venv/Scripts/python.exe code/history_experiment/main.py
```

The separate `glucose_history_experiment_2022_v1` folders under
`code/data/processed` and `results` contain the frozen inputs and outputs,
respectively. The results folder includes the report, fold checks, selected
settings, training parameters, predictions, metrics and reproducibility records.
Use `--results-dir` and `--processed-output-dir` with fresh paths for reruns.
