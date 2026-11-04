# Longer-history coverage audit

Run from the thesis workspace:

```powershell
$env:PYTHONPATH='code'
code/.venv/Scripts/python.exe -m unittest discover -s code/onset_assessment/tests -p test_history_coverage.py -v
code/.venv/Scripts/python.exe code/onset_assessment/history_main.py
```

This descriptive audit reads the verified 2022 onset preparation. It compares
14-, 21- and 28-calendar-day past-only histories, excluding the current aligned
measurement. Glucose source days are t-L-1 through t-2; the current glucose value
from source day t-1 remains separate. It does not fit a longer-history model or
alter the existing features, cohort, outcomes or participant folds.

For each history length, count quality-approved glucose days, days with at least
two wearable signals, and their intersection. Wearable channels can differ across
days, so these joint counts do not establish complete per-channel sequences.
Count absent calendar slots as missing, without filling values or extending the
window. Record longest gaps including edges, slots before the available timeline,
missing slots within it, oldest-week coverage and support beyond the last 14 days.

Report coverage sensitivities of 50%, 70%, 80%, 90% and 100%, rounding the minimum
day count upwards. These are history-day sensitivities, distinct from the existing
within-day CGM quality rule. Report the full timeline, current-glucose days and the
unchanged shared modelling cohort separately. Preserve unknown outcomes outside
the supervised cohort and distinguish positive days from distinct reference starts.
All histories use measurements before filtering for labels or other sensors.

Outputs include per-day windows under `data/processed/history_coverage_2022_v1`
and a report, coverage/gap/participant/fold tables, protocol and manifest under
the workspace's `results/history_coverage_2022_v1`. Use fresh `--results-dir` and
`--processed-output-dir` paths for reruns. Input and source hashes and package
versions are saved. Existing outcomes and shared-cohort keys are rechecked, and
14-day glucose counts must match the original feature preparation exactly.

Saved test and validation assignments are retained. Fold tables check both outcome
classes and represented starts in each assessment and training complement after
each coverage restriction. They do not establish preprocessing feasibility for
new features. No extra dependencies, random sampling or model fitting are used.
