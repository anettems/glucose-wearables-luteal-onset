# mcPHASES 2022 audit

Run from the thesis workspace with Python containing pandas and NumPy:

```powershell
python code/preprocessing.py
python -m unittest discover -s code/tests -v
```

On this workstation the verified interpreter is
`C:/Users/anett/miniconda3/python.exe`. If the Windows `python` alias fails:

```powershell
& C:/Users/anett/miniconda3/python.exe code/preprocessing.py
```

The workflow is independent of `scratch.py` and does not run models or change
`main.py`. It audits labels/hormones, glucose, resting heart rate, HRV,
computed temperature and sleep. Other exports are inventoried only.

## Outputs

- `code/data/processed/audit_2022/`: six audited 2022 tables and
  `daily_audit_2022.csv`, with one row per participant and calendar study day.
- `results/preprocessing_2022/`: audit report, table summaries, per-table
  missingness, flagged source rows, numeric checks, participant coverage,
  value distributions, phase counts, unknown phase records and a run manifest.

Raw values are preserved after exact duplicate removal. Additional `audit_`
columns describe structural problems and timing. Conflicting measurement keys
are retained and flagged, not averaged. A session's daily audit key is its end
day; both original start and end fields are preserved. Timing is represented
in seconds on the exported relative-study-day axis, not as verified UTC times.

`daily_audit_2022.csv` preserves all days within each participant's union of
audited recording spans. Missing record counts are zero; missing labels stay
missing. `luteal_status` and `exact_luteal_onset` use nullable integers:
unknown outcomes are blank, never zero. Exact onset requires adjacent known
calendar-day labels. `uncertain_luteal_entry` marks luteal observations after
an unknown predecessor, including the recording boundary; it is not a second
set of confirmed events.

`*_available_records` means that the primary numeric value is present and the
record has no conflicting key, invalid time, or invalid audited numeric field.
Missing secondary fields do not by themselves exclude a record. HRV coverage
above one is reported without treating it as invalid. These counts are audit
indicators, not a final common modelling cohort or verified biological validity.

Glucose coverage uses the union of nominal forward five-minute intervals,
clipped at midnight. Longest uncovered time includes boundary gaps. This is a
documented sampling assumption, not proof that every covered second was valid.
Glucose distributions include flagged records for transparent inspection; no
unit conversions are made. Participant 6 and 11 unit questions remain open.

## Repeating a run

Existing non-empty output directories are refused to protect prior work.
Choose fresh destinations to reproduce or compare a run:

```powershell
python code/preprocessing.py --processed-dir code/data/processed/audit_2022_repeat --report-dir results/preprocessing_2022_repeat
```

The manifest records source hashes, the script hash, software versions and
paths. Outputs are deterministic for the same inputs and software; the manifest
naturally records different destination paths on a repeat run. Source hashes
are verified again before writing. Audited source exports and generated data
are not intended to be committed to Git as part of the implementation.

Before modelling, fix the morning cutoff, measurement-availability assumptions,
duplicate resolution, units, minimum coverage, history requirements and common
eligibility rules. This audit does not establish independent hormone validation
or prospective prediction performance.
