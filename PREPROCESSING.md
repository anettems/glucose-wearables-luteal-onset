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

## Eligibility decision review

Run the extension with fresh output directories:

```powershell
& C:/Users/anett/miniconda3/python.exe code/preprocessing.py --eligibility-review --processed-dir code/data/processed/eligibility_2022 --report-dir results/eligibility_2022
```

The additional `decision_log.md` explains six decisions, cites the reviewed
primary documentation, and gives before/after participant, labelled-day and
exact-transition counts. `decision_sources.json` records the source review
date, URLs, local README hash and selected settings. The program uses this
documented evidence locally; it does not fetch web pages on each run.

Default quality-review rules are:

- Exclude IDs 6 and 11 from every primary input comparison until their glucose
  units are independently confirmed. Raw values remain unchanged. A difference
  in value scale or integer precision does not establish a unit conversion.
  Other IDs provisionally follow the export documentation. Reinstatement
  requires documented correction and regeneration of all paired cohorts.
- Resting heart rate requires a positive finite value and an unambiguous daily
  key. Equal values on different study days are retained. No averaging rule is
  introduced for 2022 records.
- HRV requires coverage at least 0.8 and at least six usable five-minute
  intervals per exported day. Values above one are retained without clipping
  or weighting because Fitbit does not document an upper bound. Their cause
  remains unverified. The alternative upper bound of one is reported separately.
- CGM requires a nominal observed-time fraction of at least 0.70 and a longest
  uncovered period of at most four hours, including gaps at midnight. Sample
  counts, occupied bins, cadence and interior/boundary gaps are also reported.
- Sleep selection requires a single valid main sleep per end day, without
  overlap with another main sleep. Naps and ambiguous multiple-main days are
  excluded. Elapsed time and exported sleep duration may differ by at most
  60 seconds to allow minute granularity. Temperature requires SKIN type,
  positive integer sample counts, and an exact start/end match with the selected
  main sleep; multiple qualifying temperatures on one end day are excluded.
- Exact phase transitions require adjacent recognised non-luteal/luteal labels.
  Uncertain entries are classified as interval-censored, left-censored or
  possible continuations of an already observed luteal segment. They are not
  counted as extra exact events.

The coverage, window-count and duration-tolerance thresholds are analyst
choices for a reproducible quality review, not validated clinical limits.
Coverage sensitivity tables compare CGM coverage of 0.70/0.80/0.90 and maximum
gaps of two/four hours, and HRV coverage of 0.70/0.80/0.90 with three/six/twelve
intervals and optional exclusion of values above one. The chosen settings
are also included when they differ from these values. CLI settings are
`--min-glucose-coverage`, `--max-glucose-gap-hours`, `--min-hrv-coverage` and
`--min-hrv-windows`.

Additional processed tables are `quality_review_daily_2022.csv`,
`session_review_2022.csv` and `phase_transition_review_2022.csv`. Reports are
`rule_effects.csv`, `same_day_quality_attrition.csv`, `glucose_unit_evidence.csv`,
`coverage_sensitivity.csv` and `session_end_time_sensitivity.csv`.

All effects currently use the original exported calendar day. Transition counts
require that reference day alone, without surrounding sensor histories.
Independent rule effects are not additive. `same_day_quality_overlap` is one
shared preview mask for the three input sets; it must not be used directly for
prospective modelling. Hypothetical session end-clock comparisons at 06:00,
08:00, 10:00 and 12:00 quantify timing sensitivity without selecting a cutoff.
They do not prove actual device availability. Fitbit documents HRV processing
after sleep and synchronisation; the export lacks verified sync/processing
times and timezone conventions. Prior-day CGM, overnight HRV alignment, the
fixed morning cutoff and trailing calendar histories remain a separate step.

## Morning alignment and three-day eligibility

The accepted temporal protocol is implemented as an optional extension. It
also runs the initial audit and quality review, preserving those separate
descriptive outputs. Use fresh directories:

```powershell
& C:/Users/anett/miniconda3/python.exe code/preprocessing.py --temporal-review --processed-dir code/data/processed/temporal_2022 --report-dir results/temporal_2022
```

The primary decision is at **10:00**, with an **08:00** sensitivity analysis.
The exported clocks are assumed to represent mutually aligned participant-local
times. Timezones, daylight-saving changes and device synchronisation/publication
times are unverified. No timezone conversion or arbitrary processing buffer is
applied. Completed prior-day summaries and completed main-sleep measurements
are assumed available by cutoff; the outputs do not demonstrate actual
real-time availability. These assumptions are recorded in `run_config.json`
and `temporal_protocol.md`.

For each prediction day `t`:

- CGM uses the completed calendar day `t-1`, with the existing 70% nominal
  observed-time and maximum four-hour uncovered-gap requirements.
- Daily resting heart rate uses `t-1`, because current-day summary availability
  is not established.
- Sleep and matching temperature use the selected main sleep ending on `t`
  at or before the cutoff. A session ending later is unavailable; no fallback
  to an earlier night is substituted for that day's measurement.
- HRV intervals must fall wholly inside that selected sleep, including
  intervals beginning on the preceding calendar day. Start and end boundaries
  are inclusive, and each interval ends five minutes after its timestamp.
  At least six usable intervals with coverage at least 0.8 are required per
  selected sleep. Source day keys are preserved in the alignment diagnostics.
- IDs 6 and 11 remain excluded consistently from every primary input set.

Exactly **three usable consecutive calendar days** of aligned inputs are
required: `t-2`, `t-1`, and `t`. Consequently, the glucose/RHR source dates
span `t-3` through `t-1`, and wearable sleep end dates span `t-2` through `t`.
Calendar-key joins prevent missing dates from being compressed into consecutive
rows and prevent participant or study-interval boundaries from being crossed.
All required signals must pass on all three days, even for the single-modality
comparisons, so the three primary models share identical eligible keys.
Only the target day's recognised phase label is required; earlier phase labels
are not predictors or history requirements. There is no imputation and no
seven-day alternative or long-term personal baseline in this protocol.

Additional processed outputs:

- `hrv_sleep_alignment_2022.csv`: original interval keys, matched main-sleep
  end day, and interval quality/containment flags.
- `morning_alignment_10_2022.csv` and `morning_alignment_08_2022.csv`: full
  calendar grids, lagged CGM diagnostics, source-day provenance, sleep timing,
  aligned quality flags, usable history counts and `eligible_prediction`.
- `eligible_observations_10_2022.csv` and `eligible_observations_08_2022.csv`:
  the single shared eligible observation table for each cutoff.

`temporal_attrition.csv` and `temporal_protocol.md` report cumulative effects
and assumptions. Exact transition counts require an eligible reference date;
they do not yet establish evaluable onset events or account for event-matching
requirements. Each cutoff has its own shared cohort. Comparisons between
cutoffs must account for the changed cohort or restrict to common keys.
The script does not construct model features or participant splits, and
`model_ready` remains false. Feature construction, participant-disjoint splits
and onset-scoring rules remain subsequent work.
