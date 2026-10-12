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
`main.py`. It audits labels/hormones, glucose, sleeping HR from sleep scores, daily RHR for description, HRV,
computed temperature and sleep. Other exports are inventoried only.

## Outputs

- `code/data/processed/audit_2022/`: seven audited 2022 tables and
  `daily_audit_2022.csv`, with one row per participant and calendar study day.
- `results/preprocessing_2022/`: audit report, table summaries, per-table
  missingness, flagged source rows, numeric checks, participant coverage,
  value distributions, phase counts, unknown phase records and a run manifest.

Source values are preserved after exact duplicate removal. Glucose is
standardised to mmol/L as described below. Additional `audit_`
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
Glucose distributions include flagged records for transparent inspection.
`glucose_value` is standardised to mmol/L. On 12 October 2026 the user confirmed
that participants 6 and 11 have mg/dL source values. These participants are
converted using `mmol/L = mg/dL * 0.0555`, following the
[NIDDK conversion table](https://www.niddk.nih.gov/-/media/Files/Strategic-Plans/Diabetes-in-America-3rd-Edition/DIA_Conversions.pdf).
Other participants provisionally retain the README's mmol/L units.

The audited glucose table retains `glucose_source_value`, `glucose_source_unit`,
`glucose_unit`, `glucose_conversion_factor`, and `glucose_unit_converted`.
The manifest records the mapping, factor, confirmation date and provenance.
The confirmation came from the user; it was not independently checked with the
provider. Conversion is based on this explicit confirmation, never inferred
from value magnitude. Source numeric checks and duplicate detection precede
conversion, so malformed, non-positive and conflicting records remain flagged.
Missing values remain missing, and attempting to convert an already
standardised table raises an error. Raw exports are never changed.

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

The optional reviews below implement quality and temporal rules. Remaining
measurement-availability assumptions must be considered before modelling.
This audit does not establish independent hormone validation
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

- Standardise the user-confirmed mg/dL source values for IDs 6 and 11 to mmol/L.
  Remove their former unit-based exclusion from every primary input comparison.
  They must still pass the same signal, timing and history checks as everyone
  else. Other IDs provisionally follow the export documentation. The unit-evidence
  table reports both original and standardised distributions.
- Daily RHR is audited descriptively. Primary eligibility requires sleeping HR
  matched to the selected main sleep, as specified below.
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

## Noon alignment with sleeping HR

The active protocol uses **12:00 only** and sleep-score HR in place of daily RHR.
Historical outputs are preserved; they are not the current cohort. Run with fresh
output directories:

```powershell
& C:/Users/anett/miniconda3/python.exe code/preprocessing.py --temporal-review --processed-dir code/data/processed/sleeping_hr_primary_2022 --report-dir results/sleeping_hr_primary_2022
```

For a prediction on day t:

- CGM uses completed day t-1 and the unchanged 70% nominal coverage / four-hour
  maximum uncovered-gap rules.
- Sleeping HR comes from `sleep_score.csv.resting_heart_rate` (bpm). Require finite
  positive HR, no conflicting score key, and an exact match of participant,
  study interval, end day and end time to one selected valid main sleep. Use the
  supplied value, preserving its original field; do not call it minimum HR.
- The same main sleep must end on t at or before noon. Naps, ambiguous matches
  and unmatched scores are excluded; do not average daily scores or guess matches.
- HRV is from `heart_rate_variability_details.csv`: at least six usable five-minute
  RMSSD intervals with coverage >=0.8 must fit wholly inside that main sleep.
  Timestamps denote interval starts, including preceding-day intervals.
- Skin temperature must match the same selected session under the existing rules.
- Require all aligned inputs on three consecutive calendar days t-2, t-1 and t.
  Glucose source days are t-3 through t-1; sleeping HR/HRV/temperature session end
  days are t-2 through t. Historical labels are not required or used as predictors.

Daily RHR remains in the descriptive audit and a source-change comparison only;
missing or zero daily RHR does not affect primary eligibility. Sleep is alignment
and quality metadata; duration, efficiency, stages and sleep scores are not model
predictors. No imputation, future data or alternative prediction cutoff is used.

The 2022 sleep-score assessment found 3,372 positive HR records across 42
participants, with no duplicate session keys and 123 days containing multiple
scores. Exactly 3,238 match selected main sleeps, 128 match non-main sessions,
and six lack an exact end match. Same-day end discrepancies for those six are
hours rather than small rounding errors; their cause remains unverified. These
six are excluded without correcting timestamps. Of the 3,238 selected matches,
2,848 end by noon, before CGM/HRV/temperature/history requirements.

Outputs include:

- `sleep_score_2022_audited.csv`: preserved source values and structural flags.
- `sleeping_hr_session_review_2022.csv`: matching decisions, end-time discrepancies
  and selected HR; `sleeping_hr_excluded_scores.csv` records excluded scores.
- `morning_alignment_12_2022.csv`: full calendar grid, source days, sleeping HR
  value/source timestamp, aligned signal flags and three-day history counts.
- `eligible_observations_12_2022.csv`: one shared eligible table for all models.
- `temporal_attrition.csv`, `temporal_protocol.md` and `sleeping_hr_decision.md`:
  rules, cumulative cohort counts and remaining assumptions.
- `hr_source_cohort_comparison.csv` and `hr_source_eligibility_changes.csv`:
  legacy daily-RHR versus sleeping-HR cohorts, including gained/lost keys.

Current eligibility is 36 participants, 1,289 labelled prediction days and 38
eligible exact reference-transition dates, compared with 35/1,120/32 for daily
RHR under otherwise identical noon rules. These are cohort counts, not model
performance; onset-event evaluation rules and participant splits remain pending.

The score timestamp is assumed to identify the matching sleep end, not the
publication time. Exact HR aggregation and publication/synchronisation times
remain undocumented. Export clocks are assumed mutually aligned participant-local
times; no timezone conversion is applied. The client computer's timezone does
not redefine the participant measurement clocks. All audited raw-file hashes,
parameters and software versions are recorded. No model-ready matrix is produced.
