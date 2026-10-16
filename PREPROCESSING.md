# mcPHASES 2022 analysis preparation

Run from the thesis workspace with Python containing pandas and NumPy:

```powershell
python code/main.py
python -m unittest discover -s code/tests -v
```

On this workstation the verified interpreter is
`C:/Users/anett/miniconda3/python.exe`. If the Windows `python` alias fails:

```powershell
& C:/Users/anett/miniconda3/python.exe code/main.py
```

The main entry point prepares features without fitting models or choosing participant
splits. The audit stage checks labels/hormones, glucose, sleeping HR from sleep scores, daily RHR for description, HRV,
computed temperature and sleep. Other exports are inventoried only.

## Descriptive audit outputs

Running `preprocessing.py` alone produces audit outputs. Use `main.py` for the
active 14:00 feature datasets described below.

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
fixed prediction time remain a separate step in the primary protocol below.

## Active 14:00 analysis datasets

Run `python code/main.py` from the thesis workspace. The defaults are:

- `code/data/processed/primary_1400_2022/`: audited sources, alignment,
  eligible observations and the shared feature datasets.
- `results/primary_1400_2022/`: quality reports, `data_reduction.csv`,
  `dataset_summary.json`, `temporal_protocol.md` and `run_config.json`.

For another run use fresh paths; existing non-empty destinations are refused:

```powershell
python code/main.py --processed-dir code/data/processed/primary_1400_repeat --report-dir results/primary_1400_repeat
```

For prediction day t, CGM uses the whole of calendar day t-1, with at least 70%
nominal coverage and no uncovered gap over four hours, including day boundaries.
Use one selected main sleep ending on t at or before **14:00**, including a sleep
ending exactly at that time. Actual publication/synchronisation times are unknown.
The computer's timezone does not redefine the assumed participant-local clocks.

At least two of these three wearable summaries must be usable:

- Sleeping HR: positive finite sleep-score HR with an exact, unambiguous match
  to the selected main-sleep end. Naps and unmatched scores remain unavailable.
- HRV: median RMSSD from at least six valid five-minute periods wholly inside
  that sleep, each with coverage >=0.8, followed by natural log. Intervals
  beginning on the previous day are matched by the sleep end day.
- Skin temperature: valid SKIN value and positive integer sample count, exactly
  matched to that sleep. Conflicting/ambiguous sessions remain unavailable.

Daily RHR, sleep duration, sleep score and hormone values are not predictors.
No three-day history, multi-day trends or participant baseline is required.
The existing `add_calendar_history` helper is only for a possible later extension;
no current output uses it. Historical noon/three-day outputs are preserved and
superseded; do not mix them with the current datasets.

### Feature files and missing values

- `prediction_alignment_14_2022.csv`: full calendar grid and quality decisions.
- `eligible_observations_14_2022.csv`: days meeting the agreed rules.
- `observations_2022.csv`: five features, outcomes, timing and missingness diagnostics.
- `glucose_features_2022.csv`: keys plus median glucose (mmol/L) and glucose CV (%).
- `wearable_features_2022.csv`: keys plus sleeping HR (bpm), log median RMSSD (ms)
  and skin temperature (degrees Celsius).
- `combined_features_2022.csv`: keys plus all five features.
- `outcomes_2022.csv`: keys and phase/outcome fields, kept separate from predictors.

All feature files have identical participant/day keys and row order. CGM CV is
100 * sample standard deviation / mean, using equally weighted usable readings.
Use only `model_inputs.FEATURE_SETS` columns as predictors. Keys, labels, onset
flags and missing-value flags are metadata, never predictors.

The verified primary data have **42 participants, 2,450 days, 808 luteal days
and 84 luteal starts**. Of these days, 2,044 have all three wearable summaries
and 406 have two (30 missing HR, 331 missing HRV, 45 missing temperature).
The data-reduction rows are 42/3,697/1,235/123, 42/2,746/906/92,
42/2,529/835/87 and 42/2,450/808/84 (participants/days/luteal days/starts).
These are data-availability counts, not model results.

Missing wearable values deliberately remain blank in exported files. After
choosing participant-disjoint folds, call:

```python
from model_inputs import FEATURE_SETS, replace_missing_wearables

train_filled, validation_filled, medians = replace_missing_wearables(train, validation)
x_train = train_filled[FEATURE_SETS["combined"]]
x_validation = validation_filled[FEATURE_SETS["combined"]]
```

Pass subsets of `observations_2022.csv` with its diagnostics intact. Repeat this
step inside every training/validation split; after selecting model settings,
refit on all development participants before applying to the held-out test set.
The helper rejects participant overlap and an entirely missing training feature.
Use the same fitted medians for wearable-only and combined models. Original
missingness flags stay unchanged for auditing after replacement. Fit any scaling
on training data only, after replacement. Do not use full-dataset or test medians.

No participant folds, model training or alert rules are created by this command.
`features_ready=true` records the prepared features; `model_ready=false` means
split-specific replacement/scaling and model development remain to be done.
Run configuration records audited-source hashes, code hashes and package versions.
Raw data remain read-only; generated datasets are kept locally and not committed.
