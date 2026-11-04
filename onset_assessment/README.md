# Measurement assessment around luteal onset

This separate folder inspects quality-approved measurements around the 81
reference onsets represented by the existing three-day prediction task. It reads
the full 2022 onset audit rather than restricting each signal to days when all
other model inputs are available. It does not fit or change any prediction model.

From the thesis workspace:

```powershell
code/.venv/Scripts/python.exe -m unittest discover -s code/onset_assessment/tests -v
code/.venv/Scripts/python.exe code/onset_assessment/main.py
```

`analysis.py` creates calendar-aligned event windows from day -7 to +7. The axis
uses the existing prediction-day alignment: CGM at t comes from calendar day
t-1, while wearable values come from the night ending on t by 14:00. Day 0 is
an exact supplied non-luteal-to-luteal label transition on consecutive days.
Glucose median, glucose coefficient of variation, temperature, sleeping heart
rate and HRV are included. HRV is displayed as RMSSD in milliseconds by reversing
the stored natural logarithm; the model's logged feature remains unchanged.

Each event is centred on its own mean for days -7 to -4, requiring at least
three observed days. This is a descriptive reference, not a change to model
features. Compare -3 to -1 with 0 to +2 using events with at least two observed
days in each window and a usable baseline. This paired requirement keeps the
event mix identical in before/after tables for each measurement. Daily curves
use all available centred data and can have different coverage across days.
Average events within each participant before averaging participants equally.
Use 2,000 participant-bootstrap draws with seed 20261021 and pointwise percentile
intervals. Gaps and missing signals remain missing. No new dependencies are added.

Generated event grids are saved under
`code/data/processed/onset_assessment_2022_v1`. Tables, a report, figures in PNG
and PDF, input/source hashes, design settings, versions and a complete manifest
are saved under `results/onset_assessment_2022_v1`. Both destinations must be
fresh; override them with `--processed-output-dir` and `--results-dir` to rerun.

The figures show participant variation and measurement coverage. These onset-
conditioned comparisons can identify candidate advance signals or changes that
appear after onset, but they do not measure performance against all negative
days or independently verify the biological date of ovulation. Missing-data
selection, small event counts and possible reference-date uncertainty remain
relevant when interpreting the patterns.

## Actual glucose predictors on positive and negative days

Run the additional comparison with:

```powershell
code/.venv/Scripts/python.exe code/onset_assessment/glucose_comparison.py
```

This reads the exact six glucose predictors from the shared model cohort and
checks them against the saved logistic model input freeze. Positive days have
an exact reference onset within the next three calendar days; negative days
have no start on all three known future days. Require both classes within a
participant for paired comparisons, saving coverage for all 41 participants.
No minimum count beyond one day in each class is imposed; report actual counts.

For each feature, compare positive and negative means and medians within each
participant. Average participant differences equally and use 2,000 participant
bootstrap draws with seed 20261021 for percentile intervals. Also calculate
each person's single-feature AUC with a fixed higher-value direction and half
credit for ties, then average these AUCs across participants. This describes
directional overlap, rather than fitted or cross-validated model performance.

The distribution figure subtracts each person's negative-day mean for display
only, with equal total weight per person within each class. These label-based
display values are never used as prediction inputs. The second figure shows
every participant's mean difference and the group interval in native units.
Neither feature windows nor model settings change, and no models are refitted.

Intermediate display data and paired observation keys/features go under
`code/data/processed/glucose_feature_assessment_2022_v1`. Coverage, per-person
results, summary intervals, PNG/PDF figures, a report and provenance go under
`results/glucose_feature_assessment_2022_v1`. Both output folders must be fresh;
use `--processed-output-dir` and `--results-dir` for later runs. Input locations
can be set with `--processed-dir`, `--preparation-dir` and `--frozen-dir`.
# Glucose phase and successive-cycle assessment

Run from the thesis workspace:

```powershell
code/.venv/Scripts/python.exe code/onset_assessment/cycle_main.py
```

Choose fresh `--results-dir` and `--processed-output-dir` destinations for another run.
The defaults are `results/glucose_cycle_assessment_2022_v1` and
`code/data/processed/glucose_cycle_assessment_2022_v1`.

This independent descriptive analysis uses all quality-approved 2022 CGM from the
verified onset preparation, regardless of wearable availability. Match each glucose
summary and its unchanged 14-day personal deviation to the actual source day.
Identify cycles between consecutive exact menstrual-label starts; audit missing
labels/dates, missing phases and phase reversals. Leading/trailing partial cycles
remain in participant coverage counts. Do not assume an initial menstrual block
begins on its first observed day.

Compare phase means and medians with the same cycle's follicular values, requiring
at least three measured days and 50% labelled-phase coverage. Average cycle effects
within participants before group estimates and 2,000 participant bootstraps
(seed 20261021). A stricter five-day sensitivity retains the 50% requirement.
Raw median glucose and variability are the main measurements; the existing personal
glucose deviation is secondary. Report all phase-specific exclusions and sample sizes.

Successive-cycle comparisons require adjacent bounded cycles, never skipping missing
comparisons. Report sign agreement and absolute effect-size change, averaging pairs
within each person first. These are descriptive repeatability measures, not model
accuracy. Shared population direction alone can cause agreement; 50% is not a formal
chance null. No predictive model, new feature window or hormone adjustment is added.

Focused checks:

```powershell
$env:PYTHONPATH='code'
code/.venv/Scripts/python.exe -m unittest discover -s code/onset_assessment/tests -p test_cycles.py
```
