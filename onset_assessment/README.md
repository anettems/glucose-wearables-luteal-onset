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
