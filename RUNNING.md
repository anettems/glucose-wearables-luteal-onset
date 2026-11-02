# Run the thesis analysis

From PowerShell in the thesis workspace, run:

```powershell
.\code\run_analysis.cmd
```

From inside `code`, use `.\run_analysis.cmd`. The launcher also works from other
folders when given its full path. It uses the existing `code/.venv` environment,
so no environment activation or package installation is needed.

Alternatively, with Python available on the command line:

```powershell
python code/run_analysis.py
```

The Python entry point automatically selects the project environment too.

The command runs the personal deviations and trends analysis currently described
in the thesis. It prepares histories, freezes the protocol and folds, evaluates
the existing nine model/input combinations, calculates the 2,000 participant
bootstrap resamples per feature variant, and generates `REPORT.md` and figures.
This remains exploratory because the primary test results were already inspected.
The within-day glucose extension is a separate analysis and is not run here.

Each invocation chooses fresh timestamped directories under
`code/data/processed/` and `results/`. The final message gives the report path.
Existing runs and raw data are preserved, including after a failed run.

The launcher requires the existing prepared inputs in
`code/data/processed/primary_1400_2022`, their preparation report in
`results/primary_1400_2022`, and the original protocol in
`results/primary_protocol_2022_v4`. It saves a fresh primary freeze with current
source hashes, then verifies the original settings, software versions, prepared
inputs and participant assignments before preparing the exploratory freeze.
Any difference in those checks stops the analysis before model fitting.
Current source files are recorded rather than silently treated as the old freeze.
The original freeze is preserved. `entry_point.json` records the launcher hash
and links the new run to both protocol directories.

Run the focused launcher checks with:

```powershell
.\code\.venv\Scripts\python.exe -m unittest discover -s code/tests -p test_run_analysis.py -v
```
