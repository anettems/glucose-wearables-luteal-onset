# Three-day forecasting thesis figures

Generate the four scientific figures from the thesis workspace:

```powershell
code/.venv/Scripts/python.exe code/onset_prediction/thesis_figures.py --latex-figures-dir latex/figures
```

This creates vector PDFs, 300 dpi PNG previews and `figure_manifest.json` under
`results/onset_thesis_figures_2022_v2`, and copies the four new PDFs into
`latex/figures`. Existing files are never overwritten. For a later run, supply a
fresh `--results-dir` and omit the LaTeX export when the assets already exist.

The default inputs are the completed `evaluation` directories in
`results/onset_logistic_2022_v1` and `results/onset_forest_2022_v1`.
Override them with `--logistic-dir` and `--forest-dir` when appropriate.
Their sibling `frozen` directories must be available.

The figures show:

1. The 14:00 prediction cutoff, prior-day glucose, overnight wearable measurements,
   and illustrative three-day outcomes.
2. Five participant-disjoint test folds, validation for model selection,
   training-only preprocessing and refitting before separate testing.
3. Pooled average precision (AP) for all six configurations, together with the
   paired combined-minus-wearable AP difference within each model family.
4. Observed frequencies against mean predicted probabilities in the ten fixed
   bins, with exact bin counts for all three input sets in each model family.

The script verifies the hashes of the saved inputs and historical freeze links.
It checks identical cohorts, outcomes, participant folds and participant bootstrap
draws across both model families. Pooled metrics and calibration bins are checked
against the saved test probabilities, and confidence intervals against the saved
bootstrap metrics. It does not refit models, resample participants or change data.
The uncertainty intervals remain conditional on the fitted cross-validation models.

Serif fonts, embedded vector text, light grids and colour plus marker shape are
used for readable scientific figures. No new plotting dependency is required;
the existing reporting requirements include Matplotlib.

Run focused tests from the code directory:

```powershell
.venv/Scripts/python.exe -m unittest onset_prediction.tests.test_figures -v
```

`figure_data.py` validates saved results, `figure_diagrams.py` draws the methods
schematics, and `thesis_figures.py` draws and exports the figures. The manifest
records input and source hashes, output hashes, style settings and package versions.
