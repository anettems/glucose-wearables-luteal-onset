# Thesis methods and code consistency review

The final primary protocol is `2026-10-20-primary-v4`. This review concerns the
methods snapshot saved alongside the protocol and the analysis sources identified
in `provenance.json`. The freeze stage validates data and folds without fitting
models or generating test predictions. It does not make claims about whether any
earlier draft evaluations were run.

| Component | Agreed thesis method and code implementation |
| --- | --- |
| Scope | 2022 only; current-day luteal classification followed by onset alerts; no secondary analyses |
| Inputs | Two previous-calendar-day glucose summaries; three main-sleep wearable summaries available by 14:00; shared eligible keys for all models |
| Outcomes | Supplied phase labels; exact starts from consecutive calendar days with non-luteal then luteal labels; full-alignment references |
| Replacement | Training-participant feature medians for missing wearables; identical replacements for wearable-only and combined inputs |
| Scaling | Training-only means and population standard deviations; constant features use scale one |
| Model | Scikit-learn L2 logistic regression, unpenalised intercept, no class weights; direct C grid 0.01, 0.1, 1, 10; lbfgs, 1,000 iterations, tolerance 1e-10; convergence failure stops evaluation |
| Testing folds | Five participant-disjoint folds; seed 20261020; capacity-balanced allocation and at most 20 improving swap passes balancing days, luteal days and full-alignment exact starts |
| Validation folds | Three participant-disjoint folds within each development cohort; seed 20261020 plus test-fold number; identical allocation across input sets |
| Feasibility | Both outcome classes and at least one evaluable reference start in every assessment fold; each training complement has observed values for every wearable feature |
| Model tuning | Highest mean validation non-interpolated AP; exact ties choose smaller C; refit on all development participants |
| Daily estimates | Pooled out-of-fold AP, complementary ROC-AUC, and luteal prevalence; fold metrics are stability checks |
| Primary effect | Combined minus wearable-only pooled AP on identical held-out keys |
| Bootstrap | 2,000 participant draws with replacement, seed 20261021; all held-out days, including repeated selections; same draw for all models; 2.5th/97.5th percentiles with linear interpolation; intervals conditional on fitted models |
| Undefined bootstrap metrics | AP undefined with no positives; ROC-AUC undefined with one class; retain draws, use finite values for intervals, and report contributing counts |
| Alert selection | Use validation predictions for selected C; thresholds 0.1 to 0.9 in 0.1 steps; one or two consecutive calendar days; highest mean event F1, then higher threshold, then two days |
| Positive prediction | Probability at or above the threshold |
| Alert episodes | Start inactive; alert on confirmation day; never backdate; missing days break pending confirmation but preserve an active episode; reset only on a subsequent available probability below threshold |
| Reference evaluability | At least one eligible prediction day within the inclusive three-day window on either side of the exact reference start |
| Event matching | Within each participant and interval, greedily take the closest available pair, then earlier alert, then earlier reference; each alert/start used at most once |
| Event reporting | Event F1 = 2M/(2M + missed evaluable starts + unmatched alerts), or zero if denominator zero; report counts with mean absolute timing error; no matched events means timing error is undefined |
| Reproducibility | Final protocol, input/source hashes, methods snapshot, software versions and saved assignments precede test prediction; later evaluation verifies and reuses the freeze |

The thesis programming-environment paragraph distinguishes the earlier data
preparation versions from the frozen modelling versions. The frozen files record
all numerical dependencies used by the model and bootstrap implementation.

These choices assess agreement with supplied labels. They do not independently
validate ovulation or establish clinical or causal performance. Actual upload
times are unavailable; availability by 14:00 remains an analysis assumption.
