#### Allocate and audit participant-disjoint folds for the new binary outcome.
#### Balance participant counts, eligible days, positive days and distinct represented starts.
#### Check all test, development, validation and training cohorts without fitting transforms.

import numpy as np
import pandas as pd

from onset_prediction import protocol


def represented_events(data, links):
    #### Select distinct reference starts with at least one prediction in the current cohort.
    selected = links.merge(data[protocol.KEY], on=protocol.KEY, validate="many_to_one")
    return selected[["id", "study_interval", "reference_day"]].drop_duplicates()


def participant_profile(data, links):
    #### Summarise each participant's daily outcomes and distinct represented onset events.
    profile = data.groupby("id", sort=True).agg(
        eligible_days=(protocol.TARGET, "size"), positive_days=(protocol.TARGET, "sum")).reset_index()
    counts = represented_events(data, links).groupby("id").size().rename("represented_starts")
    profile = profile.join(counts, on="id").fillna({"represented_starts": 0})
    return profile.astype({"eligible_days": int, "positive_days": int, "represented_starts": int})


def allocation_score(assignment, metrics, fold_count):
    #### Penalise imbalance and reject groups without both outcomes or represented starts.
    capacities = np.bincount(assignment, minlength=fold_count).astype(float)
    totals = np.zeros((fold_count, metrics.shape[1]), dtype=float)
    np.add.at(totals, assignment, metrics)
    expected = capacities[:, None] / capacities.sum() * metrics.sum(axis=0)[None, :]
    score = np.square((totals - expected) / np.maximum(expected, 1)).sum()
    invalid = (totals[:, 1] <= 0) | (totals[:, 0] <= totals[:, 1]) | (totals[:, 2] <= 0)
    return float(score + 1_000_000 * invalid.sum())


def improve_assignment(assignment, metrics, fold_count):
    #### Improve a seeded allocation using at most twenty deterministic participant-swap passes.
    best = assignment.copy()
    best_score = allocation_score(best, metrics, fold_count)
    for _ in range(protocol.MAX_SWAP_PASSES):
        candidate_best, candidate_score = best, best_score
        for first in range(len(best)):
            for second in range(first + 1, len(best)):
                if best[first] == best[second]:
                    continue
                candidate = best.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                score = allocation_score(candidate, metrics, fold_count)
                if score + 1e-12 < candidate_score:
                    candidate_best, candidate_score = candidate, score
        if candidate_score + 1e-12 >= best_score:
            break
        best, best_score = candidate_best, candidate_score
    return best


def assign_folds(profile, fold_count, seed):
    #### Make capacity-balanced folds with deterministic greedy allocation and improving swaps.
    if profile.id.duplicated().any() or len(profile) < fold_count:
        raise ValueError("Need unique participants and enough people for every fold")
    profile = profile.sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    capacities = np.full(fold_count, len(profile) // fold_count, dtype=int)
    capacities[:len(profile) % fold_count] += 1
    metrics = profile[["eligible_days", "positive_days", "represented_starts"]].to_numpy(dtype=float)
    targets = capacities[:, None] / len(profile) * metrics.sum(axis=0)[None, :]
    order = profile.assign(tie=rng.random(len(profile))).sort_values(
        ["represented_starts", "positive_days", "eligible_days", "tie"],
        ascending=[False, False, False, True], kind="mergesort").index
    assignment = np.full(len(profile), -1, dtype=int)
    counts, totals = np.zeros(fold_count, dtype=int), np.zeros_like(targets)
    for person in order:
        choices = rng.permutation(np.flatnonzero(counts < capacities))
        scores = []
        for fold in choices:
            projected = totals.copy()
            projected[fold] += metrics[person]
            scores.append(np.square((projected - targets) / np.maximum(targets, 1)).sum())
        chosen = choices[int(np.argmin(scores))]
        assignment[person] = chosen
        counts[chosen] += 1
        totals[chosen] += metrics[person]
    assignment = improve_assignment(assignment, metrics, fold_count)
    return profile[["id"]].assign(fold=assignment + 1)


def validate_assignment(data, assignments, column, count):
    #### Require one saved fold per eligible participant and all planned folds to be present.
    if assignments.id.duplicated().any() or set(assignments.id) != set(data.id):
        raise ValueError("Fold assignment must contain every participant exactly once")
    if set(assignments[column]) != set(range(1, count + 1)):
        raise ValueError("Participant assignment has missing or unexpected folds")


def cohort_counts(data, links, stage, test_fold, validation_fold):
    #### Check outcome classes, event counts and observed predictors in one planned cohort.
    positive = int(data[protocol.TARGET].sum())
    events = len(represented_events(data, links))
    observed = data[protocol.feature_columns("combined")].notna().sum()
    feasible = positive > 0 and len(data) > positive and events > 0 and observed.gt(0).all()
    record = {"stage": stage, "test_fold": test_fold, "validation_fold": validation_fold,
              "participants": data.id.nunique(), "eligible_days": len(data), "positive_days": positive,
              "negative_days": len(data) - positive, "represented_starts": events,
              "participants_with_positive_days": data.loc[data[protocol.TARGET].eq(1), "id"].nunique(),
              "all_predictors_observed": bool(observed.gt(0).all()), "feasible": bool(feasible)}
    availability = [{"stage": stage, "test_fold": test_fold, "validation_fold": validation_fold,
                     "feature": name, "observed_days": int(value)} for name, value in observed.items()]
    return record, availability


def audit_folds(data, links, test, validation):
    #### Verify participant separation and count all forty assessment and training cohorts.
    validate_assignment(data, test, "test_fold", protocol.TEST_FOLDS)
    if set(validation.test_fold) != set(range(1, protocol.TEST_FOLDS + 1)):
        raise ValueError("Missing development-set validation assignment")
    rows, availability = [], []
    for fold in range(1, protocol.TEST_FOLDS + 1):
        test_ids = set(test.loc[test.test_fold.eq(fold), "id"])
        testing, development = data.loc[data.id.isin(test_ids)], data.loc[~data.id.isin(test_ids)]
        inner = validation.loc[validation.test_fold.eq(fold)]
        validate_assignment(development, inner, "validation_fold", protocol.VALIDATION_FOLDS)
        cohorts = [("test", 0, testing), ("development", 0, development)]
        for number in range(1, protocol.VALIDATION_FOLDS + 1):
            validation_ids = set(inner.loc[inner.validation_fold.eq(number), "id"])
            if test_ids & validation_ids:
                raise ValueError("Test participants leaked into validation")
            assessing = development.loc[development.id.isin(validation_ids)]
            training = development.loc[~development.id.isin(validation_ids)]
            if set(training.id) & (test_ids | validation_ids):
                raise ValueError("Training participants overlap an assessment cohort")
            cohorts.extend([("validation", number, assessing), ("training", number, training)])
        for stage, number, cohort in cohorts:
            record, observed = cohort_counts(cohort, links, stage, fold, number)
            rows.append(record)
            availability.extend(observed)
    return pd.DataFrame(rows), pd.DataFrame(availability)


def build_folds(data, links):
    #### Save newly balanced assignments for this changed outcome, independent of model scores.
    test = assign_folds(participant_profile(data, links), protocol.TEST_FOLDS, protocol.SEED)
    test = test.rename(columns={"fold": "test_fold"})
    parts = []
    for fold in range(1, protocol.TEST_FOLDS + 1):
        development = data.loc[~data.id.isin(test.loc[test.test_fold.eq(fold), "id"])]
        inner = assign_folds(participant_profile(development, links), protocol.VALIDATION_FOLDS,
                             protocol.SEED + fold).rename(columns={"fold": "validation_fold"})
        parts.append(inner.assign(test_fold=fold))
    validation = pd.concat(parts, ignore_index=True)
    counts, observed = audit_folds(data, links, test, validation)
    return test, validation, counts, observed
