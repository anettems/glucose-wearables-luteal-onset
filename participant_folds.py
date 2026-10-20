"""Deterministic participant folds and pre-evaluation feasibility checks."""

import numpy as np
import pandas as pd

import primary_protocol as protocol
from model_inputs import WEARABLE_FEATURES
from onset_alerts import reference_evaluability


def participant_profile(observations, references):
    """Summarise the outcome and onset burden used to balance participant folds."""
    grouped = observations.groupby("id", sort=True)
    profile = grouped.agg(
        eligible_days=("luteal_status", "size"),
        luteal_days=("luteal_status", "sum"),
    ).reset_index()
    profile["non_luteal_days"] = profile["eligible_days"] - profile["luteal_days"]
    starts = references.groupby("id", sort=True).size().rename("reference_starts")
    profile = profile.join(starts, on="id").fillna({"reference_starts": 0})
    profile["reference_starts"] = profile["reference_starts"].astype(int)
    return profile


def allocation_score(assignments, metrics, fold_count):
    """Score a participant assignment by imbalance in prespecified quantities."""
    capacities = np.array([np.sum(assignments == fold) for fold in range(fold_count)], dtype=float)
    targets = capacities / capacities.sum()
    totals = np.zeros((fold_count, metrics.shape[1]), dtype=float)
    np.add.at(totals, assignments, metrics)
    expected = targets[:, None] * metrics.sum(axis=0)[None, :]
    scales = np.maximum(expected, 1.0)
    imbalance = np.square((totals - expected) / scales).sum()
    invalid = (totals[:, 0] <= 0) | (totals[:, 1] <= 0) | (totals[:, 0] <= totals[:, 1])
    missing_starts = totals[:, 2] <= 0
    return float(imbalance + 1_000_000 * invalid.sum() + 1_000_000 * missing_starts.sum())


def improve_assignment(assignments, metrics, fold_count):
    """Use deterministic improving participant swaps after a seeded random start."""
    best = assignments.copy()
    best_score = allocation_score(best, metrics, fold_count)
    for _ in range(protocol.MAX_FOLD_SWAP_PASSES):
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
    return best, best_score


def assign_balanced_folds(profile, fold_count, seed):
    """Produce a seed-controlled, capacity-balanced participant fold allocation."""
    if profile.id.duplicated().any():
        raise ValueError("Participant profiles must contain each participant once")
    if profile.id.nunique() < fold_count:
        raise ValueError(f"Need at least {fold_count} participants for participant-disjoint folds")
    rng = np.random.default_rng(seed)
    profile = profile.reset_index(drop=True)
    participant_count = len(profile)
    capacities = np.full(fold_count, participant_count // fold_count, dtype=int)
    capacities[:participant_count % fold_count] += 1
    metrics = profile[["eligible_days", "luteal_days", "reference_starts"]].to_numpy(dtype=float)
    targets = capacities[:, None] / participant_count * metrics.sum(axis=0)[None, :]
    scales = np.maximum(targets, 1.0)
    order = profile.assign(_tie=rng.random(participant_count)).sort_values(
        ["reference_starts", "luteal_days", "eligible_days", "_tie"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).index.to_numpy()
    assignment = np.full(participant_count, -1, dtype=int)
    counts = np.zeros(fold_count, dtype=int)
    totals = np.zeros((fold_count, metrics.shape[1]), dtype=float)
    for participant in order:
        candidates = np.flatnonzero(counts < capacities)
        candidate_order = rng.permutation(candidates)
        scores = []
        for fold in candidate_order:
            projected = totals.copy()
            projected[fold] += metrics[participant]
            scores.append(np.square((projected - targets) / scales).sum())
        chosen = candidate_order[int(np.argmin(scores))]
        assignment[participant] = chosen
        counts[chosen] += 1
        totals[chosen] += metrics[participant]
    best_assignment, _ = improve_assignment(assignment, metrics, fold_count)
    result = profile[["id"]].copy()
    result["fold"] = best_assignment + 1
    return result.sort_values("id").reset_index(drop=True)


def fold_counts(observations, references, assignments, fold_column):
    """Return planned feasibility counts for one participant-fold assignment."""
    labelled = observations.merge(assignments[["id", fold_column]], on="id", validate="many_to_one")
    starts = references.merge(assignments[["id", fold_column]], on="id", validate="many_to_one")
    summary = labelled.groupby(fold_column, sort=True).agg(
        participants=("id", "nunique"),
        eligible_days=("luteal_status", "size"),
        luteal_days=("luteal_status", "sum"),
    )
    summary["non_luteal_days"] = summary["eligible_days"] - summary["luteal_days"]
    summary["reference_starts"] = starts.groupby(fold_column).size().reindex(summary.index, fill_value=0)
    evaluable = reference_evaluability(observations, references)
    evaluable = evaluable.loc[evaluable.evaluable].merge(assignments, on="id", validate="many_to_one")
    summary["evaluable_reference_starts"] = evaluable.groupby(fold_column).size().reindex(
        summary.index, fill_value=0,
    )
    return summary.reset_index()


def training_cohort_counts(observations, assignments, fold_column):
    """Check every training complement before any held-out prediction is made."""
    rows = []
    for fold, group in assignments.groupby(fold_column, sort=True):
        training = observations.loc[~observations.id.isin(group.id)]
        observed = training[WEARABLE_FEATURES].notna().sum()
        if set(training.luteal_status) != {0, 1} or observed.eq(0).any():
            raise ValueError(f"Training feasibility failed for {fold_column}={fold}")
        rows.append({fold_column: fold, "training_participants": training.id.nunique(),
                     "training_days": len(training),
                     **{f"training_{name}_observed": int(count) for name, count in observed.items()}})
    return pd.DataFrame(rows)


def validate_fold_assignment(observations, references, assignments, fold_column, fold_count):
    """Reject leakage, missing classes, or folds without an onset reference event."""
    if assignments.id.duplicated().any() or set(assignments.id) != set(observations.id):
        raise ValueError("Fold assignment must contain each eligible participant exactly once")
    expected = set(range(1, fold_count + 1))
    observed = set(assignments[fold_column])
    if observed != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, found {sorted(observed)}")
    counts = fold_counts(observations, references, assignments, fold_column)
    invalid = counts.loc[
        counts.luteal_days.le(0)
        | counts.non_luteal_days.le(0)
        | counts.evaluable_reference_starts.le(0),
    ]
    if not invalid.empty:
        raise ValueError(f"Fold feasibility failed for {fold_column}: {invalid.to_dict('records')}")
    return counts.merge(training_cohort_counts(observations, assignments, fold_column),
                        on=fold_column, validate="one_to_one")


def build_fold_assignments(observations, references):
    """Create and validate all saved outer and nested participant allocations."""
    profile = participant_profile(observations, references)
    outer = assign_balanced_folds(profile, protocol.OUTER_FOLDS, protocol.PRIMARY_SEED).rename(columns={"fold": "outer_test_fold"})
    feasibility = validate_fold_assignment(
        observations, references, outer, "outer_test_fold", protocol.OUTER_FOLDS,
    ).assign(stage="outer")

    inner_parts = []
    for outer_fold in range(1, protocol.OUTER_FOLDS + 1):
        development_ids = outer.loc[outer.outer_test_fold.ne(outer_fold), "id"]
        development = observations.loc[observations.id.isin(development_ids)]
        development_references = references.loc[references.id.isin(development_ids)]
        development_profile = participant_profile(development, development_references)
        inner = assign_balanced_folds(
            development_profile, protocol.INNER_FOLDS, protocol.PRIMARY_SEED + outer_fold,
        ).rename(columns={"fold": "validation_fold"})
        counts = validate_fold_assignment(
            development, development_references, inner, "validation_fold", protocol.INNER_FOLDS,
        )
        counts["stage"] = "validation"
        counts["outer_test_fold"] = outer_fold
        feasibility = pd.concat([feasibility, counts], ignore_index=True)
        inner["outer_test_fold"] = outer_fold
        inner_parts.append(inner)
    inner = pd.concat(inner_parts, ignore_index=True)
    return outer, inner, feasibility
