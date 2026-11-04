#### Describe glucose phase contrasts within fully observed menstrual cycles in 2022.
#### Use existing quality-approved CGM on its actual source day, independently of wearable
#### eligibility. Bound cycles by consecutive exact menstrual-label starts. Audit incomplete,
#### missing-label and non-sequential cycles. Compare phases with the same cycle's follicular
#### phase, then average cycles within participants. Assess successive-cycle agreement without
#### selecting participants by effect direction. These are descriptive, retrospective analyses.

import numpy as np
import pandas as pd

from onset_assessment.analysis import bootstrap_means
from onset_prediction.protocol import KEY
from onset_prediction.outcomes import validate_keys


PHASES = ['menstrual', 'follicular', 'fertility', 'luteal']
SIGNALS = {'glucose_median_mmol_l': 'Median glucose (mmol/L)',
           'glucose_cv_percent': 'Glucose variability (percentage points)',
           'glucose_median_mmol_l__deviation': 'Personal glucose deviation (mmol/L)'}
CYCLE_KEY = ['id', 'study_interval', 'cycle_start']


def source_day_data(audit):
    #### Align approved glucose and its existing past-only deviation to its measurement day.
    validate_keys(audit)
    if not audit.phase.dropna().isin(PHASES).all():
        raise ValueError('Unknown phase label')
    observed = audit.glucose_median_mmol_l.notna()
    if not audit.loc[observed, 'prior_day_source_day'].eq(
            audit.loc[observed, 'day_in_study'] - 1).all():
        raise ValueError('Invalid glucose source-day alignment')
    glucose = audit.loc[observed, KEY + list(SIGNALS)].copy()
    glucose.day_in_study -= 1
    if np.isinf(glucose[list(SIGNALS)].to_numpy(float)).any():
        raise ValueError('Infinite glucose measurement')
    return audit[KEY + ['phase']].merge(glucose, on=KEY, how='left', validate='one_to_one')


def identify_cycles(days):
    #### Retain every bounded interval and flag gaps, unknown labels and phase reversals.
    validate_keys(days)
    records, assigned = [], []
    for (person, interval), group in days.groupby(['id', 'study_interval']):
        group = group.sort_values('day_in_study')
        previous = group.phase.shift()
        consecutive = group.day_in_study.diff().eq(1)
        starts = group.loc[group.phase.eq('menstrual') & previous.notna()
                           & previous.ne('menstrual') & consecutive, 'day_in_study'].tolist()
        for ordinal, (start, end) in enumerate(zip(starts[:-1], starts[1:]), 1):
            selected = group.loc[group.day_in_study.ge(start) & group.day_in_study.lt(end)].copy()
            codes = selected.phase.map({name: index for index, name in enumerate(PHASES)})
            complete = len(selected) == end - start and selected.phase.notna().all()
            ordered = complete and codes.diff().dropna().ge(0).all()
            all_phases = set(selected.phase.dropna()) == set(PHASES)
            accepted = bool(complete and ordered and all_phases)
            reason = ('accepted' if accepted else 'missing_days_or_labels' if not complete
                      else 'phase_reversal' if not ordered else 'missing_phase')
            records.append({'id': person, 'study_interval': interval, 'cycle_start': start,
                            'next_start': end, 'cycle_number': ordinal, 'length_days': end - start,
                            'accepted': accepted, 'reason': reason})
            selected['cycle_start'], selected['cycle_number'] = start, ordinal
            selected['accepted'] = accepted
            assigned.append(selected)
    if not records:
        raise ValueError('No cycles bounded by exact menstrual starts')
    return pd.DataFrame(records), pd.concat(assigned, ignore_index=True)


def phase_summaries(assigned, minimum=3):
    #### Require the stated observed-day minimum and at least half of each labelled phase.
    long = assigned.loc[assigned.accepted].melt(
        id_vars=CYCLE_KEY + ['cycle_number', 'day_in_study', 'phase'],
        value_vars=list(SIGNALS), var_name='measurement', value_name='value')
    result = long.groupby(CYCLE_KEY + ['cycle_number', 'measurement', 'phase']).value.agg(
        mean='mean', median='median', observed_days='count', labelled_days='size').reset_index()
    result['eligible'] = result.observed_days.ge(minimum) & (
        2 * result.observed_days >= result.labelled_days)
    return result


def cycle_contrasts(phases):
    #### Subtract each cycle's eligible follicular mean and median from its other phases.
    eligible = phases.loc[phases.eligible].copy()
    baseline = eligible.loc[eligible.phase.eq('follicular'), CYCLE_KEY +
                            ['measurement', 'mean', 'median']].rename(
                                columns={'mean': 'reference_mean', 'median': 'reference_median'})
    joined = eligible.loc[eligible.phase.ne('follicular')].merge(
        baseline, on=CYCLE_KEY + ['measurement'], validate='many_to_one')
    joined['difference'] = joined['mean'] - joined.reference_mean
    joined['median_difference'] = joined['median'] - joined.reference_median
    return joined


def participant_effects(contrasts):
    #### Average available paired cycles equally within each participant before group inference.
    return contrasts.groupby(['id', 'measurement', 'phase'], as_index=False).agg(
        difference=('difference', 'mean'), median_difference=('median_difference', 'mean'),
        cycles=('difference', 'size'))


def effect_summary(participants):
    #### Bootstrap participants for average phase differences and describe direction consistency.
    rows = []
    for (measurement, phase), group in participants.groupby(['measurement', 'phase']):
        intervals = bootstrap_means(group[['difference', 'median_difference']].to_numpy())
        row = {'measurement': measurement, 'phase': phase, 'participants': len(group),
               'cycles': int(group.cycles.sum()),
               'increased': int(group.difference.gt(1e-12).sum()),
               'decreased': int(group.difference.lt(-1e-12).sum())}
        for index, column in enumerate(['difference', 'median_difference']):
            row[column] = group[column].mean()
            row[column + '_ci_lower'] = intervals.iloc[index].ci_lower
            row[column + '_ci_upper'] = intervals.iloc[index].ci_upper
        rows.append(row)
    return pd.DataFrame(rows)


def adjacent_pairs(contrasts):
    #### Pair successive bounded cycles only; never bridge a rejected or unmeasured cycle.
    keys = ['id', 'study_interval', 'measurement', 'phase', 'cycle_number']
    earlier = contrasts[keys + ['cycle_start', 'difference']].copy()
    earlier.cycle_number += 1
    earlier = earlier.rename(columns={'cycle_start': 'earlier_start', 'difference': 'earlier'})
    pairs = contrasts.merge(earlier, on=keys, validate='one_to_one').rename(
        columns={'difference': 'later'})
    pairs['nonzero_pair'] = pairs.earlier.abs().gt(1e-12) & pairs.later.abs().gt(1e-12)
    pairs['same_direction'] = np.where(pairs.nonzero_pair,
                                      np.sign(pairs.earlier).eq(np.sign(pairs.later)), np.nan)
    pairs['absolute_change'] = (pairs.later - pairs.earlier).abs()
    return pairs


def repeat_summary(pairs):
    #### Give participants equal weight in adjacent-cycle agreement and effect-size changes.
    rows = []
    for (measurement, phase), group in pairs.groupby(['measurement', 'phase']):
        people = group.groupby('id')[['same_direction', 'absolute_change']].mean()
        intervals = bootstrap_means(people.to_numpy())
        rows.append({'measurement': measurement, 'phase': phase, 'participants': len(people),
                     'pairs': len(group), 'nonzero_pairs': int(group.nonzero_pair.sum()),
                     'agreement_participants': int(people.same_direction.notna().sum()),
                     'agreement': people.same_direction.mean(),
                     'agreement_ci_lower': intervals.iloc[0].ci_lower,
                     'agreement_ci_upper': intervals.iloc[0].ci_upper,
                     'mean_absolute_change': people.absolute_change.mean(),
                     'change_ci_lower': intervals.iloc[1].ci_lower,
                     'change_ci_upper': intervals.iloc[1].ci_upper})
    return pd.DataFrame(rows)
