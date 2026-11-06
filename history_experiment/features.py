#### Build one past-only glucose contrast from the quality-approved 2022 timeline.
#### Compare daily median glucose in the most recent and oldest weeks of 21 aligned
#### calendar days. Keep missing days explicit and retain counts for the eligibility audit.

import pandas as pd

from onset_assessment.history_coverage import availability
from onset_prediction.protocol import KEY


FEATURE = 'glucose_recent_minus_oldest_week_mmol_l'
MIN_HISTORY = 15
MIN_WEEK = 4


def build_history(audit):
    #### Calculate week summaries before applying the shared cohort or outcome requirements.
    observed = availability(audit)
    source = observed[KEY].assign(glucose=audit.glucose_median_mmol_l.where(observed.glucose))
    parts = []
    for _, person in source.groupby(KEY[:2]):
        days = person.day_in_study
        calendar = person.set_index('day_in_study').glucose.reindex(range(days.min(), days.max() + 1))
        #### Shift past t before taking rolling statistics. The oldest week ends at t-15.
        past = calendar.shift(1)
        recent = past.rolling(7, min_periods=MIN_WEEK)
        oldest = calendar.shift(15).rolling(7, min_periods=MIN_WEEK)
        counts = past.notna().astype(int)
        stats = pd.DataFrame({
            'history_days': counts.rolling(21, min_periods=1).sum().astype(int),
            'recent_week_days': counts.rolling(7, min_periods=1).sum().astype(int),
            'oldest_week_days': counts.shift(14, fill_value=0).rolling(7, min_periods=1).sum().astype(int),
            FEATURE: recent.median() - oldest.median(),
        })
        stats[FEATURE] = stats[FEATURE].where(stats.history_days.ge(MIN_HISTORY))
        parts.append(person[KEY].join(stats, on='day_in_study', validate='one_to_one'))
    return pd.concat(parts, ignore_index=True)
