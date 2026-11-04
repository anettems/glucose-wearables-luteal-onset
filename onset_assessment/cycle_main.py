#### Run a reproducible glucose phase and successive-cycle assessment from verified 2022 data.
#### Save a design record before calculating results, audit every bounded cycle and retain
#### participant-level estimates. No model is fitted and no missing glucose value is replaced.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from onset_assessment.cycle_analysis import (SIGNALS, adjacent_pairs, cycle_contrasts,
    effect_summary, identify_cycles, participant_effects, phase_summaries,
    repeat_summary, source_day_data)
from onset_assessment.cycle_plots import plot_effects, plot_repeatability
from onset_assessment.main import check_destination, load_inputs, write_json
from onset_prediction.protocol import ROOT
from onset_prediction.freezing import verify_hashes
from preprocessing import file_sha256


def report(results, coverage, cycles, effects, repeats, sensitivity):
    #### Explain phase differences, successive-cycle stability and exclusions using saved results.
    lines = ['# Glucose across phases and successive cycles', '',
        f'The full timeline contains {len(coverage)} participants. Quality-approved glucose is available '
        f'for {int(coverage.glucose_days.sum())} source-calendar days.',
        f'{len(cycles)} intervals are bounded by two exact menstrual starts; '
        f'{int(cycles.accepted.sum())} have all four phases in order with no missing labels/dates.',
        f'{int(coverage.outside_bounded_days.sum())} timeline days lie outside these bounded intervals '
        '(including leading/trailing partial cycles). They are retained in the coverage audit.', '',
        '## Design', '',
        'Use the full glucose-available cohort, independently of wearable or model eligibility. '
        'Glucose is matched to its actual measurement day, not the next-day prediction label. '
        'The existing quality-approved aligned export supplies glucose; a source day absent from '
        'that export cannot enter this assessment. No new quality checks or interpolation are applied.',
        'An exact menstrual start requires a menstrual label immediately after a known non-menstrual '
        'label on the preceding calendar day. Consecutive starts bound a cycle; the next start belongs '
        'to the next cycle. A menstrual block already underway at recording start is not assumed exact.',
        'Each phase needs at least three observed days and 50% of its labelled days. Compare each '
        'eligible phase with the same cycle’s eligible follicular phase. Average cycle contrasts '
        'within each person, then give people equal weight. Sensitivity analysis requires five days '
        'and the same 50% coverage. The median-of-days contrast checks sensitivity to unusual days.',
        'The personal-deviation analysis is secondary: it reuses the unchanged past-only 14-day '
        'reference (at least seven days), shifted with its glucose source day. Its availability and '
        'moving reference differ from raw glucose; it is not an independent replication.',
        'Intervals use 2,000 participant resamples (seed 20261021). They describe uncertainty in '
        'group averages, not the range of individual effects; related intervals are not multiplicity-adjusted.', '',
        '## Phase differences', '',
        '| Measurement | Phase minus follicular | Participants / cycles | Mean difference [95% interval] | Increased / decreased |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in effects.itertuples():
        lines.append(f'| {SIGNALS[row.measurement]} | {row.phase} | {row.participants} / {row.cycles} | '
                     f'{row.difference:+.3f} [{row.difference_ci_lower:+.3f}, {row.difference_ci_upper:+.3f}] | '
                     f'{row.increased} / {row.decreased} |')
    lines += ['', '![Phase differences](phase_differences.png)', '', '## Successive-cycle repeatability', '',
        'Only adjacent bounded cycles with both phase contrasts are paired. Rejected or unavailable '
        'cycles are never bridged. Same direction means both contrasts are positive or both negative; '
        'values within 1e-12 of zero do not contribute to sign agreement. Participants receive equal '
        'weight if they have more than one pair. Agreement is descriptive, not classification accuracy '
        'or evidence of stability beyond a shared group tendency. A 50% line is not a formal chance null.', '',
        '| Measurement | Phase | Participants / pairs | Same-direction proportion [95% interval] | Mean absolute change |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in repeats.itertuples():
        lines.append(f'| {SIGNALS[row.measurement]} | {row.phase} | {row.participants} / {row.pairs} | '
                     f'{row.agreement:.1%} [{row.agreement_ci_lower:.1%}, {row.agreement_ci_upper:.1%}] | '
                     f'{row.mean_absolute_change:.3f} |')
    lines += ['', '![Successive cycles](successive_cycle_agreement.png)', '',
        '## Stricter five-day coverage sensitivity', '',
        '| Measurement | Luteal minus follicular: participants / cycles | Difference [95% interval] |',
        '| --- | ---: | ---: |']
    for row in sensitivity.loc[sensitivity.phase.eq('luteal')].itertuples():
        lines.append(f'| {SIGNALS[row.measurement]} | {row.participants} / {row.cycles} | '
                     f'{row.difference:+.3f} [{row.difference_ci_lower:+.3f}, {row.difference_ci_upper:+.3f}] |')
    raw = effects.loc[effects.measurement.eq('glucose_median_mmol_l')
                      & effects.phase.eq('luteal')].iloc[0]
    deviation = effects.loc[effects.measurement.eq('glucose_median_mmol_l__deviation')
                            & effects.phase.eq('luteal')].iloc[0]
    repeat = repeats.loc[repeats.measurement.eq('glucose_median_mmol_l')
                         & repeats.phase.eq('luteal')].iloc[0]
    lines += ['', '## Findings and next steps', '',
        f'Raw luteal glucose is {raw.difference:+.3f} mmol/L higher on average than the same '
        f'cycle’s follicular glucose, but its interval [{raw.difference_ci_lower:+.3f}, '
        f'{raw.difference_ci_upper:+.3f}] includes zero. Only {raw.increased} of '
        f'{raw.participants} participant-average effects are positive. The stricter five-day '
        'and median-of-days summaries should be read alongside this result.',
        f'The secondary personal-deviation contrast is larger ({deviation.difference:+.3f} mmol/L; '
        f'interval [{deviation.difference_ci_lower:+.3f}, {deviation.difference_ci_upper:+.3f}]). '
        'This supports a possible broad phase-related change relative to recent personal history. '
        'The rolling reference, differing coverage and multiple summaries prevent treating it as '
        'proof of a stable individual marker.',
        f'The raw luteal difference keeps the same sign in {repeat.agreement:.1%} of successive '
        f'comparisons ({repeat.pairs} pairs from {repeat.participants} participants; interval '
        f'{repeat.agreement_ci_lower:.1%} to {repeat.agreement_ci_upper:.1%}). Its mean absolute '
        f'change between cycles is {repeat.mean_absolute_change:.3f} mmol/L. These data do not '
        'demonstrate a dependable personal luteal-rise signature.',
        'Prioritise a longer-history coverage audit before fitting any sequence model. This analysis '
        'supports investigating broad phase physiology, but provides limited support for using the '
        'previous cycle’s glucose contrast to predict the next one. A larger model cannot be assumed '
        'to overcome cycle-to-cycle variation. No longer-history model has been evaluated here.', '',
        '## Interpretation and practical limits', '',
        'A positive phase average can coexist with weak daily classification and unstable effects '
        'between cycles. The successive-cycle scatter should be read alongside its small paired cohort, '
        'coverage, effect sizes and uncertainty. This analysis does not estimate future-onset prediction.',
        'Phase labels are supplied reference categories; fertility is the supplied fertile-window '
        'category, not an independently confirmed ovulation day. Menstrual starts and complete-cycle '
        'selection are retrospective. Food, activity, sensor changes and other time-varying influences '
        'are not adjusted for, so effects cannot be attributed solely to reproductive hormones.',
        'Short phases and missing CGM can exclude comparisons. Different phases and the stricter '
        'sensitivity can involve different participants. Apparent changes across these analyses are '
        'not paired estimates of the effect of a coverage rule.',
        'Participant coverage, all bounded-cycle audits, phase coverage, cycle contrasts, participant '
        'effects, adjacent pairs and sensitivity summaries are saved as CSV files. Leading/trailing '
        'partial cycles are not forced into the repeatability analysis.', '']
    (results / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(results, processed_output):
    #### Verify inputs, save the design and export all descriptive tables and figures afresh.
    check_destination(results, ROOT.parent / 'results')
    check_destination(processed_output, ROOT / 'data/processed')
    (audit, _, _), inputs = load_inputs(ROOT / 'data/processed/onset_prediction_2022_v1',
                                       ROOT.parent / 'results/onset_prediction_2022_v1')
    results.mkdir(parents=True, exist_ok=True)
    processed_output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('cycle_analysis.py'),
               Path(__file__).with_name('cycle_plots.py'), Path(__file__).with_name('analysis.py'),
               Path(__file__).with_name('main.py'), Path(__file__).with_name('README.md'),
               Path(__file__).parent / 'tests/test_cycles.py']
    write_json(results / 'assessment_protocol.json', {
        'version': '2026-11-04-glucose-cycle-consistency-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
        'sources': {str(path): file_sha256(path) for path in sources},
        'cohort': 'all quality-approved 2022 glucose; no wearable/history eligibility restriction',
        'cycle_rule': 'exact consecutive-day menstrual starts; all four phases in order; no label gaps',
        'phase_minimum_days': 3, 'phase_minimum_fraction': .5, 'sensitivity_minimum_days': 5,
        'reference': 'same-cycle follicular', 'weighting': 'cycles within participant; equal participants',
        'repeatability': 'adjacent bounded cycles; equal-participant sign agreement and absolute change',
        'glucose_alignment': 'actual source calendar day; not next-day prediction label',
        'signals': SIGNALS, 'resamples': 2000, 'seed': 20261021,
        'versions': {'python': platform.python_version(), **{
            name: version(name) for name in ['numpy', 'pandas', 'matplotlib']}}})
    days = source_day_data(audit)
    cycles, assigned = identify_cycles(days)
    phases = phase_summaries(assigned)
    contrasts = cycle_contrasts(phases)
    people = participant_effects(contrasts)
    effects = effect_summary(people)
    pairs = adjacent_pairs(contrasts)
    repeats = repeat_summary(pairs)
    strict = cycle_contrasts(phase_summaries(assigned, minimum=5))
    strict_summary = effect_summary(participant_effects(strict))
    coverage = days.groupby('id', as_index=False).agg(
        timeline_days=('day_in_study', 'size'), labelled_days=('phase', 'count'),
        glucose_days=('glucose_median_mmol_l', 'count'))
    bounded = assigned.groupby('id').size()
    coverage['bounded_days'] = coverage.id.map(bounded).fillna(0).astype(int)
    coverage['outside_bounded_days'] = coverage.timeline_days - coverage.bounded_days
    days.to_csv(processed_output / 'source_day_glucose.csv', index=False)
    assigned.to_csv(processed_output / 'cycle_days.csv', index=False)
    tables = {'participant_coverage': coverage, 'cycle_audit': cycles, 'phase_coverage': phases,
              'cycle_contrasts': contrasts, 'participant_effects': people, 'phase_summary': effects,
              'adjacent_cycle_pairs': pairs, 'repeatability_summary': repeats,
              'five_day_cycle_contrasts': strict, 'five_day_phase_summary': strict_summary}
    for name, table in tables.items():
        table.to_csv(results / (name + '.csv'), index=False)
    plot_effects(people, effects, results)
    plot_repeatability(pairs, results)
    report(results, coverage, cycles, effects, repeats, strict_summary)
    verify_hashes(inputs)
    files = [path for folder in [processed_output, results] for path in folder.iterdir() if path.is_file()]
    write_json(results / 'manifest.json', {'status': 'complete',
        'files': {str(path): file_sha256(path) for path in files}})
    print(effects.to_string(index=False))
    print(repeats.to_string(index=False))


def main():
    #### Resolve fresh generated-output paths without changing the primary model protocol.
    parser = argparse.ArgumentParser(description='Assess glucose phase and cycle consistency')
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT.parent / 'results/glucose_cycle_assessment_2022_v1')
    parser.add_argument('--processed-output-dir', type=Path,
                        default=ROOT / 'data/processed/glucose_cycle_assessment_2022_v1')
    args = parser.parse_args()
    run(args.results_dir.resolve(), args.processed_output_dir.resolve())


if __name__ == '__main__':
    main()
