#### Generate compact thesis figures from the verified, completed glucose cycle assessment.
#### Read saved participant effects, phase summaries and adjacent-cycle pairs; never refit or
#### recalculate analysis outcomes. Export vector PDF and 300 dpi PNG with consistent typography.
#### The main figure presents raw glucose; supporting panels present variability and the
#### existing personal deviation. Record input/source hashes and optionally export new LaTeX assets.

import argparse
from importlib.metadata import version
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from onset_assessment.main import check_destination, write_json
from onset_prediction.freezing import read_json, verify_hashes
from onset_prediction.protocol import ROOT
from preprocessing import file_sha256


BLUE = '#356783'
ORANGE = '#B75625'
PHASES = ['menstrual', 'fertility', 'luteal']
STYLE = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .6,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}


def clean_axes(ax):
    #### Keep reference lines visible without boxing in the observations.
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.6, length=3)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E8E8E8', linewidth=.5)


def effect_panel(ax, people, summary, measurement, title, unit):
    #### Plot participant effects with deterministic spacing and the saved mean intervals.
    clean_axes(ax)
    for index, phase in enumerate(PHASES):
        values = people.loc[people.measurement.eq(measurement) & people.phase.eq(phase)]
        row = summary.loc[summary.measurement.eq(measurement) & summary.phase.eq(phase)].iloc[0]
        if len(values) != row.participants or not np.isclose(values.difference.mean(), row.difference):
            raise ValueError('Saved phase mean or participant count does not match plotted values')
        offsets = np.linspace(-.15, .15, len(values))
        ax.scatter(index + offsets, values.difference, s=15, alpha=.7, color=BLUE,
                   linewidths=.3, edgecolors='white', zorder=3)
        ax.errorbar(index + .24, row.difference,
                    yerr=[[row.difference - row.difference_ci_lower],
                          [row.difference_ci_upper - row.difference]],
                    fmt='D', markersize=4, color=ORANGE, capsize=3, linewidth=1.2, zorder=4)
    ax.axhline(0, color='#444444', linewidth=.8, zorder=2)
    ax.set_xticks(range(3))
    counts = [int(summary.loc[summary.measurement.eq(measurement) & summary.phase.eq(phase),
                              'participants'].iloc[0]) for phase in PHASES]
    ax.set_xticklabels([f'{label}\n$n={count}$' for label, count in
                        zip(['Menstrual', 'Fertile window', 'Luteal'], counts)])
    ax.set_xlim(-.4, 2.58)
    ax.set_ylabel('Difference from follicular phase\n' + unit)
    ax.set_title(title, loc='left', pad=9)


def repeat_panel(ax, pairs, measurement, title, unit):
    #### Show observed adjacent-cycle effects at equal scales, without participant identifiers.
    clean_axes(ax)
    data = pairs.loc[pairs.measurement.eq(measurement) & pairs.phase.eq('luteal')]
    if data.empty or not np.isfinite(data[['earlier', 'later']].to_numpy()).all():
        raise ValueError('Expected finite adjacent-cycle comparisons')
    bound = max(.1, np.abs(data[['earlier', 'later']].to_numpy()).max()) * 1.15
    ax.plot([-bound, bound], [-bound, bound], '--', color='#999999', linewidth=.8, zorder=1)
    ax.axhline(0, color='#555555', linewidth=.7)
    ax.axvline(0, color='#555555', linewidth=.7)
    ax.scatter(data.earlier, data.later, s=28, color=BLUE, edgecolors='white', linewidths=.5, zorder=3)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
           xlabel='Earlier cycle: luteal minus follicular\n' + unit,
           ylabel='Next cycle: luteal minus follicular\n' + unit)
    ax.set_title(title, loc='left', pad=9)
    nonzero = data.loc[data.nonzero_pair]
    same = int(np.sign(nonzero.earlier).eq(np.sign(nonzero.later)).sum())
    ax.text(.98, .02, f'{len(data)} pairs; {same}/{len(nonzero)} same direction',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2, 'alpha': .9})


def export_figure(fig, destination, stem):
    #### Export PDF text as embedded fonts and a matching high-resolution preview.
    for suffix in ['pdf', 'png']:
        path = destination / f'{stem}.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def draw_figures(people, summary, pairs, destination):
    #### Produce a two-panel main figure and four supporting panels at thesis column width.
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
        fig.subplots_adjust(left=.09, right=.99, bottom=.23, top=.87, wspace=.58)
        effect_panel(axes[0], people, summary, 'glucose_median_mmol_l',
                     '(a) Phase differences', '(mmol/L)')
        repeat_panel(axes[1], pairs, 'glucose_median_mmol_l',
                     '(b) Successive cycles', '(mmol/L)')
        handles = [Line2D([], [], marker='o', linestyle='', color=BLUE, markersize=4),
                   Line2D([], [], marker='D', linestyle='', color=ORANGE, markersize=4)]
        fig.legend(handles, ['Participant mean', 'Group mean and 95% interval'],
                   loc='lower center', bbox_to_anchor=(.46, -.015), ncol=2,
                   frameon=False, fontsize=8, handletextpad=.4, columnspacing=1.3)
        export_figure(fig, destination, 'glucose_cycle_main_v2')

        fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))
        fig.subplots_adjust(left=.1, right=.99, bottom=.1, top=.94, wspace=.58, hspace=.58)
        measurements = [('glucose_cv_percent', '(percentage points)', 'Glucose variability'),
                        ('glucose_median_mmol_l__deviation', '(mmol/L)', 'Personal deviation')]
        for column, (measurement, unit, title) in enumerate(measurements):
            effect_panel(axes[0, column], people, summary, measurement,
                         f'({chr(97 + column)}) {title}', unit)
            repeat_panel(axes[1, column], pairs, measurement,
                         f'({chr(99 + column)}) Successive cycles', unit)
        export_figure(fig, destination, 'glucose_cycle_support_v2')


def run(assessment, destination, latex_figures=None):
    #### Verify historical results and save fresh figures, optionally copying new thesis assets.
    check_destination(destination, ROOT.parent / 'results')
    manifest = read_json(assessment / 'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Expected a completed cycle assessment')
    verify_hashes(manifest['files'])
    names = ['participant_effects', 'phase_summary', 'adjacent_cycle_pairs']
    paths = [assessment / (name + '.csv') for name in names]
    for path in paths:
        if manifest['files'].get(str(path.resolve())) != file_sha256(path):
            raise ValueError('Figure input is outside the verified assessment')
    destination.mkdir(parents=True, exist_ok=True)
    draw_figures(*(pd.read_csv(path) for path in paths), destination)
    exported = []
    if latex_figures is not None:
        allowed = (ROOT.parent / 'latex/figures').resolve()
        if latex_figures.resolve() != allowed:
            raise ValueError('LaTeX assets must go to the thesis figures directory')
        for source in destination.glob('*.pdf'):
            target = latex_figures / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(source, target)
            exported.append(target)
    inputs = {str(path.resolve()): file_sha256(path) for path in paths + [assessment / 'manifest.json']}
    outputs = list(destination.glob('*.pdf')) + list(destination.glob('*.png')) + exported
    sources = [Path(__file__), Path(__file__).parent / 'tests/test_thesis_figures.py']
    write_json(destination / 'figure_manifest.json', {
        'inputs': inputs, 'sources': {str(path): file_sha256(path) for path in sources},
        'outputs': {str(path.resolve()): file_sha256(path) for path in outputs},
        'versions': {name: version(name) for name in ['numpy', 'pandas', 'matplotlib']},
        'description': 'Saved estimates only; no refitting, resampling or outcome changes',
        'style': STYLE, 'png_dpi': 300})


def main():
    #### Accept explicit result and optional LaTeX export destinations for reproducible generation.
    parser = argparse.ArgumentParser(description='Create clean glucose cycle thesis figures')
    parser.add_argument('--assessment-dir', type=Path,
                        default=ROOT.parent / 'results/glucose_cycle_assessment_2022_v2')
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT.parent / 'results/glucose_thesis_figures_2022_v2')
    parser.add_argument('--latex-figures-dir', type=Path)
    args = parser.parse_args()
    run(args.assessment_dir.resolve(), args.results_dir.resolve(), args.latex_figures_dir)


if __name__ == '__main__':
    main()
