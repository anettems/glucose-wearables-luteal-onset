#### Generate four thesis figures for the completed 2022 three-day forecasting analysis.
#### Read verified logistic-regression and random-forest results on the same held-out days.
#### Export vector PDFs, 300 dpi previews and a provenance manifest to a fresh results folder.
#### Optionally copy new PDFs into latex/figures. Existing files are never overwritten.
#### The figures describe timing, participant folds, paired AP and fixed-bin calibration.
#### This reporting step does not fit models, alter outcomes or introduce decision thresholds.

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from onset_prediction.figure_data import FAMILIES, digest, load_results
from onset_prediction.figure_diagrams import evaluation_diagram, prediction_timeline


WORKSPACE = Path(__file__).resolve().parents[2]
INPUTS = ('glucose', 'wearable', 'combined')
LABELS = {'glucose': 'Glucose only', 'wearable': 'Wearable only', 'combined': 'Combined'}
FAMILY_LABELS = {'logistic_regression': 'Logistic regression', 'random_forest': 'Random forest'}
COLOURS = {'glucose': '#646464', 'wearable': '#356783', 'combined': '#B75625'}
MARKERS = {'glucose': 's', 'wearable': 'o', 'combined': 'D'}
STEMS = ('onset_prediction_timeline_v2', 'onset_participant_evaluation_v1',
         'onset_predictive_value_v1', 'onset_calibration_v1')
STYLE = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .6,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}


def clean_axes(ax):
    #### Use light reference grids and remove unnecessary chart borders.
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.6, length=3)
    ax.set_axisbelow(True)
    ax.grid(color='#E6E6E6', linewidth=.5)


def interval_point(ax, y, row, colour, marker):
    #### Draw a saved estimate and its asymmetric 95% participant-bootstrap interval.
    ax.plot([row.ci_lower, row.ci_upper], [y, y], color=colour, lw=1.2)
    ax.plot([row.ci_lower, row.ci_upper], [y, y], '|', color=colour, ms=6, mew=.9)
    ax.plot(row.estimate, y, marker, color=colour, ms=4.5)


def performance_figure(tables):
    #### Show absolute AP and the paired glucose contribution on separately labelled axes.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.18, right=.98, bottom=.24, top=.85, wspace=.72)
    for ax in axes:
        clean_axes(ax)
    labels, positions = [], []
    for group, family in enumerate(FAMILIES):
        summary = tables[family]['model_summary']
        for index, name in enumerate(INPUTS):
            y = 6 - group * 4 - index
            row = summary.loc[summary.input_set.eq(name) & summary.metric.eq('average_precision')].iloc[0]
            interval_point(axes[0], y, row, COLOURS[name], MARKERS[name])
            labels.append(LABELS[name])
            positions.append(y)
        axes[0].text(.002, 6.65 - group * 4, FAMILY_LABELS[family], fontsize=8, weight='bold')
        row = tables[family]['paired_comparison'].iloc[0]
        interval_point(axes[1], 1 - group, row, COLOURS['combined'], 'D')
        axes[1].text(.5, (1.23 - group + .55) / 2.2,
                     f'{row.estimate:+.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]',
                     transform=axes[1].transAxes, ha='center', fontsize=7.5)
    prevalence = float(tables[FAMILIES[0]]['model_summary'].prevalence.iloc[0])
    upper_ap = max(tables[family]['model_summary'].query("metric == 'average_precision'")
                   .ci_upper.max() for family in FAMILIES)
    differences = [tables[family]['paired_comparison'].iloc[0] for family in FAMILIES]
    lower = min(0, *(row.ci_lower for row in differences))
    upper = max(0, *(row.ci_upper for row in differences))
    padding = max(.005, (upper - lower) * .12)
    axes[0].axvline(prevalence, color='#646464', ls='--', lw=.8)
    axes[0].set(yticks=positions, yticklabels=labels, ylim=(-.5, 7.05),
                xlim=(0, min(1.05, upper_ap * 1.15)), xlabel='Average precision (AP)')
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].set_title('(a) Pooled held-out performance', loc='left', pad=20)
    axes[1].axvline(0, color='#646464', ls='--', lw=.8)
    axes[1].set(yticks=[1, 0], yticklabels=['Logistic\nregression', 'Random\nforest'],
                ylim=(-.55, 1.65), xlim=(lower - padding, upper + padding),
                xlabel='AP difference\n(combined minus wearable only)')
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].set_title('(b) Added value of glucose', loc='left', pad=20)
    fig.text(.18, .055, f'Dashed lines: positive-day proportion ({prevalence:.2%}) in (a); zero difference in (b).',
             fontsize=8)
    return fig


def calibration_panel(ax, counts, bins, family):
    #### Plot occupied bins and exact count rows so sparse high-probability bins remain clear.
    clean_axes(ax)
    clean_axes(counts)
    ax.plot([0, 1], [0, 1], ls='--', color='#888888', lw=.8)
    for index, name in enumerate(INPUTS):
        data = bins.loc[bins.input_set.eq(name)].sort_values('bin')
        occupied = data.days.gt(0)
        ax.scatter(data.loc[occupied, 'mean_probability'], data.loc[occupied, 'observed_frequency'],
                   s=25, marker=MARKERS[name], color=COLOURS[name],
                   edgecolors='white', linewidths=.4, zorder=3)
        centres = (data.lower + data.upper) / 2
        for centre, days in zip(centres, data.days):
            counts.text(centre, index, str(int(days)), ha='center', va='center',
                        color=COLOURS[name], fontsize=7.5)
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025))
    ax.set_title(FAMILY_LABELS[family], loc='left')
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    counts.set(xlim=(-.025, 1.025), ylim=(2.6, -.6), yticks=[0, 1, 2],
               yticklabels=['Glucose', 'Wearable', 'Combined'],
               xlabel='Predicted probability (10% bins)')
    counts.tick_params(axis='y', length=0, labelsize=7)
    counts.spines[['left', 'bottom']].set_visible(False)
    for edge in np.linspace(0, 1, 11):
        counts.axvline(edge, color='#E6E6E6', linewidth=.5)
    counts.set_title('Days per fixed bin', loc='left', fontsize=8, pad=3)
    counts.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    counts.set_xticks(np.linspace(0, 1, 6))


def calibration_figure(tables):
    #### Keep the full probability range and all ten bins visible without smoothing.
    fig = plt.figure(figsize=(6.5, 4.7))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], left=.12, right=.98,
                           bottom=.20, top=.92, hspace=.23, wspace=.36)
    for column, family in enumerate(FAMILIES):
        ax, counts = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
        calibration_panel(ax, counts, tables[family]['calibration_bins'], family)
        ax.tick_params(labelbottom=False)
        if column == 0:
            ax.set_ylabel('Observed positive-day proportion')
    handles = [Line2D([], [], marker=MARKERS[name], linestyle='', color=COLOURS[name],
                      markersize=4, label=LABELS[name]) for name in INPUTS]
    handles.append(Line2D([], [], ls='--', color='#888888', lw=.8, label='Perfect agreement'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045),
               ncol=4, frameon=False, fontsize=8, handletextpad=.4, columnspacing=1)
    return fig


def check_destinations(destination, latex_figures):
    #### Reject existing outputs and any destination outside the approved generated folders.
    destination = destination.resolve()
    if (WORKSPACE / 'results').resolve() not in destination.parents:
        raise ValueError('Figure outputs must be under results/')
    if destination.exists():
        raise FileExistsError(f'Choose a fresh figure directory: {destination}')
    if latex_figures is not None:
        if latex_figures.resolve() != (WORKSPACE / 'latex/figures').resolve():
            raise ValueError('LaTeX exports must go to latex/figures')
        for stem in STEMS:
            if (latex_figures / f'{stem}.pdf').exists():
                raise FileExistsError(latex_figures / f'{stem}.pdf')


def export_figure(fig, destination, stem):
    #### Export embedded vector text and a matching high-resolution preview.
    for suffix in ('pdf', 'png'):
        path = destination / f'{stem}.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.05, metadata=metadata)
    plt.close(fig)


def run(logistic, forest, destination, latex_figures=None):
    #### Validate saved results, create the figures and record their complete provenance.
    check_destinations(destination, latex_figures)
    tables, config, inputs = load_results(logistic, forest)
    destination.mkdir(parents=True)
    with plt.rc_context(STYLE):
        figures = (prediction_timeline(), evaluation_diagram(),
                   performance_figure(tables), calibration_figure(tables))
        for fig, stem in zip(figures, STEMS):
            export_figure(fig, destination, stem)
    outputs = sorted(destination.iterdir())
    if latex_figures is not None:
        for stem in STEMS:
            target = latex_figures / f'{stem}.pdf'
            shutil.copyfile(destination / target.name, target)
            outputs.append(target)
    sources = [Path(__file__), Path(__file__).with_name('figure_data.py'),
               Path(__file__).with_name('figure_diagrams.py'),
               Path(__file__).with_name('metrics.py'), Path(__file__).with_name('protocol.py'),
               Path(__file__).parent / 'tests/test_figures.py']
    manifest = {'protocol': config['version'], 'description': 'Saved predictions and intervals; no model refitting',
                'inputs': {str(path.resolve()): digest(path) for path in inputs},
                'sources': {str(path.resolve()): digest(path) for path in sources},
                'outputs': {str(path.resolve()): digest(path) for path in outputs},
                'versions': {'python': platform.python_version(),
                             **{name: version(name) for name in ('matplotlib', 'numpy', 'pandas', 'scikit-learn')}},
                'style': STYLE, 'png_dpi': 300, 'bootstrap_scope': config['bootstrap_scope']}
    (destination / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported four verified figures to {destination}', flush=True)


def main():
    #### Accept explicit historical inputs and fresh output destinations.
    parser = argparse.ArgumentParser(description='Create scientific three-day forecasting thesis figures')
    parser.add_argument('--logistic-dir', type=Path,
                        default=WORKSPACE / 'results/onset_logistic_2022_v1/evaluation')
    parser.add_argument('--forest-dir', type=Path,
                        default=WORKSPACE / 'results/onset_forest_2022_v1/evaluation')
    parser.add_argument('--results-dir', type=Path,
                        default=WORKSPACE / 'results/onset_thesis_figures_2022_v2')
    parser.add_argument('--latex-figures-dir', type=Path)
    args = parser.parse_args()
    run(args.logistic_dir.resolve(), args.forest_dir.resolve(), args.results_dir.resolve(),
        args.latex_figures_dir)


if __name__ == '__main__':
    main()
