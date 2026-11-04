#### Draw phase contrasts and successive-cycle glucose changes with their observed coverage.
#### Export PNG and PDF figures using a non-interactive plotting backend.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from onset_assessment.cycle_analysis import SIGNALS


def save(fig, results, name):
    #### Export the same labelled figure in screen and vector formats.
    fig.savefig(results / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(results / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_effects(people, summary, results):
    #### Show every participant's cycle-averaged phase effect and the group mean interval.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    phases = ['menstrual', 'fertility', 'luteal']
    for ax, (measurement, label) in zip(axes, SIGNALS.items()):
        for index, phase in enumerate(phases):
            values = people.loc[people.measurement.eq(measurement) & people.phase.eq(phase)]
            row = summary.loc[summary.measurement.eq(measurement) & summary.phase.eq(phase)].iloc[0]
            offsets = np.linspace(-.13, .13, len(values))
            ax.scatter(index + offsets, values.difference, color='#526d82', alpha=.65, s=19)
            ax.errorbar(index + .22, row.difference,
                        yerr=[[row.difference - row.difference_ci_lower],
                              [row.difference_ci_upper - row.difference]],
                        fmt='o', color='#d46b28', capsize=4)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(range(3), ['Menstrual', 'Fertile-window\nlabel', 'Luteal'])
        ax.set_title(label)
        ax.set_ylabel('Difference from the same cycle’s follicular phase')
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Within-participant phase differences\nDots: participants; orange: equal-participant mean and 95% interval')
    fig.tight_layout()
    save(fig, results, 'phase_differences')


def plot_repeatability(pairs, results):
    #### Compare earlier and later luteal-minus-follicular contrasts on identical axes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (measurement, label) in zip(axes, SIGNALS.items()):
        data = pairs.loc[pairs.measurement.eq(measurement) & pairs.phase.eq('luteal')]
        ax.scatter(data.earlier, data.later, color='#526d82')
        bound = max(.1, np.abs(data[['earlier', 'later']].to_numpy()).max()) * 1.18
        ax.plot([-bound, bound], [-bound, bound], '--', color='grey', linewidth=1)
        ax.axhline(0, color='black', linewidth=.7)
        ax.axvline(0, color='black', linewidth=.7)
        for row in data.itertuples():
            ax.annotate(str(row.id), (row.earlier, row.later), xytext=(3, 3),
                        textcoords='offset points', fontsize=7)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
               xlabel='Earlier cycle: luteal − follicular',
               ylabel='Next cycle: luteal − follicular', title=f'{label}\n{len(data)} adjacent pairs')
    fig.suptitle('Does the luteal glucose difference repeat?\nSame-sign quadrants indicate direction agreement; dashed line indicates identical effects')
    fig.tight_layout()
    save(fig, results, 'successive_cycle_agreement')
