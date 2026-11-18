#### Draw scientific schematics of the agreed prediction timing and participant evaluation.
#### Positions illustrate the frozen calendar windows and model-development sequence.
#### Sleep timing is schematic, and the outcome examples are illustrative, not participant data.
#### Return Matplotlib figures for vector export by thesis_figures.py; no analysis is performed.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BLUE = '#356783'
ORANGE = '#B75625'
GREEN = '#397C69'
GREY = '#646464'


def box(ax, x, y, width, height, text, colour=BLUE, fill='#F4F7F9', size=9):
    #### Place a labelled rectangle using axes coordinates for reproducible alignment.
    ax.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=colour,
                           linewidth=.8))
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=size)


def arrow(ax, start, end):
    #### Connect diagram stages with a small, unobtrusive arrow.
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': GREY, 'lw': .8})


def diagram_axis(ax, title):
    #### Remove chart furniture and add a consistent panel heading.
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=10, pad=7)


def measurement_panel(ax):
    #### Separate completed measurements, the prediction cutoff and future outcome days.
    diagram_axis(ax, '(a) Measurements available at prediction')
    boundaries = [.03, .24, .45, .62, .79, .96]
    labels = ['$t-1$', '$t$', '$t+1$', '$t+2$', '$t+3$']
    for left, right, label in zip(boundaries[:-1], boundaries[1:], labels):
        ax.text((left + right) / 2, .9, label, ha='center', fontsize=10)
        ax.plot([left, left], [.16, .81], color='#D6D6D6', lw=.6)
    ax.plot([.96, .96], [.16, .81], color='#D6D6D6', lw=.6)
    box(ax, .03, .55, .21, .22, 'Completed glucose\nday $t-1$', size=8)
    box(ax, .18, .25, .16, .21, 'Main sleep\nending on $t$', size=8)
    cutoff = .24 + .21 * 14 / 24
    ax.plot([cutoff, cutoff], [.12, .82], color=ORANGE, lw=1.3)
    ax.text(cutoff + .012, .47, '14:00\nprediction', color=ORANGE, va='center', fontsize=8)
    box(ax, .45, .55, .51, .22, 'Reference start on any of the next three days',
        GREEN, '#F0F6F3', size=8)
    ax.text(.7, .29, 'Future labels define the outcome;\nfuture sensors are not model inputs.',
            ha='center', va='center', fontsize=8)
    ax.text(.03, -.015, 'Sleep timing is schematic; the retained sleep must end by 14:00.',
            fontsize=8, color=GREY)


def history_panel(ax):
    #### Show past calendar windows with open right edges that exclude the current value.
    diagram_axis(ax, '(b) Personal history on the aligned prediction-day axis')
    ax.set_xlim(-17, 1.8)
    for y, start, colour, label in [(.68, -14, BLUE, 'Median reference: at least 7 observed days'),
                                    (.35, -7, GREEN, 'Trend: at least 4 observed days')]:
        ax.plot([start, -.15], [y, y], color=colour, lw=5, solid_capstyle='butt')
        ax.plot([start], [y], '|', color=colour, ms=12, mew=1.2)
        ax.plot([0], [y], 'o', color=colour, mfc='white', ms=5, zorder=3)
        ax.text(start, y + .13, label, fontsize=8, va='bottom')
    for x, label in [(-14, '$t-14$'), (-7, '$t-7$'), (0, '$t$')]:
        ax.text(x, .08, label, ha='center', fontsize=9)
    ax.plot([0, 0], [.2, .92], color=ORANGE, lw=.8, ls='--')
    ax.text(.55, .52, 'Current\naligned\nvalue', color=ORANGE, fontsize=8, va='center')
    ax.text(-16.8, -.15, 'Histories exclude $t$ and future values. For glucose, $x(s)$ summarises day $s-1$.',
            fontsize=8)


def outcome_panel(ax):
    #### Illustrate positive, negative and unknown outcomes using future start statuses.
    diagram_axis(ax, '(c) Outcome examples from the full phase-label timeline')
    centres = [.48, .60, .72]
    for x, label in zip(centres, ['$t+1$', '$t+2$', '$t+3$']):
        ax.text(x, .9, label, ha='center', fontsize=9)
    examples = [('Positive', ['No start', 'Start', '?'], '1', GREEN),
                ('Negative', ['No start'] * 3, '0', BLUE),
                ('Unknown', ['No start', '?', 'No start'], 'Excluded', GREY)]
    for y, (label, statuses, result, colour) in zip([.64, .37, .10], examples):
        ax.text(.03, y + .075, label, fontsize=9, va='center')
        for x, status in zip(centres, statuses):
            box(ax, x - .055, y, .11, .15, status, colour, size=7.5)
        ax.text(.84, y + .075, result, fontsize=9, va='center', color=colour)
    ax.text(.03, -.16, '? = unknown start status. A start on $t$ does not count towards the outcome.',
            fontsize=8)


def prediction_timeline():
    #### Combine measurement timing, past histories and example future outcomes.
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.1))
    fig.subplots_adjust(left=.025, right=.98, bottom=.07, top=.95, hspace=.5)
    for draw, ax in zip((measurement_panel, history_panel, outcome_panel), axes):
        draw(ax)
    return fig


def test_fold_panel(ax):
    #### Show the five test rounds without implying equal-sized participant groups.
    diagram_axis(ax, '(a) Five participant-disjoint test rounds')
    for column in range(5):
        ax.text(.24 + column * .155, .94, f'Fold {column + 1}', ha='center', fontsize=8)
    for row in range(5):
        y = .72 - row * .145
        ax.text(.015, y + .05, f'Round {row + 1}', va='center', fontsize=8)
        for column in range(5):
            test = row == column
            box(ax, .17 + column * .155, y, .14, .105, 'Test' if test else 'Develop',
                ORANGE if test else BLUE, '#FAF0E9' if test else '#F0F5F8', 8)
    ax.text(.015, -.045, 'All days from a participant stay together. Cell widths do not represent group sizes.',
            fontsize=8)


def validation_panel(ax):
    #### Expand one development set into three validation rounds and a final refit.
    diagram_axis(ax, '(b) Model selection within each development set')
    for column in range(3):
        ax.text(.22 + column * .22, .95, f'Validation fold {column + 1}',
                ha='center', fontsize=8)
    for row in range(3):
        y = .70 - row * .19
        ax.text(.015, y + .055, str(row + 1), va='center', fontsize=8)
        for column in range(3):
            validate = row == column
            box(ax, .12 + column * .22, y, .20, .12,
                'Validate' if validate else 'Train', GREEN if validate else BLUE,
                '#F0F6F3' if validate else '#F0F5F8', 8)
    box(ax, .80, .31, .18, .51, 'Select setting\nby mean\nvalidation AP', size=8)
    arrow(ax, (.77, .57), (.80, .57))
    ax.text(.015, .03, 'For every candidate fit: learn replacement values on training participants only;\n'
            'also learn scaling for logistic regression. Apply unchanged to validation participants.',
            fontsize=8, va='bottom')


def refit_panel(ax):
    #### Separate final development fitting, held-out testing and pooled evaluation.
    diagram_axis(ax, '(c) Refit, test and pool predictions')
    box(ax, .015, .39, .30, .40, 'Refit selected model and\npreprocessing on all\ndevelopment participants', size=8)
    box(ax, .36, .39, .26, .40, 'Predict the separate\ntest participants\nonce', ORANGE, '#FAF0E9', 8)
    box(ax, .67, .39, .31, .40, 'Pool five test folds:\none held-out probability\nper day and configuration', size=8)
    arrow(ax, (.315, .59), (.36, .59))
    arrow(ax, (.62, .59), (.67, .59))
    ax.text(.015, .10, 'Identical participants, days and folds for all six configurations:\n'
            '2 model families $\\times$ 3 input sets. Test outcomes do not choose model settings.', fontsize=8)


def evaluation_diagram():
    #### Summarise participant separation without displaying identifiers or new results.
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0),
                             gridspec_kw={'height_ratios': [1.15, 1.1, .85]})
    fig.subplots_adjust(left=.025, right=.98, bottom=.025, top=.95, hspace=.36)
    for draw, ax in zip((test_fold_panel, validation_panel, refit_panel), axes):
        draw(ax)
    return fig
