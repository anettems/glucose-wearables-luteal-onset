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
    diagram_axis(ax, '(a) Prediction timing')
    boundaries = [.025, .215, .405, .595, .785, .975]
    labels = ['$t-1$', '$t$', '$t+1$', '$t+2$', '$t+3$']
    for left, right, label in zip(boundaries[:-1], boundaries[1:], labels):
        ax.text((left + right) / 2, .92, label, ha='center', va='center', fontsize=10)
    for boundary in boundaries:
        ax.plot([boundary, boundary], [.25, .80], color='#D6D6D6', lw=.6, zorder=1)

    prior_day, prediction_day, first_future_day = boundaries[:3]
    box(ax, prior_day, .55, prediction_day - prior_day, .23, 'Daily glucose')
    box(ax, .165, .25, .135, .23, 'Main sleep')
    box(ax, first_future_day, .55, boundaries[-1] - first_future_day, .23,
        'Outcome window', GREEN, '#F0F6F3')

    #### Place the cutoff at 14 hours into the prediction day.
    cutoff = prediction_day + (first_future_day - prediction_day) * 14 / 24
    ax.plot([cutoff, cutoff], [.21, .81], color=ORANGE, lw=1.3)
    ax.text(cutoff, .12, '14:00 prediction', color=ORANGE, ha='center', va='center',
            fontsize=9)


def outcome_panel(ax):
    #### Illustrate positive, negative and unknown outcomes using future start statuses.
    diagram_axis(ax, '(c) Example start statuses')
    centres = [.135, .365, .595]
    for x, label in zip(centres, ['$t+1$', '$t+2$', '$t+3$']):
        ax.text(x, .92, label, ha='center', va='center', fontsize=10)
    ax.text(.84, .92, 'Outcome', ha='center', va='center', fontsize=9)

    #### Include a positive example with an unknown day to show that a known start suffices.
    examples = [('Positive', ['No start', 'Start', 'Unknown'], GREEN),
                ('Negative', ['No start'] * 3, BLUE),
                ('Unknown', ['No start', 'Unknown', 'No start'], GREY)]
    for y, (outcome, statuses, colour) in zip([.62, .35, .08], examples):
        for x, status in zip(centres, statuses):
            box(ax, x - .11, y, .22, .20, status, colour)
        ax.text(.84, y + .10, outcome, ha='center', va='center', fontsize=9, color=colour)


def prediction_timeline():
    #### Stack measurement timing and future outcomes with the original panel letters.
    fig, (measurements, outcomes) = plt.subplots(2, 1, figsize=(6.5, 3.2))
    fig.subplots_adjust(left=.025, right=.975, bottom=.03, top=.90, hspace=.25)
    measurement_panel(measurements)
    outcome_panel(outcomes)
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
