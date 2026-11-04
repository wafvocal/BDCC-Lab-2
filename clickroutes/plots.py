"""Every figure in the report, drawn with one fixed, validated palette.

Colour follows the entity: each route and each cohort keeps the same
colour in every figure. Route colours are the first five slots of a
colour-blind-checked categorical palette; cohorts are ordered, so they use
light-to-dark steps of one blue. Lines break at missing months instead of
joining across them, and the two reference events are drawn as thin
annotations, not as causes.
"""
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from . import config
from .preparation import month_range

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
WARNING = '#fab219'
NEGATIVE = '#e34948'
POSITIVE = '#2a78d6'
ROUTE_COLORS = {'search': '#2a78d6', 'link': '#eb6834',
                'unattributed': '#1baf7a', 'wikimedia': '#eda100',
                'external': '#e87ba4'}
COHORT_COLORS = dict(zip(config.COHORT_ORDER,
                         ('#0d366b', '#1c5cab', '#3987e5', '#86b6ef')))


def apply_style():
    """Set the shared Matplotlib style (thin marks, recessive grid).

    Keys that an older Matplotlib does not know are skipped, so the style
    works on any recent version installed on Jojie.
    """
    style = {
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8, 'axes.labelcolor': INK_2,
        'axes.titlecolor': INK, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID,
        'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'font.size': 10,
        'legend.frameon': False, 'legend.fontsize': 9,
        'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round', 'figure.dpi': 110,
        'savefig.dpi': 160, 'savefig.bbox': 'tight',
    }
    for key, value in style.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value


def save(fig, name):
    """Save a figure as ``figures/<name>.png`` and return its path."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / f'{name}.png'
    fig.savefig(path)
    return path


# -------------------------------------------------------------- helpers ----
def _calendar():
    """All months of the window and their dates (middle of the month)."""
    months = month_range(config.FIRST_MONTH, config.LAST_MONTH)
    dates = pd.to_datetime([f'{m}-15' for m in months])
    return months, dates


def _wide(table, value, by=None):
    """Pivot a monthly table to the full calendar (missing months NaN)."""
    months, dates = _calendar()
    if by is None:
        wide = table.set_index('month')[[value]]
    else:
        wide = table.pivot_table(index='month', columns=by, values=value,
                                 aggfunc='sum')
    wide = wide.reindex(months)
    wide.index = dates
    return wide


def _time_axis(ax, label_years=True):
    """Yearly ticks over the whole window."""
    months, dates = _calendar()
    ax.set_xlim(dates[0] - pd.Timedelta(days=25),
                dates[-1] + pd.Timedelta(days=25))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if not label_years:
        ax.tick_params(labelbottom=False)


def _events(ax, top=True):
    """Thin reference lines for COVID-19 and ChatGPT, labelled once."""
    for month, label in config.EVENTS.items():
        when = pd.Timestamp(f'{month}-15')
        ax.axvline(when, color=MUTED if top else AXIS, linewidth=0.8,
                   zorder=0)
        if top:
            ax.annotate(label, (when, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=INK_2)


def _end_label(ax, series, text, color_dot):
    """Direct label at the last non-missing point of a line."""
    clean = series.dropna()
    if clean.empty:
        return
    ax.plot(clean.index[-1], clean.iloc[-1], 'o', ms=5, color=color_dot,
            mec=SURFACE, mew=1.2, zorder=4)
    ax.annotate(text, (clean.index[-1], clean.iloc[-1]),
                xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=8.5, color=INK)


# ------------------------------------------------------------- figure 1 ----
def _box(ax, x, y, w, h, title, body, face, edge, dark=False):
    """Draw one rounded box with a bold title and wrapped body text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.25,rounding_size=1.0',
                                fc=face, ec=edge, lw=0.9))
    title_color = '#ffffff' if dark else INK
    body_color = '#ffffff' if dark else INK_2
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top',
            fontsize=9.5, weight='bold', color=title_color)
    ax.text(x + w / 2, y + h - 5.4, textwrap.fill(body, int(w * 1.9)),
            ha='center', va='top', fontsize=7.8, color=body_color,
            linespacing=1.35)


def _arrow(ax, start, end):
    """Plain arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': MUTED, 'lw': 1.1,
                            'shrinkA': 0, 'shrinkB': 0})


def pipeline_diagram():
    """Figure 1: the pipeline from raw files to the answer.

    Returns
    -------
    matplotlib.figure.Figure
        The diagram.
    """
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis('off')
    prep = [
        ('77 monthly files', 'English clickstream, Nov 2017 to Mar 2026; '
         'tab-separated .gz, about 30 GB and 2.48 billion rows'),
        ('Read', 'Explicit 4-column schema, no header, quoting off, '
         'corrupt-record column; month from the file name'),
        ('Stage and profile', 'A batch of months as Parquet: nulls, '
         'malformed lines, labels by type, duplicates, size of n'),
        ('Map and aggregate', 'One CASE maps prev to 5 routes; month x '
         'article x route sums for all pairs and n >= 50, 100'),
        ('Filter and scale', 'Drop malformed rows, set Main_Page aside, '
         'divide by days in the month'),
    ]
    width, gap, y_prep = 17.0, 3.0, 43.0
    for i, (title, body) in enumerate(prep):
        x = 1 + i * (width + gap)
        _box(ax, x, y_prep, width, 14.5, title, body, '#f0efec', AXIS)
        if i:
            _arrow(ax, (x - gap + 0.2, y_prep + 7.2), (x - 0.2,
                                                       y_prep + 7.2))
    analyses = [
        ('Q1 Route composition', 'SUM by month and route, window shares; '
         'search over all and over attributed views'),
        ('Q2 Popularity cohorts', 'Top 10,000 by ORDER BY ... LIMIT, '
         'broadcast join; ranks 1-100, 101-1,000, 1,001-10,000, tail'),
        ('Q3 Where the change comes from', '2018 vs 2025 matched months, '
         'full outer join on title; within, between, entry, exit'),
        ('Q4 Checks', 'Main_Page in, unattributed out, 2018 ranking, '
         'pairs n >= 50 / 100; totals reconcile'),
    ]
    width_q, gap_q, y_q = 22.0, 2.67, 19.5
    bus = y_q + 16.5
    ax.plot([1 + width_q / 2, 1 + 3 * (width_q + gap_q) + width_q / 2],
            [bus, bus], color=MUTED, lw=1.1)
    last_x = 1 + 4 * (width + gap) + width / 2
    ax.plot([last_x, last_x], [y_prep - 0.4, bus], color=MUTED, lw=1.1)
    ax.plot([1 + 3 * (width_q + gap_q) + width_q / 2, last_x],
            [bus, bus], color=MUTED, lw=1.1)
    for i, (title, body) in enumerate(analyses):
        x = 1 + i * (width_q + gap_q)
        _arrow(ax, (x + width_q / 2, bus), (x + width_q / 2, y_q + 13.4))
        _box(ax, x, y_q, width_q, 13.0, title, body, '#e6effb', '#9ec5f4')
        _arrow(ax, (x + width_q / 2, y_q - 0.4), (x + width_q / 2, 10.6))
    _box(ax, 1, 2.0, 98, 8.2, 'Answer', 'How the mix of arrival routes '
         'changed from 2017 to 2026, described without assigning causes',
         '#1c5cab', '#1c5cab', dark=True)
    ax.text(1, 60.5, 'Preparation (once, Spark)', fontsize=9, color=INK_2,
            weight='bold')
    ax.text(1, bus + 1.2, 'Analysis (Spark SQL on the aggregated table)',
            fontsize=9, color=INK_2, weight='bold')
    return fig


# ------------------------------------------------------------- figure 2 ----
def coverage_grid(coverage):
    """Figure 2: which months of the window have a file.

    Parameters
    ----------
    coverage : pandas.DataFrame
        Output of :func:`clickroutes.preparation.coverage_table`.

    Returns
    -------
    matplotlib.figure.Figure
        Year by month grid.
    """
    years = sorted(coverage['year'].unique())
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(years) + 1.2))
    for row in coverage.itertuples(index=False):
        y = years.index(row.year)
        ax.add_patch(Rectangle((row.month_of_year - 0.46, y - 0.42), 0.92,
                               0.84, fc=POSITIVE if row.present else GRID,
                               ec=SURFACE, lw=1.5))
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(len(years) - 0.5, -0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                        'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.grid(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    present = int(coverage['present'].sum())
    ax.set_title(f'{present} of {len(coverage)} months in the window have '
                 'an English file')
    ax.legend(handles=[Patch(fc=POSITIVE, label='File present'),
                       Patch(fc=GRID, label='Missing from the archive')],
              loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return fig


# ------------------------------------------------------------- figure 3 ----
def volume_panels(monthly, flags):
    """Figure 3: rows and requests per day by month, with flagged jumps.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`clickroutes.explore.monthly_profile`.
    flags : pandas.DataFrame
        Output of :func:`clickroutes.checks.release_flags` (flags).

    Returns
    -------
    matplotlib.figure.Figure
        Two stacked panels sharing the time axis.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    panels = [('rows', 1e6, '(prev, curr) pairs per month (millions)'),
              ('views_per_day', 1e6,
               'Requests per day, all usable rows (millions)')]
    flagged = set(flags['month']) if len(flags) else set()
    for ax, (column, scale, title) in zip(axes, panels):
        series = _wide(monthly, column)[column] / scale
        ax.plot(series.index, series.values, color=POSITIVE,
                marker='o', ms=3.5, mec=SURFACE, mew=0.8)
        marks = series[[m.strftime('%Y-%m') in flagged
                        for m in series.index]]
        ax.plot(marks.index, marks.values, linestyle='none', marker='^',
                ms=8, color=WARNING, mec=INK_2, mew=0.6, zorder=5)
        ax.set_title(title, pad=18 if ax is axes[0] else 6)
        ax.set_ylim(bottom=0)
        _time_axis(ax)
        _events(ax, top=ax is axes[0])
    axes[0].legend(handles=[
        Line2D([], [], color=POSITIVE, marker='o', ms=3.5, label='Month'),
        Line2D([], [], color=WARNING, marker='^', ms=8, mec=INK_2,
               linestyle='none', label='Possible release change')],
        loc='lower left', ncol=2)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- figure 4 ----
def n_distribution(cdf):
    """Figure 4: cumulative share of rows and requests by pair size.

    Parameters
    ----------
    cdf : pandas.DataFrame
        Output of :func:`clickroutes.explore.n_cdf`.

    Returns
    -------
    matplotlib.figure.Figure
        Two cumulative lines on a log axis with the 50 and 100 floors.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    upper = 2.0 ** (cdf['log2_bin'] + 1)
    for column, color, text in (('cum_rows_%', POSITIVE, 'Rows (pairs)'),
                                ('cum_requests_%', ROUTE_COLORS['link'],
                                 'Requests')):
        ax.plot(upper, cdf[column], color=color, marker='o', ms=4,
                mec=SURFACE, mew=0.8, label=text)
    for i, minimum in enumerate(config.MIN_PAIR_COUNTS):
        ax.axvline(minimum, color=MUTED, linewidth=0.8, zorder=0)
        ax.annotate(f'n = {minimum}', (minimum, 1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(-3 if i == 0 else 3, 3),
                    textcoords='offset points', fontsize=8, color=INK_2,
                    ha='right' if i == 0 else 'left', va='bottom')
    ax.legend(loc='lower right')
    ax.set_xscale('log')
    ax.set_ylim(0, 102)
    ax.set_xlabel('n, requests per pair in the month (log scale)')
    ax.set_ylabel('Share in pairs smaller than n (%)')
    ax.set_title('Rows and requests in pairs smaller than n, all months',
                 pad=18)
    ax.grid(True, axis='x', color=GRID, linewidth=0.6)
    return fig


# ------------------------------------------------------------- figure 5 ----
def share_bars(shares, title):
    """Figure 5: route shares by month as stacked columns.

    Parameters
    ----------
    shares : pandas.DataFrame
        Output of :func:`clickroutes.routes.route_shares`.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
        One column per month; missing months stay empty.
    """
    wide = _wide(shares, 'share', by='route') * 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bottom = np.zeros(len(wide))
    for route in config.ROUTES:
        values = wide[route].fillna(0).to_numpy()
        ax.bar(wide.index, values, width=24, bottom=bottom,
               color=ROUTE_COLORS[route], edgecolor=SURFACE, linewidth=0.6,
               label=config.ROUTE_NAMES[route])
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of arrivals (%)')
    ax.set_title(title, pad=18)
    _time_axis(ax)
    _events(ax)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.09))
    return fig


# ------------------------------------------------------------- figure 6 ----
def volume_by_route(shares):
    """Figure 6: requests per day for each route, one panel per route.

    Parameters
    ----------
    shares : pandas.DataFrame
        Output of :func:`clickroutes.routes.route_shares`.

    Returns
    -------
    matplotlib.figure.Figure
        Small multiples with independent y-axes (scales differ by 100x),
        plus a sixth panel with all routes together.
    """
    wide = _wide(shares, 'views_per_day', by='route') / 1e6
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), sharex=True)
    panels = list(config.ROUTES) + ['all']
    for ax, route in zip(axes.flat, panels):
        if route == 'all':
            series = wide[list(config.ROUTES)].sum(axis=1, min_count=1)
            color, name = INK_2, 'All routes'
        else:
            series = wide[route]
            color, name = ROUTE_COLORS[route], config.ROUTE_NAMES[route]
        ax.plot(series.index, series.values, color=color, marker='o',
                ms=2.5)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(bottom=0)
        _time_axis(ax)
        _events(ax, top=False)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.text(0.0, 0.5, 'Requests per day (millions)', rotation=90,
             va='center', ha='left', color=INK_2, fontsize=10)
    return fig


# ------------------------------------------------------------- figure 7 ----
def search_share_lines(all_views, attributed, unattributed):
    """Figure 7: search share with and without unattributed traffic.

    Parameters
    ----------
    all_views, attributed : pandas.DataFrame
        Search rows of :func:`clickroutes.routes.route_shares` for the
        baseline and the attributed-only variant.
    unattributed : pandas.DataFrame
        Unattributed rows of the baseline.

    Returns
    -------
    matplotlib.figure.Figure
        Three lines in percent on one axis.
    """
    fig, ax = plt.subplots(figsize=(11, 4.4))
    lines = [(attributed, ROUTE_COLORS['link'],
              'Search, share of attributed views'),
             (all_views, ROUTE_COLORS['search'],
              'Search, share of all views'),
             (unattributed, ROUTE_COLORS['unattributed'],
              'Unattributed, share of all views')]
    for table, color, text in lines:
        series = _wide(table, 'share')['share'] * 100
        ax.plot(series.index, series.values, color=color, label=text,
                marker='o', ms=2.5)
        _end_label(ax, series, f'{series.dropna().iloc[-1]:.1f}%', color)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share (%)')
    ax.set_title('Search share depends on whether unattributed arrivals '
                 'are counted', pad=18)
    _time_axis(ax)
    _events(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
    return fig


# ------------------------------------------------------------- figure 8 ----
def cohort_panels(cohort_shares, concentration):
    """Figure 8: route shares within each popularity cohort.

    Parameters
    ----------
    cohort_shares : pandas.DataFrame
        :func:`clickroutes.routes.route_shares` grouped by ``cohort``.
    concentration : pandas.DataFrame
        Output of :func:`clickroutes.cohorts.concentration`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per route plus one for each cohort's share of all
        arrivals; one line per cohort.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, route in zip(axes.flat, config.ROUTES):
        part = cohort_shares[cohort_shares['route'] == route]
        wide = _wide(part, 'share', by='cohort') * 100
        for cohort in config.COHORT_ORDER:
            if cohort in wide:
                ax.plot(wide.index, wide[cohort], color=COHORT_COLORS[cohort],
                        lw=1.8, marker='o', ms=2)
        ax.set_title(f'{config.ROUTE_NAMES[route]} share (%)', fontsize=10)
        ax.set_ylim(bottom=0)
        _time_axis(ax)
        _events(ax, top=False)
    ax = axes.flat[5]
    wide = _wide(concentration, 'share_%', by='cohort')
    for cohort in config.COHORT_ORDER:
        if cohort in wide:
            ax.plot(wide.index, wide[cohort], color=COHORT_COLORS[cohort],
                    lw=1.8, marker='o', ms=2)
    ax.set_title('Cohort share of all arrivals (%)', fontsize=10)
    ax.set_ylim(bottom=0)
    _time_axis(ax)
    handles = [Line2D([], [], color=COHORT_COLORS[c], lw=2,
                      label=f'Ranks {c}' if c[0].isdigit() and '+' not in c
                      else c) for c in config.COHORT_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- figure 9 ----
def decomposition_bars(summary):
    """Figure 9: the change in search share split into its sources.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row of :func:`clickroutes.decomposition.summarize`.

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal bars in percentage points, coloured by sign.
    """
    row = summary.iloc[0]
    items = [('Total change in search share', row['change_pp'], True),
             ('Same articles, reached differently (within)',
              row['within_pp'], False),
             ('Readers shifted to other articles (between)',
              row['between_pp'], False),
             ('Articles that appeared (entry)', row['entry_pp'], False),
             ('Articles that disappeared (exit)', row['exit_pp'], False)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = [i[0] for i in items][::-1]
    values = np.array([i[1] for i in items][::-1], dtype=float)
    totals = [i[2] for i in items][::-1]
    colors = [INK_2 if t else (NEGATIVE if v < 0 else POSITIVE)
              for v, t in zip(values, totals)]
    ax.barh(labels, values, height=0.55, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    span = max(np.abs(values).max(), 1e-9)
    for y, value in enumerate(values):
        ax.annotate(f'{value:+.2f} pp', (value, y),
                    xytext=(5 if value >= 0 else -5, 0),
                    textcoords='offset points', va='center',
                    ha='left' if value >= 0 else 'right', fontsize=9,
                    color=INK)
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.set_xlabel('Percentage points')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ax.set_title(f'Search share {row["search_share_early_%"]:.1f}% '
                 f'({config.EARLY_YEAR}) to '
                 f'{row["search_share_late_%"]:.1f}% ({config.LATE_YEAR}), '
                 'matched months')
    return fig


# ------------------------------------------------------------ figure 10 ----
def robustness_dots(table):
    """Figure 10: each finding under every handling choice.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`clickroutes.checks.robustness`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per headline finding, one dot per variant.
    """
    findings = [f'{config.ROUTE_NAMES["search"]} share, all articles',
                f'Search share, top {config.COHORT_ORDER[0]}',
                'Search share, long tail',
                'Decomposition: within (same articles)']
    findings = [f for f in findings if f in set(table['finding'])]
    variants = list(dict.fromkeys(table['variant']))
    position = {name: len(variants) - 1 - i for i, name in
                enumerate(variants)}
    fig, axes = plt.subplots(1, len(findings),
                             figsize=(3.2 * len(findings), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, finding in zip(axes, findings):
        part = table[table['finding'] == finding].dropna(subset=['value_pp'])
        y = [position[v] for v in part['variant']]
        colors = [INK if v == 'baseline' else POSITIVE
                  for v in part['variant']]
        ax.scatter(part['value_pp'], y, s=46, c=colors, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(list(position.values()))
        ax.set_yticklabels(list(position))
        ax.set_ylim(-0.6, len(variants) - 0.4)
        ax.grid(True, axis='x')
        ax.grid(False, axis='y')
        ax.set_title(textwrap.fill(finding, 26), fontsize=9.5)
        ax.set_xlabel('Change (pp)')
        low, high = part['value_pp'].min(), part['value_pp'].max()
        pad = max(abs(low), abs(high)) * 0.25 + 1e-6
        ax.set_xlim(min(low, 0) - pad, max(high, 0) + pad)
    fig.tight_layout()
    return fig
