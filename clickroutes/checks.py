"""Release-artifact flags, robustness reruns and reconciliation checks.

A result is called robust only if its direction and rough size survive
every handling choice; "rough size" is made operational as a ratio to the
baseline between ``config.ROBUST_RATIO[0]`` and ``config.ROBUST_RATIO[1]``.
"""
import numpy as np
import pandas as pd

from . import config
from .decomposition import VARIANTS, decompose, summarize
from .routes import period_change, route_columns, route_shares


# ------------------------------------------------------ release artifacts ----
def modified_z(values):
    """Modified z-scores of Iglewicz and Hoaglin (1993).

    Parameters
    ----------
    values : array-like
        Numbers to score.

    Returns
    -------
    numpy.ndarray
        ``0.6745 (x - median) / MAD``; when the MAD is zero the mean
        absolute deviation is used instead, as the authors recommend.
    """
    x = np.asarray(values, dtype=float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad > 0:
        return 0.6745 * (x - median) / mad
    mean_ad = np.nanmean(np.abs(x - median))
    if mean_ad > 0:
        return (x - median) / (1.253314 * mean_ad)
    return np.zeros_like(x)


def release_flags(monthly, label_sets):
    """Flag month-to-month jumps that look like changes in the release.

    Only consecutive calendar months are compared, so a gap is never read
    as a jump.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`clickroutes.explore.monthly_profile`.
    label_sets : pandas.DataFrame
        Output of :func:`clickroutes.explore.label_sets`.

    Returns
    -------
    changes : pandas.DataFrame
        Every consecutive pair of months with the change in rows,
        requests per day and minimum n, and the modified z-scores.
    flags : pandas.DataFrame
        The pairs flagged, with the reason.
    """
    cols = ['month', 'rows', 'views_per_day', 'n_min']
    table = monthly[cols].copy()
    table['before'] = [str(pd.Period(m, freq='M') - 1)
                       for m in table['month']]
    previous = table[cols].rename(columns={
        'month': 'before', 'rows': 'rows_before',
        'views_per_day': 'views_per_day_before', 'n_min': 'n_min_before'})
    changes = table.merge(previous, on='before', how='inner')
    changes['rows_change_%'] = 100 * (changes['rows']
                                      / changes['rows_before'] - 1)
    changes['views_change_%'] = 100 * (changes['views_per_day']
                                       / changes['views_per_day_before']
                                       - 1)
    changes['rows_z'] = modified_z(np.log(changes['rows']
                                          / changes['rows_before']))
    changes['views_z'] = modified_z(np.log(changes['views_per_day']
                                           / changes['views_per_day_before']))
    sets = dict(zip(label_sets['month'], label_sets['labels']))
    changes['labels_changed'] = [sets.get(m) != sets.get(b) for m, b in
                                 zip(changes['month'], changes['before'])]
    changes['min_n_changed'] = changes['n_min'] != changes['n_min_before']
    reasons = []
    for _, row in changes.iterrows():
        why = []
        if abs(row['rows_z']) > config.ARTIFACT_Z:
            why.append(f'rows {row["rows_change_%"]:+.1f}%')
        if abs(row['views_z']) > config.ARTIFACT_Z:
            why.append(f'requests/day {row["views_change_%"]:+.1f}%')
        if row['min_n_changed']:
            why.append(f'min n {int(row["n_min_before"])} to '
                       f'{int(row["n_min"])}')
        if row['labels_changed']:
            why.append('label set changed')
        reasons.append('; '.join(why))
    changes['reason'] = reasons
    flags = changes[changes['reason'] != ''][
        ['before', 'month', 'reason']].reset_index(drop=True)
    return changes, flags


# ------------------------------------------------------------ robustness ----
SHARE_VARIANTS = {
    'baseline': {},
    'Main_Page included': {'include_main_page': True},
    'attributed views only': {'attributed_only': True},
    'pairs with n >= 50': {'suffix': '_ge50'},
    'pairs with n >= 100': {'suffix': '_ge100'},
}


def verdict(baseline, value):
    """Compare a rerun with the baseline.

    Parameters
    ----------
    baseline, value : float
        Changes in percentage points.

    Returns
    -------
    str
        ``robust``, ``direction holds, size differs``, ``direction
        reverses``, ``no change in baseline`` or ``not measurable`` (no
        pairs left under that variant).
    """
    if pd.isna(value) or pd.isna(baseline):
        return 'not measurable'
    if abs(baseline) < config.NO_CHANGE_PP:
        return 'no change in baseline'
    if np.sign(value) != np.sign(baseline):
        return 'direction reverses'
    low, high = config.ROBUST_RATIO
    if low <= value / baseline <= high:
        return 'robust'
    return 'direction holds, size differs'


def robustness(spark, early, late):
    """Rerun the main findings under every handling choice.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with ``monthly_cohort``, ``article_routes`` and
        ``rank_full`` registered.
    early, late : list of str
        Matched months.

    Returns
    -------
    pandas.DataFrame
        One row per finding and variant: the value in percentage points,
        the baseline value and the verdict.
    """
    top, rest = config.COHORT_ORDER[0], config.COHORT_ORDER[-1]
    records = []
    for name, options in SHARE_VARIANTS.items():
        overall = period_change(route_shares(spark, **options), early, late)
        for row in overall.itertuples(index=False):
            records.append((f'{config.ROUTE_NAMES[row.route]} share, all '
                            'articles', name, row.change_pp))
        cohort = period_change(
            route_shares(spark, group='cohort', **options), early, late,
            by=('cohort', 'route'))
        search = cohort[cohort['route'] == 'search'].set_index('cohort')
        records.append((f'Search share, top {top}', name,
                        search.loc[top, 'change_pp']))
        records.append(('Search share, long tail', name,
                        search.loc[rest, 'change_pp']))
    base_rank = period_change(route_shares(spark, group='cohort_base'),
                              early, late, by=('cohort_base', 'route'))
    search = base_rank[base_rank['route'] == 'search'].set_index(
        'cohort_base')
    label = f'{config.RANKING_BASE_YEAR} ranking'
    records.append((f'Search share, top {top}', label,
                    search.loc[top, 'change_pp']))
    records.append(('Search share, long tail', label,
                    search.loc[rest, 'change_pp']))
    for name in VARIANTS:
        summary = summarize(decompose(spark, early, late, name), name)
        records.append(('Decomposition: within (same articles)', name,
                        summary['within_pp'].iloc[0]))
        records.append(('Decomposition: between (other articles)', name,
                        summary['between_pp'].iloc[0]))
    table = pd.DataFrame(records, columns=['finding', 'variant', 'value_pp'])
    base = table[table['variant'] == 'baseline'].set_index('finding')[
        'value_pp']
    table['baseline_pp'] = table['finding'].map(base)
    table['verdict'] = [verdict(b, v) if name != 'baseline' else 'baseline'
                        for b, v, name in zip(table['baseline_pp'],
                                              table['value_pp'],
                                              table['variant'])]
    return table


def robustness_summary(table):
    """One line per finding: does it survive every variant?

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`robustness`.

    Returns
    -------
    pandas.DataFrame
        Baseline value, range across variants and overall verdict.
    """
    rows = []
    for finding, group in table.groupby('finding', sort=False):
        others = group[group['variant'] != 'baseline']
        verdicts = set(others['verdict']) - {'not measurable'}
        if 'direction reverses' in verdicts:
            overall = 'not robust (direction reverses)'
        elif verdicts == {'no change in baseline'}:
            overall = 'no change in baseline'
        elif verdicts <= {'robust'}:
            overall = 'robust'
        else:
            overall = 'direction holds; size varies'
        if 'not measurable' in set(others['verdict']):
            overall += ' (some variants not measurable)'
        rows.append({'finding': finding,
                     'baseline_pp': group['baseline_pp'].iloc[0],
                     'min_pp': group['value_pp'].min(),
                     'max_pp': group['value_pp'].max(),
                     'runs_incl_baseline': len(group),
                     'verdict': overall})
    return pd.DataFrame(rows)


# --------------------------------------------------------- reconciliation ----
def integrity_checks(spark, monthly, monthly_cohort, funnel_table, files,
                     shares, decompositions, violations):
    """Run every reconciliation check and report pass or fail.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with ``article_routes`` registered.
    monthly : pandas.DataFrame
        Output of :func:`clickroutes.explore.monthly_profile`.
    monthly_cohort : pandas.DataFrame
        Output of :func:`clickroutes.cohorts.build_monthly_cohort`.
    funnel_table : pandas.DataFrame
        Output of :func:`clickroutes.explore.funnel`.
    files : pandas.DataFrame
        Files in the window.
    shares : pandas.DataFrame
        Baseline output of :func:`clickroutes.routes.route_shares`.
    decompositions : pandas.DataFrame
        Stacked outputs of :func:`clickroutes.decomposition.summarize`.
    violations : pandas.DataFrame
        Output of :func:`clickroutes.explore.type_violations`.

    Returns
    -------
    pandas.DataFrame
        ``check``, ``observed``, ``expected`` and ``passed``.
    """
    routes = route_columns()
    route_sum = ' + '.join(routes)
    ge_checks = ' OR '.join(
        f'n_{r}{s} > n_{r}' for r in config.ROUTES
        for s, m in config.THRESHOLDS if m)
    article = spark.sql(f"""
        SELECT SUM(CASE WHEN n_total <> {route_sum} + n_unmapped
                        THEN 1 ELSE 0 END) AS total_mismatch,
               SUM(CASE WHEN {ge_checks} THEN 1 ELSE 0 END) AS ge_mismatch,
               SUM(pairs) AS pairs
        FROM article_routes""").collect()[0]
    by_month = monthly_cohort.groupby('month')[
        ['n_total', 'n_unmapped'] + routes].sum()
    route_total = by_month[routes].sum(axis=1) + by_month['n_unmapped']
    profile = monthly.set_index('month')
    analysis = monthly_cohort[~monthly_cohort['is_main_page']][routes]
    share_sums = shares.groupby('month')['share'].sum()
    rows = [
        ('Monthly files found', len(files), config.EXPECTED_FILES, None),
        ('Rows in all files', int(monthly['rows'].sum()),
         config.EXPECTED_ROWS, None),
        ('Articles whose route sums differ from their total',
         int(article['total_mismatch']), 0, True),
        ('Articles where a thresholded sum exceeds the full sum',
         int(article['ge_mismatch']), 0, True),
        ('Months where route totals differ from article totals',
         int((route_total != by_month['n_total']).sum()), 0, True),
        ('Months where the article table differs from usable raw requests',
         int((by_month['n_total'] != profile['valid_views']
              .reindex(by_month.index)).sum()), 0, True),
        ('Pairs in the article table vs usable raw rows',
         int(article['pairs']), int(monthly['valid_rows'].sum()), True),
        ('Requests analysed vs funnel result',
         int(analysis.to_numpy().sum()),
         int(funnel_table['requests_kept'].iloc[-1]), True),
        ('Largest deviation of monthly shares from 100%',
         round(float((share_sums - 1).abs().max()), 12), 0.0, True),
        ('Largest decomposition residual (pp)',
         round(float(decompositions['residual_pp'].abs().max()), 9), 0.0,
         True),
        ('Rows whose type contradicts the referrer label',
         int(violations['rows'].sum()) if len(violations) else 0, 0,
         True),
        ('Requests with unexpected referrer labels',
         int(monthly['n_unmapped'].sum()), 0, True),
    ]
    table = pd.DataFrame(rows, columns=['check', 'observed', 'expected',
                                        'strict'], dtype=object)
    table['passed'] = [
        (abs(float(o) - float(e)) < 1e-6) if strict or o == e else
        'differs (information)'
        for o, e, strict in zip(table['observed'], table['expected'],
                                table['strict'])]
    return table.drop(columns='strict')
