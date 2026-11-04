"""Descriptive tables for the raw-data, filtering and Main_Page sections.

Everything here works on the small per-month profile tables written by
:mod:`clickroutes.preparation`, so none of it touches the 30 GB of raw
files again.
"""
import numpy as np
import pandas as pd

from . import config
from .preparation import analysis_key, cached_table, days_in_month

LABEL_ORDER = ('other-search', 'other-empty', 'other-internal',
               'other-external', 'other-other', '(article)', '(null)')


def monthly_profile(tables):
    """One row per month with sizes, views and quality counts.

    Parameters
    ----------
    tables : dict of pandas.DataFrame
        Output of :func:`clickroutes.preparation.load_tables`.

    Returns
    -------
    pandas.DataFrame
        The profile plus ``days``, ``valid_rows``, ``valid_views``,
        ``views_per_day`` and the duplicate counts.
    """
    profile = tables['profile'].copy()
    profile['days'] = profile['month'].map(days_in_month)
    profile['valid_rows'] = profile['rows'] - profile['rows_invalid']
    profile['valid_views'] = profile['n_sum'] - profile['n_invalid']
    profile['views_per_day'] = profile['valid_views'] / profile['days']
    duplicates = tables['duplicates']
    if len(duplicates):
        profile = profile.merge(duplicates, on='month', how='left')
    return profile.sort_values('month').reset_index(drop=True)


def raw_overview(monthly, files):
    """Headline size of the raw data across all months.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.
    files : pandas.DataFrame
        Files in the window.

    Returns
    -------
    pandas.DataFrame
        ``measure`` and ``value`` rows, formatted for display.
    """
    rows = [
        ('Monthly files read', f'{len(files):,}'),
        ('Compressed size', f'{files["gz_bytes"].sum() / 2 ** 30:,.1f} GB'),
        ('Rows ((prev, curr) pairs)', f'{monthly["rows"].sum():,}'),
        ('Requests (sum of n)', f'{monthly["n_sum"].sum():,}'),
        ('Rows per month (min / median / max)',
         _range(monthly['rows'], 'M')),
        ('Requests per day (min / median / max)',
         _range(monthly['views_per_day'], 'M')),
        ('Distinct requested articles per month (approx.)',
         _range(monthly['curr_distinct'], 'M')),
        ('Distinct referrers per month (approx.)',
         _range(monthly['prev_distinct'], 'M')),
    ]
    return pd.DataFrame(rows, columns=['measure', 'value'])


def yearly_summary(monthly):
    """Size of the data and of ``n`` by calendar year.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        Per year: months present, mean rows per month and requests per day
        (millions), the smallest ``n``, the median of the monthly median,
        90th and 99th percentiles of ``n``, and the largest ``n``.
    """
    table = monthly.assign(year=monthly['month'].str[:4])
    grouped = table.groupby('year', as_index=False).agg(
        months=('month', 'count'),
        rows_per_month_M=('rows', 'mean'),
        requests_per_day_M=('views_per_day', 'mean'),
        smallest_n=('n_min', 'min'),
        median_n=('n_p50', 'median'),
        p90_n=('n_p90', 'median'),
        p99_n=('n_p99', 'median'),
        largest_n=('n_max', 'max'))
    grouped['rows_per_month_M'] /= 1e6
    grouped['requests_per_day_M'] /= 1e6
    return grouped


def _range(series, unit=''):
    """Format min / median / max of a series, in millions if asked."""
    scale = 1e6 if unit == 'M' else 1
    values = [series.min(), series.median(), series.max()]
    return ' / '.join(f'{v / scale:,.2f}{unit}' for v in values)


def column_table(monthly, labels):
    """Describe each raw column with simple statistics.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.
    labels : pandas.DataFrame
        The ``labels`` profile table.

    Returns
    -------
    pandas.DataFrame
        One row per column: type, meaning, nulls and key statistics.
    """
    usable = labels[labels['valid']]
    views = usable.groupby('prev_label')['n'].sum()
    article_share = 100 * views.get('(article)', 0) / views.sum()
    n_labels = int(sum(str(k).startswith('other-') for k in views.index))
    type_views = usable.groupby('type')['n'].sum()
    type_text = ', '.join(f'{k} {100 * v / type_views.sum():.1f}%'
                          for k, v in type_views.sort_values(
                              ascending=False).items())
    q = {f'p{p}': monthly[f'n_p{p}'] for p in (25, 50, 75, 90, 99)}
    rows = [
        ('prev', 'string', 'Mapped referrer: an article title or one of '
         'five other-* labels', int(monthly['null_prev'].sum()),
         f'{article_share:.1f}% of usable requests come from article '
         f'titles; the rest from {n_labels} other-* labels'),
        ('curr', 'string', 'Title of the requested article',
         int(monthly['null_curr'].sum()),
         f'{monthly["curr_distinct"].min() / 1e6:.2f}M to '
         f'{monthly["curr_distinct"].max() / 1e6:.2f}M distinct per month'),
        ('type', 'string', 'link, external or other (relation of prev to '
         'curr)', int(monthly['null_type'].sum()),
         f'Share of requests: {type_text}'),
        ('n', 'long', 'Requests for the pair in the month',
         int(monthly['null_n'].sum()),
         f'min {monthly["n_min"].min():,}; median of monthly medians '
         f'{q["p50"].median():,.0f}; monthly p99 {q["p99"].min():,.0f} to '
         f'{q["p99"].max():,.0f}; max {monthly["n_max"].max():,}'),
    ]
    return pd.DataFrame(rows, columns=['column', 'type', 'meaning',
                                       'nulls', 'statistics'])


def quality_table(monthly):
    """Null, blank and malformed counts per column across all months.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        Total count and number of months affected for each problem.
    """
    checks = {
        'Malformed line (wrong field count or unparseable n)':
            'rows_malformed',
        'prev is null': 'null_prev', 'curr is null': 'null_curr',
        'type is null': 'null_type', 'n is null': 'null_n',
        'prev is blank': 'blank_prev', 'curr is blank': 'blank_curr',
        'Unusable row (any of the above)': 'rows_invalid',
    }
    rows = [(label, int(monthly[col].sum()),
             int((monthly[col] > 0).sum()))
            for label, col in checks.items()]
    return pd.DataFrame(rows, columns=['problem', 'rows', 'months_affected'])


def label_type_table(labels):
    """Cross-tabulate referrer labels against ``type`` over all months.

    Parameters
    ----------
    labels : pandas.DataFrame
        The ``labels`` profile table.

    Returns
    -------
    pandas.DataFrame
        Requests by label and type as shares of all usable requests (%),
        plus the route each label maps to.
    """
    usable = labels[labels['valid']]
    table = usable.pivot_table(index='prev_label', columns='type',
                               values='n', aggfunc='sum', fill_value=0)
    table = 100 * table / table.to_numpy().sum()
    order = [label for label in LABEL_ORDER if label in table.index]
    order += [label for label in table.index if label not in order]
    table = table.reindex(order)
    routes = []
    for label in table.index:
        if label == '(article)':
            route = 'link'
        else:
            route = config.LABEL_TO_ROUTE.get(label, '')
        routes.append(config.ROUTE_NAMES.get(route, 'unexpected'))
    table['route'] = routes
    return table.round(3).rename_axis(columns=None).reset_index()


def type_violations(labels):
    """Rows whose ``type`` contradicts their referrer label.

    Parameters
    ----------
    labels : pandas.DataFrame
        The ``labels`` profile table.

    Returns
    -------
    pandas.DataFrame
        Label, type, rows and requests of every inconsistent combination
        among usable rows (other-* labels must be ``external``; article
        titles must not be).
    """
    usable = labels[labels['valid']]
    is_label = usable['prev_label'].str.startswith('other-')
    bad = ((is_label & (usable['type'] != 'external'))
           | ((usable['prev_label'] == '(article)')
              & (usable['type'] == 'external')))
    return (usable[bad].groupby(['prev_label', 'type'], as_index=False)
            [['rows', 'n']].sum())


def label_sets(labels):
    """The set of other-* labels seen in each month.

    Parameters
    ----------
    labels : pandas.DataFrame
        The ``labels`` profile table.

    Returns
    -------
    pandas.DataFrame
        ``month``, ``labels`` (comma-separated) and ``as_expected``.
    """
    expected = ', '.join(sorted(config.LABEL_TO_ROUTE))
    rows = []
    for month, group in labels[labels['valid']].groupby('month'):
        seen = sorted(set(group.loc[group['prev_label']
                                    .str.startswith('other-'), 'prev_label']))
        rows.append((month, ', '.join(seen), ', '.join(seen) == expected))
    return pd.DataFrame(rows, columns=['month', 'labels', 'as_expected'])


def funnel(monthly):
    """Rows and requests kept at each filtering step.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        One row per step with rows and requests removed and kept, as
        counts and as percentages of the previous step and of the raw
        data.
    """
    raw_rows = int(monthly['rows'].sum())
    raw_views = int(monthly['n_sum'].sum())
    steps = [
        (f'Raw rows in the {len(monthly)} monthly files', 0, 0),
        ('Drop malformed or incomplete rows',
         int(monthly['rows_invalid'].sum()),
         int(monthly['n_invalid'].sum())),
        ('Drop unexpected referrer labels',
         int(monthly['rows_unmapped'].sum()),
         int(monthly['n_unmapped'].sum())),
        ('Set aside Main_Page as requested article',
         int(monthly['rows_main_page'].sum()),
         int(monthly['n_main_page'].sum())),
        ('Duplicate (prev, curr) pairs (kept, summed)', 0, 0),
    ]
    rows, kept_rows, kept_views = [], raw_rows, raw_views
    for label, removed_rows, removed_views in steps:
        before = kept_rows
        kept_rows -= removed_rows
        kept_views -= removed_views
        rows.append({
            'step': label,
            'rows_removed': removed_rows,
            'rows_kept': kept_rows,
            'rows_kept_%_of_previous': 100 * kept_rows / before,
            'rows_kept_%_of_raw': 100 * kept_rows / raw_rows,
            'requests_removed': removed_views,
            'requests_kept': kept_views,
            'requests_kept_%_of_raw': 100 * kept_views / raw_views,
        })
    return pd.DataFrame(rows)


def duplicate_table(monthly):
    """Duplicate (prev, curr) pairs found within a month.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        Totals over all months and the share of requests involved.
    """
    rows = [
        ('Pairs appearing more than once in a month',
         int(monthly['duplicate_pairs'].sum())),
        ('Extra rows beyond the first', int(monthly['extra_rows'].sum())),
        ('... of which the rows differ in type',
         int(monthly['pairs_type_differs'].sum())),
        ('... of which the rows are identical',
         int(monthly['pairs_identical'].sum())),
        ('Months with any duplicate pair',
         int((monthly['duplicate_pairs'] > 0).sum())),
    ]
    share = (100 * monthly['n_in_duplicate_pairs'].sum()
             / monthly['valid_views'].sum())
    rows.append(('Share of usable requests in duplicate pairs (%)',
                 round(float(share), 5)))
    return pd.DataFrame(rows, columns=['measure', 'value'], dtype=object)


def main_page_table(monthly):
    """How much traffic Main_Page receives and from where.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        Per month: Main_Page requests per day, its share of all usable
        requests, the share of its requests with an empty referrer, and
        requests that Main_Page sends to articles.
    """
    table = monthly[['month', 'days']].copy()
    table['main_page_per_day'] = monthly['n_main_page'] / monthly['days']
    table['share_of_all_%'] = (100 * monthly['n_main_page']
                               / monthly['valid_views'])
    table['empty_referrer_%'] = (100 * monthly['n_main_page_empty']
                                 / monthly['n_main_page'].replace(0, np.nan))
    table['sent_to_articles_per_day'] = (monthly['n_from_main_page']
                                         / monthly['days'])
    return table.drop(columns='days')


def n_cdf(n_bins):
    """Cumulative share of rows and requests by pair size, all months.

    Parameters
    ----------
    n_bins : pandas.DataFrame
        The ``n_bins`` profile table (log2 bins of n).

    Returns
    -------
    pandas.DataFrame
        ``lower`` (smallest n in the bin), share of rows and of requests
        in the bin, and their cumulative shares (%).
    """
    table = n_bins.groupby('log2_bin', as_index=False)[['rows', 'n']].sum()
    table['lower'] = 2.0 ** table['log2_bin']
    table['rows_%'] = 100 * table['rows'] / table['rows'].sum()
    table['requests_%'] = 100 * table['n'] / table['n'].sum()
    table['cum_rows_%'] = table['rows_%'].cumsum()
    table['cum_requests_%'] = table['requests_%'].cumsum()
    return table


def truncation_table(monthly):
    """Share of usable rows and requests in small pairs.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`monthly_profile`.

    Returns
    -------
    pandas.DataFrame
        For each minimum pair count, the share of rows and requests that
        a common floor would remove, over all months and in the first and
        last month.
    """
    rows = []
    first, last = monthly.iloc[0], monthly.iloc[-1]
    for minimum in config.MIN_PAIR_COUNTS:
        rows_col, views_col = f'rows_lt{minimum}', f'n_lt{minimum}'
        rows.append({
            'minimum_n': minimum,
            'rows_removed_%': 100 * monthly[rows_col].sum()
            / monthly['valid_rows'].sum(),
            'requests_removed_%': 100 * monthly[views_col].sum()
            / monthly['valid_views'].sum(),
            f'requests_removed_%_{first["month"]}':
                100 * first[views_col] / first['valid_views'],
            f'requests_removed_%_{last["month"]}':
                100 * last[views_col] / last['valid_views'],
        })
    return pd.DataFrame(rows)


def top_referrers(top_prev, k=12):
    """Articles that most often send readers to other articles.

    Parameters
    ----------
    top_prev : pandas.DataFrame
        The ``top_prev`` profile table (each month's top 10 referrers).
    k : int
        Number of rows.

    Returns
    -------
    pandas.DataFrame
        Referrer, months in a monthly top 10, and requests per day over
        those months.
    """
    table = top_prev.copy()
    table['days'] = table['month'].map(days_in_month)
    grouped = table.groupby('prev').agg(months_in_top10=('month', 'count'),
                                        requests=('n', 'sum'),
                                        days=('days', 'sum'))
    grouped['requests_per_day'] = grouped['requests'] / grouped['days']
    return (grouped.sort_values(['months_in_top10', 'requests_per_day'],
                                ascending=False)
            .head(k).reset_index()[['prev', 'months_in_top10',
                                    'requests_per_day']])


def unattributed_heavy(spark):
    """Articles that receive nearly all their requests unattributed.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with the ``article_routes`` view.

    Returns
    -------
    by_month : pandas.DataFrame
        Per month, how many articles with at least
        ``config.HEAVY_MIN_VIEWS_PER_DAY`` requests per day get at least
        ``config.HEAVY_UNATTRIBUTED_SHARE`` of them unattributed, and
        their share of all unattributed requests.
    latest : pandas.DataFrame
        The largest such articles in the last month.
    """
    share = config.HEAVY_UNATTRIBUTED_SHARE
    floor = config.HEAVY_MIN_VIEWS_PER_DAY
    main = config.EXCLUDED_TITLE
    key = analysis_key(share=share, floor=floor, excluded=main)
    by_month = cached_table('unattributed_heavy_by_month', key,
                            lambda: spark.sql(f"""
        SELECT month,
               SUM(CASE WHEN heavy THEN 1 ELSE 0 END) AS heavy_articles,
               SUM(CASE WHEN heavy THEN n_unattributed ELSE 0 END)
                   / SUM(n_unattributed) AS share_of_unattributed
        FROM (SELECT month, n_unattributed,
                     n_total / days >= {floor}
                     AND n_unattributed >= {share} * n_total AS heavy
              FROM article_routes WHERE curr <> '{main}') t
        GROUP BY month ORDER BY month""").toPandas())
    last = by_month['month'].max()
    latest = cached_table('unattributed_heavy_latest', key,
                          lambda: spark.sql(f"""
        SELECT curr, n_total / days AS requests_per_day,
               n_unattributed / n_total AS unattributed_share,
               n_empty / n_total AS empty_share
        FROM article_routes
        WHERE month = '{last}' AND curr <> '{main}'
          AND n_total / days >= {floor}
          AND n_unattributed >= {share} * n_total
        ORDER BY requests_per_day DESC
        LIMIT 12""").toPandas())
    return by_month, latest
