"""The route definition and the Spark SQL that turns sums into shares.

Every analysis uses :data:`ROUTE_CASE_SQL`, so there is exactly one
definition of a route. Titles on English Wikipedia always start with a
capital letter, so no article can begin with the lowercase ``other-``
prefix that marks the mapped referrer labels.
"""
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from . import config

ROUTE_CASE_SQL = """CASE
    WHEN prev = 'other-search' THEN 'search'
    WHEN prev = 'other-external' THEN 'external'
    WHEN prev = 'other-internal' THEN 'wikimedia'
    WHEN prev IN ('other-empty', 'other-other') THEN 'unattributed'
    WHEN prev LIKE 'other-%' THEN 'unmapped'
    ELSE 'link'
END"""


def register_article_view(spark, name='article_routes'):
    """Expose the month x article x route table as a temporary view.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    name : str
        Name of the temporary view.

    Returns
    -------
    pyspark.sql.DataFrame
        The table with an added ``days`` column (days in the month).
    """
    table = spark.read.parquet(str(config.ARTICLE_DIR))
    first_day = F.to_date(F.concat(F.col('month'), F.lit('-01')))
    table = table.withColumn('days', F.dayofmonth(F.last_day(first_day)))
    table.createOrReplaceTempView(name)
    return table


def route_columns(suffix='', routes=None):
    """Column names that hold the per-route sums.

    Parameters
    ----------
    suffix : str
        ``''`` for all pairs, or ``'_ge50'``/``'_ge100'`` for pairs with at
        least that many requests.
    routes : sequence of str, optional
        Routes to include; defaults to all five.

    Returns
    -------
    list of str
        For example ``['n_search', 'n_link', ...]``.
    """
    routes = config.ROUTES if routes is None else routes
    return [f'n_{route}{suffix}' for route in routes]


def share_sql(view='monthly_cohort', suffix='', include_main_page=False,
              attributed_only=False, group=None):
    """Build the query that sums n by month and route and adds shares.

    Parameters
    ----------
    view : str
        Temporary view with one row per month (and group) and one column
        per route, such as ``monthly_cohort``.
    suffix : str
        Minimum-pair-count variant of the route columns.
    include_main_page : bool
        Keep ``Main_Page`` (it is placed in the top cohort when grouped).
    attributed_only : bool
        Drop unattributed arrivals from the denominator.
    group : str, optional
        Extra grouping column, such as ``cohort`` or ``cohort_base``.

    Returns
    -------
    str
        Spark SQL that returns ``month, [grp,] route, n, views_per_day,
        share``, where ``share`` comes from ``SUM(n) OVER (PARTITION BY
        month [, grp])``. ``NULLIF`` turns an empty group into a missing
        share instead of a division-by-zero error (Spark 4 runs SQL in
        ANSI mode, which raises on division by zero).
    """
    routes = config.ATTRIBUTED_ROUTES if attributed_only else config.ROUTES
    stack = ', '.join(f"'{route}', n_{route}{suffix}" for route in routes)
    where = '' if include_main_page else 'WHERE NOT is_main_page'
    top = config.COHORTS[0][0]
    if group:
        grp_expr = (f"CASE WHEN is_main_page THEN '{top}' ELSE {group} END"
                    if include_main_page else group)
        grp_select, grp_key, grp_part = f'{grp_expr} AS grp, ', 'grp, ', \
            ', grp'
    else:
        grp_select = grp_key = grp_part = ''
    return f"""
    WITH long AS (
        SELECT month, days, {grp_select}
               stack({len(routes)}, {stack}) AS (route, n)
        FROM {view}
        {where}
    ),
    by_route AS (
        SELECT month, days, {grp_key}route, SUM(n) AS n
        FROM long
        GROUP BY month, days, {grp_key}route
    )
    SELECT month, {grp_key}route, n,
           n / days AS views_per_day,
           n / NULLIF(SUM(n) OVER (PARTITION BY month{grp_part}), 0)
               AS share
    FROM by_route
    ORDER BY month, {grp_key}route
    """


def route_shares(spark, **kwargs):
    """Run :func:`share_sql` and return a pandas table.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with the ``monthly_cohort`` view registered.
    **kwargs
        Passed to :func:`share_sql`.

    Returns
    -------
    pandas.DataFrame
        Monthly sums, per-day views and shares by route.
    """
    table = spark.sql(share_sql(**kwargs)).toPandas()
    if 'grp' in table.columns:
        table = table.rename(columns={'grp': kwargs.get('group')})
    return table


def period_change(shares, early, late, by=('route',)):
    """Compare matched months of two periods.

    Shares are pooled over the matched months from per-day views, the same
    way the decomposition pools them, so the two sections agree.

    Parameters
    ----------
    shares : pandas.DataFrame
        Output of :func:`route_shares`.
    early, late : list of str
        Matched months of the early and late periods.
    by : tuple of str
        Grouping columns, ending with ``route``.

    Returns
    -------
    pandas.DataFrame
        Share in each period (%), change in percentage points, mean views
        per day in each period and its percentage change.
    """
    keys = list(by)
    outer = [k for k in keys if k != 'route']
    table = shares[shares['month'].isin(list(early) + list(late))].copy()
    table['period'] = np.where(table['month'].isin(early), 'early', 'late')
    sums = (table.groupby(keys + ['period'])['views_per_day'].sum()
            .unstack('period'))
    if outer:
        totals = sums.groupby(level=outer).transform('sum')
    else:
        totals = sums.sum()
    shares_pct = 100 * sums / totals
    months = len(early)
    result = pd.DataFrame({
        'share_early_%': shares_pct['early'],
        'share_late_%': shares_pct['late'],
        'change_pp': shares_pct['late'] - shares_pct['early'],
        'views_per_day_early': sums['early'] / months,
        'views_per_day_late': sums['late'] / months,
    })
    result['views_change_%'] = 100 * (result['views_per_day_late']
                                      / result['views_per_day_early'] - 1)
    result = result.reset_index()
    order = {route: i for i, route in enumerate(config.ROUTES)}
    result['_order'] = result['route'].map(order)
    result = result.sort_values(outer + ['_order'])
    return result.drop(columns='_order').reset_index(drop=True)
