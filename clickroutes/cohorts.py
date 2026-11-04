"""Popularity rankings and the month x cohort summary table.

Ranking every title would need a global sort of the whole table, so only
the top ``config.TOP_N`` titles are taken with ``ORDER BY ... LIMIT`` and
joined back by broadcast; every other title belongs to the long tail.
"""
import numpy as np
import pandas as pd

from . import config
from .preparation import analysis_key, cached_table, month_range


def top_titles_sql(months=None, limit=None, view='article_routes'):
    """Query for the most-read titles over a set of months.

    Parameters
    ----------
    months : list of str, optional
        Months to rank on; all months when omitted.
    limit : int, optional
        Number of titles; defaults to ``config.TOP_N``.
    view : str
        Temporary view holding the month x article x route table.

    Returns
    -------
    str
        Spark SQL returning ``curr`` and ``views_per_day`` (the sum over
        the months of ``n_total / days``), most read first.
    """
    limit = limit or config.TOP_N
    month_filter = ''
    if months:
        listed = ', '.join(f"'{m}'" for m in months)
        month_filter = f'AND month IN ({listed})'
    return f"""
    SELECT curr, SUM(n_total / days) AS views_per_day
    FROM {view}
    WHERE curr <> '{config.EXCLUDED_TITLE}' {month_filter}
    GROUP BY curr
    ORDER BY views_per_day DESC, curr
    LIMIT {limit}
    """


def assign_cohort(rank):
    """Name of the cohort for a rank (1 = most read)."""
    for label, low, high in config.COHORTS:
        if low <= rank <= high:
            return label
    return config.REST_COHORT


def rank_table(spark, months=None):
    """Rank the top titles and attach their cohort.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with the ``article_routes`` view.
    months : list of str, optional
        Months used for the ranking; all months when omitted.

    Returns
    -------
    pandas.DataFrame
        ``rank``, ``curr``, ``views_per_day`` and ``cohort``.
    """
    top = spark.sql(top_titles_sql(months)).toPandas()
    top.insert(0, 'rank', np.arange(1, len(top) + 1))
    top['cohort'] = [assign_cohort(r) for r in top['rank']]
    return top


def register_ranks(spark, ranks, name):
    """Expose a rank table as a temporary view for broadcast joins.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    ranks : pandas.DataFrame
        Output of :func:`rank_table`.
    name : str
        View name, such as ``rank_full`` or ``rank_base``.
    """
    frame = spark.createDataFrame(
        ranks[['curr', 'rank', 'cohort']].astype({'rank': 'int64'}))
    frame.createOrReplaceTempView(name)


def build_rankings(spark):
    """Whole-window and base-year rankings, cached between runs.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with the ``article_routes`` view.

    Returns
    -------
    rank_full, rank_base : pandas.DataFrame
        Rankings over all months and over ``config.RANKING_BASE_YEAR``.
    """
    base_months = month_range(f'{config.RANKING_BASE_YEAR}-01',
                              f'{config.RANKING_BASE_YEAR}-12')
    key = analysis_key(top_n=config.TOP_N, cohorts=config.COHORTS,
                       excluded=config.EXCLUDED_TITLE,
                       base_year=config.RANKING_BASE_YEAR)
    rank_full = cached_table('rank_full', key,
                             lambda: rank_table(spark))
    rank_base = cached_table('rank_base', key,
                             lambda: rank_table(spark, base_months))
    register_ranks(spark, rank_full, 'rank_full')
    register_ranks(spark, rank_base, 'rank_base')
    return rank_full, rank_base


def monthly_cohort_sql(columns):
    """Query that sums every route column by month and cohort.

    Parameters
    ----------
    columns : list of str
        Numeric columns of the article table to sum.

    Returns
    -------
    str
        Spark SQL with broadcast joins to ``rank_full`` and ``rank_base``.
    """
    rest = config.REST_COHORT
    main = config.EXCLUDED_TITLE
    sums = ',\n           '.join(f'SUM(a.{c}) AS {c}' for c in columns)
    return f"""
    SELECT /*+ BROADCAST(r, b) */
           a.month, a.days,
           COALESCE(r.cohort, '{rest}') AS cohort,
           COALESCE(b.cohort, '{rest}') AS cohort_base,
           a.curr = '{main}' AS is_main_page,
           COUNT(*) AS articles,
           {sums}
    FROM article_routes a
    LEFT JOIN rank_full r ON a.curr = r.curr
    LEFT JOIN rank_base b ON a.curr = b.curr
    GROUP BY a.month, a.days, COALESCE(r.cohort, '{rest}'),
             COALESCE(b.cohort, '{rest}'), a.curr = '{main}'
    """


def build_monthly_cohort(spark, article_table):
    """Month x cohort x Main_Page table that feeds every share query.

    One pass over the article table produces this small table; all route
    shares, cohort shares and their variants are then computed from it in
    Spark SQL.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with ``article_routes``, ``rank_full`` and
        ``rank_base`` registered.
    article_table : pyspark.sql.DataFrame
        The article table (used for its column names).

    Returns
    -------
    pandas.DataFrame
        One row per month, cohort, base-year cohort and Main_Page flag,
        also registered as the view ``monthly_cohort``.
    """
    columns = [c for c in article_table.columns
               if c.startswith(('n_', 'pairs'))]
    key = analysis_key(top_n=config.TOP_N, cohorts=config.COHORTS,
                       excluded=config.EXCLUDED_TITLE,
                       base_year=config.RANKING_BASE_YEAR, columns=columns)
    table = cached_table(
        'monthly_cohort', key,
        lambda: spark.sql(monthly_cohort_sql(columns)).toPandas())
    table = table.sort_values(['month', 'cohort', 'cohort_base',
                               'is_main_page']).reset_index(drop=True)
    spark.createDataFrame(table).createOrReplaceTempView('monthly_cohort')
    return table


def cohort_order(table, column='cohort'):
    """Sort a table by the fixed cohort order (top cohort first)."""
    order = {name: i for i, name in enumerate(config.COHORT_ORDER)}
    return (table.assign(_order=table[column].map(order))
            .sort_values(['_order'] + [c for c in ('route', 'month')
                                       if c in table.columns])
            .drop(columns='_order').reset_index(drop=True))


def concentration(monthly):
    """Share of each month's views held by each cohort.

    Parameters
    ----------
    monthly : pandas.DataFrame
        Output of :func:`build_monthly_cohort`.

    Returns
    -------
    pandas.DataFrame
        ``month``, ``cohort``, ``articles``, ``views_per_day`` and
        ``share_%`` (Main_Page excluded).
    """
    table = monthly[~monthly['is_main_page']]
    grouped = table.groupby(['month', 'days', 'cohort'], as_index=False)[
        ['articles', 'n_total']].sum()
    grouped['views_per_day'] = grouped['n_total'] / grouped['days']
    grouped['share_%'] = 100 * grouped['n_total'] / grouped.groupby(
        'month')['n_total'].transform('sum')
    return pd.DataFrame(grouped)
