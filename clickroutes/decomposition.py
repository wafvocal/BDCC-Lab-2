r"""Split the change in search share into where it comes from.

For each article :math:`a` let :math:`w_a` be its share of all views in a
period and :math:`s_a` its search share, with 0 the early period and 1 the
late one. With :math:`\bar S = (S_0 + S_1)/2` the mean of the two overall
search shares, the change :math:`\Delta S = S_1 - S_0` splits exactly into

* within: :math:`\sum_{a \in M} \bar w_a (s_{a1} - s_{a0})`, the same
  articles reached more or less through search;
* between: :math:`\sum_{a \in M} (w_{a1} - w_{a0})(\bar s_a - \bar S)`,
  readers shifting toward articles that depend more or less on search;
* entry: :math:`\sum_{a \in E} w_{a1}(s_{a1} - \bar S)` and exit:
  :math:`-\sum_{a \in X} w_{a0}(s_{a0} - \bar S)`, articles appearing or
  disappearing,

where :math:`M`, :math:`E` and :math:`X` are titles in both periods, only
the late period and only the early period (Griliches & Regev, 1995).
Centring on :math:`\bar S` makes each term zero when its group looks like
the average, so a term is large only when that group differs.
"""
import pandas as pd

from . import config
from .preparation import analysis_key, cached_table
from .routes import route_columns


def matched_months(months, early_year=None, late_year=None):
    """Calendar months present in both comparison years.

    Parameters
    ----------
    months : iterable of str
        Months that have data.
    early_year, late_year : int, optional
        Default to ``config.EARLY_YEAR`` and ``config.LATE_YEAR``.

    Returns
    -------
    early, late : list of str
        Matched months of each year, in the same calendar order.
    """
    early_year = early_year or config.EARLY_YEAR
    late_year = late_year or config.LATE_YEAR
    months = set(months)
    common = sorted({m[5:] for m in months if m.startswith(f'{early_year}-')}
                    & {m[5:] for m in months
                       if m.startswith(f'{late_year}-')})
    return ([f'{early_year}-{mm}' for mm in common],
            [f'{late_year}-{mm}' for mm in common])


VARIANTS = {
    'baseline': {
        'routes': config.ROUTES, 'suffix': '', 'main_page': False},
    'Main_Page included': {
        'routes': config.ROUTES, 'suffix': '', 'main_page': True},
    'attributed views only': {
        'routes': config.ATTRIBUTED_ROUTES, 'suffix': '',
        'main_page': False},
    'pairs with n >= 50': {
        'routes': config.ROUTES, 'suffix': '_ge50', 'main_page': False},
    'pairs with n >= 100': {
        'routes': config.ROUTES, 'suffix': '_ge100', 'main_page': False},
}


def decomposition_sql(early, late, routes=config.ROUTES, suffix='',
                      main_page=False):
    """Query for the article-level terms of the decomposition.

    Parameters
    ----------
    early, late : list of str
        Matched months of the two periods.
    routes : sequence of str
        Routes in the denominator (drop unattributed for the attributed
        variant).
    suffix : str
        Minimum-pair-count variant of the route columns.
    main_page : bool
        Keep ``Main_Page``.

    Returns
    -------
    str
        Spark SQL; a full outer join on title keeps every view, and the
        result is summed by status (matched, entry, exit) and cohort.
    """
    total = ' + '.join(route_columns(suffix, routes))
    search = f'n_search{suffix}'
    main = '' if main_page else f"AND curr <> '{config.EXCLUDED_TITLE}'"

    def period(months):
        """Requests and search requests per day for each title."""
        listed = ', '.join(f"'{m}'" for m in months)
        return f"""
        SELECT curr, SUM(({total}) / days) AS v, SUM({search} / days) AS sv
        FROM article_routes
        WHERE month IN ({listed}) {main}
        GROUP BY curr
        HAVING SUM({total}) > 0"""

    return f"""
    WITH early AS ({period(early)}),
    late AS ({period(late)}),
    joined AS (
        SELECT COALESCE(e.curr, l.curr) AS curr,
               e.v AS v0, e.sv AS sv0, l.v AS v1, l.sv AS sv1
        FROM early e FULL OUTER JOIN late l ON e.curr = l.curr
    ),
    totals AS (
        SELECT SUM(v0) AS big_v0, SUM(sv0) AS big_sv0,
               SUM(v1) AS big_v1, SUM(sv1) AS big_sv1
        FROM joined
    ),
    terms AS (
        SELECT j.curr,
               CASE WHEN j.v0 IS NOT NULL AND j.v1 IS NOT NULL
                    THEN 'matched'
                    WHEN j.v1 IS NULL THEN 'exit' ELSE 'entry' END AS status,
               j.v0, j.v1, j.sv0, j.sv1,
               j.v0 / t.big_v0 AS w0, j.v1 / t.big_v1 AS w1,
               j.sv0 / j.v0 AS s0, j.sv1 / j.v1 AS s1,
               (t.big_sv0 / t.big_v0 + t.big_sv1 / t.big_v1) / 2 AS s_bar
        FROM joined j CROSS JOIN totals t
    )
    SELECT /*+ BROADCAST(r) */
           status, COALESCE(r.cohort, '{config.REST_COHORT}') AS cohort,
           COUNT(*) AS articles,
           SUM(v0) AS v0, SUM(v1) AS v1, SUM(sv0) AS sv0, SUM(sv1) AS sv1,
           SUM(CASE WHEN status = 'matched'
                    THEN 0.5 * (w0 + w1) * (s1 - s0) ELSE 0 END)
               AS within_term,
           SUM(CASE WHEN status = 'matched'
                    THEN (w1 - w0) * (0.5 * (s0 + s1) - s_bar) ELSE 0 END)
               AS between_term,
           SUM(CASE WHEN status = 'entry'
                    THEN w1 * (s1 - s_bar) ELSE 0 END) AS entry_term,
           SUM(CASE WHEN status = 'exit'
                    THEN -w0 * (s0 - s_bar) ELSE 0 END) AS exit_term
    FROM terms LEFT JOIN rank_full r ON terms.curr = r.curr
    GROUP BY status, COALESCE(r.cohort, '{config.REST_COHORT}')
    """


def decompose(spark, early, late, variant='baseline'):
    """Article-level decomposition terms summed by status and cohort.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session with ``article_routes`` and ``rank_full``.
    early, late : list of str
        Matched months.
    variant : str
        Key of :data:`VARIANTS`.

    Returns
    -------
    pandas.DataFrame
        One row per status and cohort, cached between runs.
    """
    spec = VARIANTS[variant]
    key = analysis_key(early=early, late=late, variant=variant,
                       top_n=config.TOP_N, cohorts=config.COHORTS)
    name = 'decomposition_' + ''.join(
        ch if ch.isalnum() else '_' for ch in variant)
    return cached_table(name, key, lambda: spark.sql(decomposition_sql(
        early, late, spec['routes'], spec['suffix'],
        spec['main_page'])).toPandas())


def summarize(terms, variant='baseline'):
    """Collapse decomposition terms into one row of percentage points.

    Parameters
    ----------
    terms : pandas.DataFrame
        Output of :func:`decompose`.
    variant : str
        Label carried into the result.

    Returns
    -------
    pandas.DataFrame
        Search share in each period, the change, each term, the residual
        (should be zero) and the share of views covered by matched titles.
    """
    v0, v1 = terms['v0'].sum(), terms['v1'].sum()
    s0 = terms['sv0'].sum() / v0
    s1 = terms['sv1'].sum() / v1
    matched = terms[terms['status'] == 'matched']
    parts = {name: 100 * terms[f'{name}_term'].sum()
             for name in ('within', 'between', 'entry', 'exit')}
    change = 100 * (s1 - s0)
    row = {
        'variant': variant,
        'search_share_early_%': 100 * s0,
        'search_share_late_%': 100 * s1,
        'change_pp': change,
        'within_pp': parts['within'],
        'between_pp': parts['between'],
        'entry_pp': parts['entry'],
        'exit_pp': parts['exit'],
        'residual_pp': change - sum(parts.values()),
        'matched_views_early_%': 100 * matched['v0'].sum() / v0,
        'matched_views_late_%': 100 * matched['v1'].sum() / v1,
        'matched_titles': int(matched['articles'].sum()),
        'entering_titles': int(terms.loc[terms['status'] == 'entry',
                                         'articles'].sum()),
        'exiting_titles': int(terms.loc[terms['status'] == 'exit',
                                        'articles'].sum()),
    }
    return pd.DataFrame([row])


def by_cohort(terms):
    """Within and between terms for each popularity cohort (pp).

    Parameters
    ----------
    terms : pandas.DataFrame
        Output of :func:`decompose` for one variant.

    Returns
    -------
    pandas.DataFrame
        One row per cohort with each term in percentage points.
    """
    columns = ['within_term', 'between_term', 'entry_term', 'exit_term']
    table = terms.groupby('cohort', as_index=False)[columns].sum()
    for column in columns:
        table[column.replace('_term', '_pp')] = 100 * table.pop(column)
    order = {name: i for i, name in enumerate(config.COHORT_ORDER)}
    table['_o'] = table['cohort'].map(order)
    return table.sort_values('_o').drop(columns='_o').reset_index(drop=True)
