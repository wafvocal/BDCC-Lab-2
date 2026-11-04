"""Find, stage, profile and aggregate the raw monthly files.

Gzip files cannot be split, so one Spark task reads each file. A batch of
months is read in parallel and staged to Parquet once, every check that
needs the raw rows runs on that stage (nulls, malformed lines, labels,
types, duplicates, the distribution of ``n``), and each month is then
aggregated into the month x article x route table that all later analysis
reads. Keeping pair-level Parquet for every month would need about
30 GB x 1.7, above the 50 GB quota, so the stage is deleted after each
batch, which is the game plan's fallback. The folder size is checked
after every month and the run stops before the quota is reached.

Finished months are recorded in ``tables/done``, so an interrupted run
resumes where it stopped.
"""
import gc
import json
import os
import re
import shutil
import time
from functools import reduce
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import config
from .routes import ROUTE_CASE_SQL

RAW_SCHEMA = T.StructType([
    T.StructField('prev', T.StringType(), True),
    T.StructField('curr', T.StringType(), True),
    T.StructField('type', T.StringType(), True),
    T.StructField('n', T.LongType(), True),
    T.StructField('_corrupt_record', T.StringType(), True),
])
TABLES = ('profile', 'labels', 'n_bins', 'duplicates', 'duplicate_examples',
          'top_prev', 'bad_rows')
QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.99)
GIB = 2 ** 30


# ------------------------------------------------------------- helpers ----
def month_range(first, last):
    """List every calendar month from ``first`` to ``last``.

    Parameters
    ----------
    first, last : str
        Months as ``'YYYY-MM'``.

    Returns
    -------
    list of str
        All months in between, inclusive.
    """
    periods = pd.period_range(first, last, freq='M')
    return [str(period) for period in periods]


def days_in_month(month):
    """Number of days in a ``'YYYY-MM'`` month."""
    return pd.Period(month, freq='M').days_in_month


def dir_size(path):
    """Total size in bytes of every file under ``path`` (0 if missing)."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def valid_row():
    """Spark condition for a row that can be used in the analysis.

    Returns
    -------
    pyspark.sql.Column
        True when the line parsed into all four fields and ``prev``,
        ``curr`` and ``n`` are present.
    """
    return (F.col('_corrupt_record').isNull()
            & F.col('prev').isNotNull()
            & F.col('curr').isNotNull()
            & F.col('n').isNotNull())


def _is_label():
    """Spark condition for a mapped referrer label such as other-search."""
    return F.col('prev').startswith('other-')


# ------------------------------------------------------------ discovery ----
def discover_files(data_dir=None, first=None, last=None):
    """Find the English monthly files and keep those inside the window.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Folder searched recursively; defaults to ``config.DATA_DIR``.
    first, last : str, optional
        Window limits as ``'YYYY-MM'``; default to the config window.

    Returns
    -------
    files : pandas.DataFrame
        One row per month in the window with ``month``, ``path`` and
        ``gz_bytes``.
    found : pandas.DataFrame
        Every English monthly file found, inside or outside the window.

    Raises
    ------
    FileNotFoundError
        If the data folder does not exist.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    first = first or config.FIRST_MONTH
    last = last or config.LAST_MONTH
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f'{data_dir} was not found. In a Jojie terminal run\n'
            "  find /mnt/data/public -maxdepth 4 -iname '*clickstream*'\n"
            'and put the folder that holds the YYYY-MM subfolders in '
            'config.DATA_DIR.')
    pattern = re.compile(config.FILE_REGEX)
    rows = []
    for path in sorted(data_dir.rglob(config.FILE_GLOB)):
        match = pattern.match(path.name)
        if match:
            rows.append({'month': match.group(1), 'path': str(path),
                         'gz_bytes': path.stat().st_size})
    found = pd.DataFrame(rows, columns=['month', 'path', 'gz_bytes'])
    inside = found['month'].between(first, last)
    files = (found[inside].drop_duplicates('month')
             .sort_values('month').reset_index(drop=True))
    return files, found


def coverage_table(files, first=None, last=None):
    """Mark which months of the window have a file.

    Parameters
    ----------
    files : pandas.DataFrame
        Output of :func:`discover_files`.
    first, last : str, optional
        Window limits; default to the config window.

    Returns
    -------
    pandas.DataFrame
        ``month``, ``year``, ``month_of_year`` and boolean ``present``.
    """
    months = month_range(first or config.FIRST_MONTH,
                         last or config.LAST_MONTH)
    present = set(files['month'])
    table = pd.DataFrame({'month': months})
    table['year'] = table['month'].str[:4].astype(int)
    table['month_of_year'] = table['month'].str[5:].astype(int)
    table['present'] = table['month'].isin(present)
    return table


# -------------------------------------------------------------- reading ----
def read_raw(spark, path, month):
    """Read one monthly file with the explicit schema.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    path : str
        Path of a ``.tsv.gz`` file.
    month : str
        Month taken from the file name.

    Returns
    -------
    pyspark.sql.DataFrame
        Columns ``prev, curr, type, n, _corrupt_record, month``. Quoting is
        disabled so titles containing quotation marks stay intact, and
        malformed lines keep their raw text in ``_corrupt_record``.
    """
    return (spark.read
            .option('sep', '\t')
            .option('header', 'false')
            .option('quote', '')
            .option('mode', 'PERMISSIVE')
            .option('columnNameOfCorruptRecord', '_corrupt_record')
            .schema(RAW_SCHEMA)
            .csv(path)
            .withColumn('month', F.lit(month)))


def stage_batch(spark, batch):
    """Read a batch of months in parallel and write them as Parquet.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    batch : pandas.DataFrame
        Rows of :func:`discover_files` to stage.
    """
    shutil.rmtree(config.STAGE_DIR, ignore_errors=True)
    frames = [read_raw(spark, row.path, row.month)
              for row in batch.itertuples(index=False)]
    raw = reduce(DataFrame.unionByName, frames)
    (raw.write.mode('overwrite').partitionBy('month')
     .option('compression', 'snappy').parquet(str(config.STAGE_DIR)))


# ------------------------------------------------------------ profiling ----
def _profile_aggregations():
    """Aggregations for the per-month profile of the raw rows."""
    n = F.col('n')
    curr_main = F.col('curr') == config.EXCLUDED_TITLE
    valid = valid_row()
    expected = list(config.LABEL_TO_ROUTE)
    unmapped = valid & _is_label() & ~F.col('prev').isin(expected)
    quantiles = ', '.join(str(q) for q in QUANTILES)

    def count_if(condition):
        """Number of rows where ``condition`` holds."""
        return F.sum(F.when(condition, 1).otherwise(0))

    def views_if(condition):
        """Sum of ``n`` over rows where ``condition`` holds."""
        return F.sum(F.when(condition, n).otherwise(F.lit(0)))

    return [
        F.count(F.lit(1)).alias('rows'),
        count_if(F.col('_corrupt_record').isNotNull()).alias(
            'rows_malformed'),
        count_if(F.col('prev').isNull()).alias('null_prev'),
        count_if(F.col('curr').isNull()).alias('null_curr'),
        count_if(F.col('type').isNull()).alias('null_type'),
        count_if(F.col('n').isNull()).alias('null_n'),
        count_if(F.trim(F.col('prev')) == '').alias('blank_prev'),
        count_if(F.trim(F.col('curr')) == '').alias('blank_curr'),
        count_if(~valid).alias('rows_invalid'),
        views_if(~valid & n.isNotNull()).alias('n_invalid'),
        F.sum(n).alias('n_sum'),
        F.min(n).alias('n_min'),
        F.max(n).alias('n_max'),
        F.avg(n).alias('n_mean'),
        F.expr(f'percentile_approx(n, array({quantiles}), '
               f'{config.PERCENTILE_ACCURACY})').alias('n_quantiles'),
        F.approx_count_distinct('prev', 0.01).alias('prev_distinct'),
        F.approx_count_distinct(
            F.when(~_is_label(), F.col('prev')), 0.01).alias(
                'prev_articles'),
        F.approx_count_distinct('curr', 0.01).alias('curr_distinct'),
        count_if(unmapped).alias('rows_unmapped'),
        views_if(unmapped).alias('n_unmapped'),
        count_if(valid & curr_main & ~unmapped).alias('rows_main_page'),
        views_if(valid & curr_main & ~unmapped).alias('n_main_page'),
        views_if(valid & curr_main
                 & (F.col('prev') == 'other-empty')).alias(
                     'n_main_page_empty'),
        views_if(valid & (F.col('prev') == config.EXCLUDED_TITLE)).alias(
            'n_from_main_page'),
        count_if(valid & (n < 50)).alias('rows_lt50'),
        views_if(valid & (n < 50)).alias('n_lt50'),
        count_if(valid & (n < 100)).alias('rows_lt100'),
        views_if(valid & (n < 100)).alias('n_lt100'),
    ]


def profile_stage(spark):
    """Profile every month in the current stage.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.

    Returns
    -------
    dict of pandas.DataFrame
        Tables named as in :data:`TABLES`, each with a ``month`` column.
    """
    stage = spark.read.parquet(str(config.STAGE_DIR))
    months = sorted(r.month for r in stage.select('month').distinct()
                    .collect())
    valid = valid_row()
    profile = stage.groupBy('month').agg(*_profile_aggregations()).toPandas()
    quantile_rows = profile.pop('n_quantiles').tolist()
    for i, q in enumerate(QUANTILES):
        profile[f'n_p{int(q * 100)}'] = [
            row[i] if row is not None else None for row in quantile_rows]

    label = (F.when(F.col('prev').isNull(), F.lit('(null)'))
             .when(_is_label(), F.col('prev'))
             .otherwise(F.lit('(article)')))
    labels = (stage.groupBy('month', label.alias('prev_label'),
                            F.coalesce('type', F.lit('(null)')).alias('type'),
                            valid.alias('valid'))
              .agg(F.count(F.lit(1)).alias('rows'), F.sum('n').alias('n'))
              .toPandas())

    n_bins = (stage.where(valid & (F.col('n') > 0))
              .groupBy('month',
                       F.floor(F.log2('n')).cast('int').alias('log2_bin'))
              .agg(F.count(F.lit(1)).alias('rows'), F.sum('n').alias('n'))
              .toPandas())

    # Duplicates: shuffle only a 64-bit hash of (month, prev, curr), then
    # look at the actual titles for the few hashes seen more than once.
    # A hash collision can only add a candidate, never hide a duplicate,
    # and the second grouping on the real titles removes any collision.
    pair_hash = F.xxhash64('month', 'prev', 'curr').alias('pair_hash')
    repeated = (stage.where(valid).select(pair_hash)
                .groupBy('pair_hash').count()
                .where(F.col('count') > 1).select('pair_hash').cache())
    pairs = (stage.where(valid).withColumn('pair_hash', pair_hash)
             .join(repeated, 'pair_hash')
             .groupBy('month', 'prev', 'curr')
             .agg(F.count(F.lit(1)).alias('rows'), F.sum('n').alias('n'),
                  (F.min('type') != F.max('type')).alias('type_differs'),
                  (F.min('n') == F.max('n')).alias('same_n'))
             .where(F.col('rows') > 1)
             .cache())
    identical = ~F.col('type_differs') & F.col('same_n')
    duplicates = (pairs.groupBy('month')
                  .agg(F.count(F.lit(1)).alias('duplicate_pairs'),
                       F.sum(F.col('rows') - 1).alias('extra_rows'),
                       F.sum('n').alias('n_in_duplicate_pairs'),
                       F.sum(F.col('type_differs').cast('int')).alias(
                           'pairs_type_differs'),
                       F.sum(identical.cast('int')).alias(
                           'pairs_identical'))
                  .toPandas())
    duplicate_examples = (pairs.orderBy(F.desc('n')).limit(20).toPandas())
    pairs.unpersist()
    repeated.unpersist()
    duplicates = (pd.DataFrame({'month': months})
                  .merge(duplicates, on='month', how='left').fillna(0))

    ranked = Window.partitionBy('month').orderBy(F.desc('n'), 'prev')
    top_prev = (stage.where(valid & ~_is_label())
                .groupBy('month', 'prev').agg(F.sum('n').alias('n'))
                .withColumn('rank', F.row_number().over(ranked))
                .where(F.col('rank') <= 10)
                .toPandas())

    bad_rows = (stage.where(~valid)
                .select('month', 'prev', 'curr', 'type', 'n',
                        '_corrupt_record')
                .limit(50).toPandas())
    return {'profile': profile, 'labels': labels, 'n_bins': n_bins,
            'duplicates': duplicates,
            'duplicate_examples': duplicate_examples,
            'top_prev': top_prev, 'bad_rows': bad_rows}


def save_tables(tables, months):
    """Write each profile table as one small CSV per month.

    Parameters
    ----------
    tables : dict of pandas.DataFrame
        Output of :func:`profile_stage`.
    months : iterable of str
        Months of the batch (a file is written even when a table has no
        rows for a month, so the month is never mistaken for missing).
    """
    for name, table in tables.items():
        folder = config.TABLE_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        for month in months:
            part = table[table['month'] == month] if len(table) else table
            part.to_csv(folder / f'{month}.csv', index=False)


def load_tables():
    """Read every per-month profile table back into pandas.

    Returns
    -------
    dict of pandas.DataFrame
        One table per name in :data:`TABLES`, months stacked. Strings such
        as ``NaN`` or ``Null`` stay strings because they are real titles.
    """
    tables = {}
    for name in TABLES:
        paths = sorted((config.TABLE_DIR / name).glob('*.csv'))
        frames = [read_small_csv(path) for path in paths
                  if path.stat().st_size > 0]
        filled = [frame for frame in frames if len(frame)]
        if filled:
            tables[name] = pd.concat(filled, ignore_index=True)
        elif frames:
            tables[name] = frames[0]
        else:
            tables[name] = pd.DataFrame()
    return tables


def read_small_csv(path):
    """Read a small result CSV, keeping titles such as ``NaN`` as text.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV written by this package.

    Returns
    -------
    pandas.DataFrame
        The table, with ``month`` read as text.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[''],
                       dtype={'month': str})


def cached_table(name, key, builder):
    """Return a small result table, rebuilding it only when inputs change.

    Parameters
    ----------
    name : str
        File stem inside ``config.RESULT_DIR``.
    key : dict
        Everything the table depends on (months converted, settings).
    builder : callable
        Function with no arguments that computes the table.

    Returns
    -------
    pandas.DataFrame
        The cached or freshly built table.
    """
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = config.RESULT_DIR / f'{name}.csv'
    meta = config.RESULT_DIR / f'{name}.json'
    key = json.loads(json.dumps(key))
    if data.exists() and meta.exists():
        if json.loads(meta.read_text()) == key:
            return read_small_csv(data)
    table = builder()
    table.to_csv(data, index=False)
    meta.write_text(json.dumps(key))
    return table


def analysis_key(**settings):
    """Fingerprint of the converted data plus the given settings.

    Parameters
    ----------
    **settings
        Any JSON-serialisable settings the cached table depends on.

    Returns
    -------
    dict
        Converted months, the time each finished, and the settings.
    """
    folder = config.TABLE_DIR / 'done'
    stamps = {p.stem: round(p.stat().st_mtime) for p in
              sorted(folder.glob('*.json'))} if folder.exists() else {}
    return {'months': stamps, 'settings': settings}


# ---------------------------------------------------------- aggregation ----
def article_aggregations():
    """Per-article sums written to the month x article x route table.

    Returns
    -------
    list of pyspark.sql.Column
        ``n_total`` and ``pairs``; ``n_<route>`` for every route at each
        minimum pair count (suffix ``''``, ``_ge50``, ``_ge100``);
        ``pairs_ge50`` and ``pairs_ge100``; and three diagnostics:
        ``n_unmapped`` (unexpected labels), ``n_empty`` (other-empty only)
        and ``n_link_other`` (article referrer without a link).
    """
    n = F.col('n')
    route = F.col('route')
    zero = F.lit(0)
    aggregations = [F.sum(n).alias('n_total'),
                    F.count(F.lit(1)).alias('pairs')]
    for suffix, minimum in config.THRESHOLDS:
        keep = F.lit(True) if minimum is None else n >= minimum
        for name in config.ROUTES:
            aggregations.append(
                F.sum(F.when(keep & (route == name), n).otherwise(zero))
                .alias(f'n_{name}{suffix}'))
        if minimum is not None:
            aggregations.append(
                F.sum(F.when(keep, 1).otherwise(0)).alias(f'pairs{suffix}'))
    link_other = (route == 'link') & (F.col('type') == 'other')
    aggregations += [
        F.sum(F.when(route == 'unmapped', n).otherwise(zero)).alias(
            'n_unmapped'),
        F.sum(F.when(F.col('prev') == 'other-empty', n).otherwise(zero))
        .alias('n_empty'),
        F.sum(F.when(link_other, n).otherwise(zero)).alias('n_link_other'),
    ]
    return aggregations


def aggregate_month(spark, month, codec):
    """Write one month of the month x article x route table.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    month : str
        Month to aggregate from the current stage.
    codec : str
        Parquet compression codec.
    """
    stage = (spark.read.parquet(str(config.STAGE_DIR))
             .where(F.col('month') == month))
    routed = (stage.where(valid_row())
              .withColumn('route', F.expr(ROUTE_CASE_SQL)))
    table = routed.groupBy('curr').agg(*article_aggregations())
    out = config.ARTICLE_DIR / f'month={month}'
    (table.coalesce(config.FILES_PER_MONTH).write.mode('overwrite')
     .option('compression', codec).parquet(str(out)))


# ------------------------------------------------------------ main loop ----
def done_months():
    """Months whose stage, profile and aggregate all finished."""
    folder = config.TABLE_DIR / 'done'
    if not folder.exists():
        return []
    return sorted(path.stem for path in folder.glob('*.json'))


def conversion_log():
    """Per-month record of the conversion run.

    Returns
    -------
    pandas.DataFrame
        Rows, sizes and seconds for each finished month.
    """
    folder = config.TABLE_DIR / 'done'
    records = [json.loads(p.read_text()) for p in
               sorted(folder.glob('*.json'))] if folder.exists() else []
    return pd.DataFrame(records)


def _release_spark_files(spark):
    """Let Spark delete shuffle files of finished jobs before the next."""
    spark.catalog.clearCache()
    gc.collect()
    try:
        spark.sparkContext._jvm.System.gc()  # noqa: SLF001
    except Exception:  # noqa: BLE001 - best effort only
        pass


def convert_all(spark, files, codec='zstd', months_per_batch=None):
    """Stage, profile and aggregate every month that is not done yet.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    files : pandas.DataFrame
        Output of :func:`discover_files`.
    codec : str
        Parquet codec for the kept table.
    months_per_batch : int, optional
        Months read in parallel; defaults to ``config.MONTHS_PER_BATCH``.

    Returns
    -------
    pandas.DataFrame
        The conversion log (see :func:`conversion_log`).

    Raises
    ------
    RuntimeError
        If the work folder grows past ``config.STORAGE_STOP_GB``.
    """
    for folder in (config.ARTICLE_DIR, config.TABLE_DIR / 'done'):
        folder.mkdir(parents=True, exist_ok=True)
    batch_size = months_per_batch or config.MONTHS_PER_BATCH
    todo = files[~files['month'].isin(done_months())]
    if todo.empty:
        print(f'All {len(files)} months are already converted; '
              'nothing to do.')
        return conversion_log()
    free_gb = shutil.disk_usage(config.WORK_DIR).free / GIB
    print(f'{len(todo)} of {len(files)} months to convert, '
          f'{batch_size} at a time. Free disk where the work folder '
          f'lives: {free_gb:.0f} GB.')
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        clock = time.perf_counter()
        stage_batch(spark, batch)
        stage_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        save_tables(profile_stage(spark), batch['month'])
        profile_seconds = time.perf_counter() - clock
        for row in batch.itertuples(index=False):
            clock = time.perf_counter()
            aggregate_month(spark, row.month, codec)
            article_bytes = dir_size(config.ARTICLE_DIR
                                     / f'month={row.month}')
            stage_bytes = dir_size(config.STAGE_DIR / f'month={row.month}')
            work_gb = dir_size(config.WORK_DIR) / GIB
            record = {
                'month': row.month,
                'gz_mb': round(row.gz_bytes / 2 ** 20, 1),
                'stage_mb': round(stage_bytes / 2 ** 20, 1),
                'article_table_mb': round(article_bytes / 2 ** 20, 1),
                'stage_seconds': round(stage_seconds / len(batch), 1),
                'profile_seconds': round(profile_seconds / len(batch), 1),
                'aggregate_seconds': round(time.perf_counter() - clock, 1),
                'work_folder_gb_after': round(work_gb, 2),
            }
            path = config.TABLE_DIR / 'done' / f'{row.month}.json'
            path.write_text(json.dumps(record))
            if work_gb > config.STORAGE_STOP_GB:
                shutil.rmtree(config.STAGE_DIR, ignore_errors=True)
                raise RuntimeError(
                    f'Work folder reached {work_gb:.1f} GB, above the '
                    f'{config.STORAGE_STOP_GB} GB stop. Finished months are '
                    'kept; free space and rerun to continue.')
        shutil.rmtree(config.STAGE_DIR, ignore_errors=True)
        _release_spark_files(spark)
        log = conversion_log()
        done = len(log)
        per_month_gb = log['article_table_mb'].mean() / 1024
        projected = per_month_gb * len(files)
        elapsed = (time.perf_counter() - started) / 60
        print(f'  {", ".join(batch["month"])} done | {done}/{len(files)} '
              f'months | work folder {work_gb:.1f} GB | projected kept '
              f'table {projected:.1f} GB | {elapsed:.1f} min so far')
    return conversion_log()
