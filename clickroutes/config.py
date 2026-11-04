"""Every path, threshold and Spark setting used by ``main.ipynb``.

The notebook prints :func:`settings_table` as its settings table, so the
report and the code always agree. Paths can be overridden with environment
variables, which is only needed when testing away from Jojie.
"""
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- data ----
DATA_DIR = Path(os.environ.get(
    'CLICKSTREAM_DATA_DIR',
    '/mnt/data/public/wikipedia/clickstream/clickstream'))
FILE_GLOB = 'clickstream-enwiki-*.tsv.gz'
FILE_REGEX = r'^clickstream-enwiki-(\d{4}-\d{2})\.tsv\.gz$'
FIRST_MONTH = '2017-11'
LAST_MONTH = '2026-03'
EXPECTED_FILES = 77
EXPECTED_ROWS = 2_479_590_236

# ------------------------------------------------- private work folder ----
WORK_DIR = Path(os.environ.get(
    'CLICKSTREAM_WORK_DIR', str(Path.home() / 'clickstream_work')))
STAGE_DIR = WORK_DIR / 'stage'
ARTICLE_DIR = WORK_DIR / 'article_routes'
TABLE_DIR = WORK_DIR / 'tables'
RESULT_DIR = WORK_DIR / 'results'
SPARK_TMP_DIR = Path(os.environ.get(
    'CLICKSTREAM_SPARK_TMP', str(WORK_DIR / 'spark_tmp')))
FIGURE_DIR = Path('figures')

# ------------------------------------------------------- storage guard ----
STORAGE_QUOTA_GB = 50
STORAGE_STOP_GB = 40
GZ_TO_PARQUET_RATIO = 1.7
MONTHS_PER_BATCH = int(os.environ.get('CLICKSTREAM_MONTHS_PER_BATCH', 4))
FILES_PER_MONTH = 4

# --------------------------------------------------------------- spark ----
MAX_CORES = int(os.environ.get('CLICKSTREAM_MAX_CORES', 8))
MIN_DRIVER_MEMORY_GB = 4
MAX_DRIVER_MEMORY_GB = 16
SHUFFLE_PARTITIONS_PER_CORE = 4
PARQUET_CODEC = 'zstd'

# -------------------------------------------------------------- routes ----
ROUTES = ('search', 'link', 'unattributed', 'wikimedia', 'external')
ROUTE_NAMES = {
    'search': 'Search engine',
    'link': 'Article link',
    'unattributed': 'Unattributed',
    'wikimedia': 'Other Wikimedia',
    'external': 'External site',
}
LABEL_TO_ROUTE = {
    'other-search': 'search',
    'other-external': 'external',
    'other-internal': 'wikimedia',
    'other-empty': 'unattributed',
    'other-other': 'unattributed',
}
ATTRIBUTED_ROUTES = tuple(r for r in ROUTES if r != 'unattributed')
EXCLUDED_TITLE = 'Main_Page'

# ------------------------------------------------------------- cohorts ----
TOP_N = 10_000
COHORTS = (('1–100', 1, 100), ('101–1,000', 101, 1_000),
           ('1,001–10,000', 1_001, 10_000))
REST_COHORT = '10,001+ (long tail)'
COHORT_ORDER = tuple(c[0] for c in COHORTS) + (REST_COHORT,)
RANKING_BASE_YEAR = 2018

# ------------------------------------------------- truncation and change ----
MIN_PAIR_COUNTS = (50, 100)
THRESHOLDS = (('', None),) + tuple(
    (f'_ge{m}', m) for m in MIN_PAIR_COUNTS)
EARLY_YEAR = 2018
LATE_YEAR = 2025

# ----------------------------------------------------- reference events ----
EVENTS = {'2020-03': 'COVID-19', '2022-11': 'ChatGPT'}

# -------------------------------------------------------------- checks ----
ARTIFACT_Z = 3.5
ROBUST_RATIO = (0.5, 2.0)
NO_CHANGE_PP = 0.05
PERCENTILE_ACCURACY = 10_000
HEAVY_UNATTRIBUTED_SHARE = 0.9
HEAVY_MIN_VIEWS_PER_DAY = 1_000
DELETE_WORK_DIR_AT_END = False


def display_path(path):
    """Show a path with the home folder replaced by ``~``.

    Parameters
    ----------
    path : str or pathlib.Path
        Any path.

    Returns
    -------
    str
        The path without the user's home directory, so no username ends
        up in the notebook outputs.
    """
    text = str(path)
    home = str(Path.home())
    if text.startswith(home):
        return '~' + text[len(home):]
    return text


def settings_table():
    """List every analysis setting with the reason it was chosen.

    Returns
    -------
    pandas.DataFrame
        One row per setting with columns ``setting``, ``value`` and
        ``reason``.
    """
    cohort_text = ', '.join(c[0] for c in COHORTS) + f', {REST_COHORT}'
    rows = [
        ('Window', f'{FIRST_MONTH} to {LAST_MONTH}',
         'Every English monthly release on Jojie (game plan).'),
        ('Files used', 'clickstream-enwiki-*.tsv.gz',
         'English only; other languages and HTML files are skipped.'),
        ('Excluded title', EXCLUDED_TITLE,
         'Home-page settings are a suspected cause of empty referrers '
         '(Wikimedia Research, n.d.); rerun with it included in '
         'Section 8.'),
        ('Routes', ', '.join(ROUTE_NAMES[r] for r in ROUTES),
         'One CASE expression on prev; other-empty and other-other are '
         'pooled as unattributed.'),
        ('Per-day views', 'n / days in month',
         'Months of 28 to 31 days would otherwise add a false cycle.'),
        ('Cohorts', cohort_text,
         'Fixed once from whole-window views so articles cannot drift '
         'between groups; ranks found with ORDER BY ... LIMIT.'),
        ('Top-N for ranking', f'{TOP_N:,}',
         'Covers the three named cohorts; everything else is the tail.'),
        ('Alternative ranking', f'{RANKING_BASE_YEAR} views only',
         'Whole-window ranking favours articles that became popular '
         'late.'),
        ('Minimum pair counts', ', '.join(map(str, MIN_PAIR_COUNTS)),
         'A common floor in every month removes the effect of the '
         'release threshold on small pairs (truncation check).'),
        ('Matched periods', f'{EARLY_YEAR} vs {LATE_YEAR}, same months',
         'Same calendar months on both sides remove seasonality.'),
        ('Robust result', f'same sign, ratio {ROBUST_RATIO[0]} to '
         f'{ROBUST_RATIO[1]} of baseline',
         'Operational meaning of "direction and rough size survive".'),
        ('Release-artifact flag', f'|modified z| > {ARTIFACT_Z}',
         'Outlier rule of Iglewicz and Hoaglin (1993) on month-to-month '
         'changes.'),
        ('Percentile accuracy', f'{PERCENTILE_ACCURACY:,}',
         'percentile_approx relative error is about 1/accuracy; all sums '
         'are exact.'),
        ('Storage stop', f'{STORAGE_STOP_GB} GB of {STORAGE_QUOTA_GB} GB',
         'Leaves room below the quota for Spark spill files.'),
        ('Months per batch', f'{MONTHS_PER_BATCH}',
         'Gzip files are unsplittable, so several months are read in '
         'parallel, one task per file.'),
        ('Parquet codec', PARQUET_CODEC,
         'Smaller files than snappy for the kept table.'),
    ]
    return pd.DataFrame(rows, columns=['setting', 'value', 'reason'])
