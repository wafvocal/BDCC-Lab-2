"""Start a local Spark session sized to what the Jojie container allows.

Jojie is shared, so the session reads the container's CPU and memory
limits (cgroups or the ``CPU_LIMIT``/``MEM_LIMIT`` variables that JupyterHub
sets) and stays below them instead of taking the whole machine.
"""
import os
import platform
import shutil
import tempfile
import uuid
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession

from . import config


def _read_number(path):
    """Read one integer from a cgroup file, or None if absent/unlimited."""
    try:
        text = Path(path).read_text().split()[0]
    except (OSError, IndexError):
        return None
    if text == 'max':
        return None
    try:
        return int(text)
    except ValueError:
        return None


def available_cores():
    """Count the CPU cores this container may use.

    Returns
    -------
    int
        The smaller of the scheduler affinity and any CPU quota.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    quota = None
    if os.environ.get('CPU_LIMIT'):
        quota = float(os.environ['CPU_LIMIT'])
    else:
        try:
            parts = Path('/sys/fs/cgroup/cpu.max').read_text().split()
            if parts[0] != 'max':
                quota = int(parts[0]) / int(parts[1])
        except (OSError, IndexError, ValueError):
            limit = _read_number('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
            period = _read_number('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
            if limit and period and limit > 0:
                quota = limit / period
    if quota:
        cores = min(cores, max(1, int(quota)))
    return max(1, cores)


def available_memory_gb():
    """Return the memory limit of this container in GiB.

    Returns
    -------
    float
        The JupyterHub or cgroup limit if one is set, else physical RAM.
    """
    if os.environ.get('MEM_LIMIT', '').isdigit():
        return int(os.environ['MEM_LIMIT']) / 2 ** 30
    for path in ('/sys/fs/cgroup/memory.max',
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        value = _read_number(path)
        if value and value < 2 ** 50:
            return value / 2 ** 30
    try:
        pages = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        return pages / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 8.0


def choose_spark_tmp():
    """Pick the folder for Spark shuffle and spill files.

    Returns
    -------
    pathlib.Path
        ``/tmp/<random>`` when /tmp has at least 30 GB free (fast local
        disk that does not count against the home quota), else
        ``config.SPARK_TMP_DIR`` inside the work folder.
    """
    if 'CLICKSTREAM_SPARK_TMP' not in os.environ:
        try:
            if shutil.disk_usage('/tmp').free > 30 * 2 ** 30:
                return Path(tempfile.mkdtemp(prefix='spark-lab2-'))
        except OSError:
            pass
    config.SPARK_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return config.SPARK_TMP_DIR


LOG4J2 = """rootLogger.level = error
rootLogger.appenderRef.stderr.ref = console
appender.console.type = Console
appender.console.name = console
appender.console.target = SYSTEM_ERR
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
logger.repl.name = org.apache.spark.repl.Main
logger.repl.level = error
logger.gateway.name = org.apache.spark.api.python.PythonGatewayServer
logger.gateway.level = error
"""
LOG4J1 = """log4j.rootCategory=ERROR, console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d %p %c{1}: %m%n
log4j.logger.org.apache.spark.repl.Main=ERROR
log4j.logger.org.apache.spark.api.python.PythonGatewayServer=ERROR
"""


def quiet_logging_options(folder):
    """Java options that show only Spark errors in the notebook.

    Parameters
    ----------
    folder : pathlib.Path
        Folder where the two logging configuration files are written.

    Returns
    -------
    str
        Options for ``spark.driver.extraJavaOptions``; they cover log4j 2
        (Spark 3.3 and later) and log4j 1 (earlier Spark 3 releases).
    """
    new_style = folder / 'log4j2.properties'
    old_style = folder / 'log4j.properties'
    new_style.write_text(LOG4J2)
    old_style.write_text(LOG4J1)
    return (f'-Dlog4j.configurationFile={new_style} '
            f'-Dlog4j.configuration=file:{old_style}')


def start_spark(app_name='bdcc-lab2-clickstream'):
    """Create the Spark session used by the whole notebook.

    Parameters
    ----------
    app_name : str
        Name shown in the Spark UI.

    Returns
    -------
    spark : pyspark.sql.SparkSession
        The running session.
    settings : dict
        Cores, driver memory, shuffle partitions, codec and spill folder,
        printed in the computing-environment table.
    """
    cores = min(available_cores(), config.MAX_CORES)
    memory = available_memory_gb()
    driver_gb = int(np.clip(memory * 0.5, config.MIN_DRIVER_MEMORY_GB,
                            config.MAX_DRIVER_MEMORY_GB))
    partitions = cores * config.SHUFFLE_PARTITIONS_PER_CORE
    tmp_dir = choose_spark_tmp()
    spark = (SparkSession.builder
             .master(f'local[{cores}]')
             .appName(app_name)
             .config('spark.driver.memory', f'{driver_gb}g')
             .config('spark.driver.maxResultSize', '2g')
             .config('spark.local.dir', str(tmp_dir))
             .config('spark.sql.shuffle.partitions', str(partitions))
             .config('spark.sql.adaptive.enabled', 'true')
             .config('spark.sql.adaptive.coalescePartitions.enabled',
                     'true')
             .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
             .config('spark.sql.sources.partitionColumnTypeInference'
                     '.enabled', 'false')
             .config('spark.sql.session.timeZone', 'UTC')
             .config('spark.sql.execution.arrow.pyspark.enabled', 'false')
             .config('spark.cleaner.periodicGC.interval', '2min')
             .config('spark.ui.showConsoleProgress', 'false')
             .config('spark.driver.extraJavaOptions',
                     quiet_logging_options(tmp_dir))
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    settings = {
        'cores': cores,
        'container_memory_gb': round(memory, 1),
        'driver_memory_gb': driver_gb,
        'shuffle_partitions': partitions,
        'parquet_codec': parquet_codec(spark),
        'spill_folder': config.display_path(tmp_dir),
        'spill_path': str(tmp_dir),
    }
    return spark, settings


def parquet_codec(spark):
    """Use zstd for the kept table if this Spark build can write it.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.

    Returns
    -------
    str
        ``'zstd'`` when a test write succeeds, otherwise ``'snappy'``.
    """
    probe = Path(tempfile.gettempdir()) / f'codec-probe-{uuid.uuid4().hex}'
    try:
        (spark.range(3).write.option('compression', config.PARQUET_CODEC)
         .parquet(str(probe)))
        return config.PARQUET_CODEC
    except Exception:  # noqa: BLE001 - any failure means fall back
        return 'snappy'
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def environment_table(spark, settings):
    """Versions and resources used for the run.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Running session.
    settings : dict
        Output of :func:`start_spark`.

    Returns
    -------
    pandas.DataFrame
        One row per item with columns ``item`` and ``value``.
    """
    jvm = spark.sparkContext._jvm  # noqa: SLF001 - read-only version
    java = jvm.java.lang.System.getProperty('java.version')
    rows = [
        ('Python', platform.python_version()),
        ('PySpark', pyspark.__version__),
        ('Spark', spark.version),
        ('Java', java),
        ('pandas', pd.__version__),
        ('NumPy', np.__version__),
        ('Matplotlib', matplotlib.__version__),
        ('Spark master', spark.sparkContext.master),
        ('Cores used', settings['cores']),
        ('Container memory (GiB)', settings['container_memory_gb']),
        ('Driver memory (GiB)', settings['driver_memory_gb']),
        ('Shuffle partitions', settings['shuffle_partitions']),
        ('Adaptive query execution', 'on'),
        ('Parquet codec (kept table)', settings['parquet_codec']),
        ('Spill folder', settings['spill_folder']),
    ]
    return pd.DataFrame(rows, columns=['item', 'value']).astype(str)
