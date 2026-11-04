"""Wall-clock timing of the slow steps, reported as a table at the end."""
import time
from contextlib import contextmanager

import pandas as pd

_RECORDS = []


@contextmanager
def timed(step):
    """Time the code inside a ``with`` block.

    Parameters
    ----------
    step : str
        Label shown in the timing table.

    Yields
    ------
    None
        Control returns to the caller; the duration is stored on exit.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        _RECORDS.append({'step': step,
                         'seconds': time.perf_counter() - start})


def timing_table(top=15):
    """List the slowest timed steps of this run.

    Parameters
    ----------
    top : int
        Number of rows to keep.

    Returns
    -------
    pandas.DataFrame
        Columns ``step``, ``minutes`` and ``share of timed run (%)``.
    """
    table = pd.DataFrame(_RECORDS, columns=['step', 'seconds'])
    if table.empty:
        return table
    table = table.groupby('step', as_index=False)['seconds'].sum()
    total = table['seconds'].sum()
    table['minutes'] = (table['seconds'] / 60).round(2)
    table['share of timed run (%)'] = (100 * table['seconds']
                                       / total).round(1)
    table = table.sort_values('seconds', ascending=False).head(top)
    return table.drop(columns='seconds').reset_index(drop=True)
