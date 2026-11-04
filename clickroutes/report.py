"""Small helpers that show tables in the notebook with readable numbers."""
import numbers

import pandas as pd
from IPython.display import HTML, display


def _format_value(value, digits):
    """Format one cell: thousands separators, fixed decimals, yes/no."""
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, numbers.Integral):
        return f'{value:,}'
    if isinstance(value, numbers.Real):
        if pd.isna(value):
            return ''
        return f'{value:,.{digits}f}'
    return '' if value is None else str(value)


def formatted(table, digits=2):
    """Return a copy of ``table`` with every cell formatted as text.

    Parameters
    ----------
    table : pandas.DataFrame
        Any table.
    digits : int
        Decimals for non-integer numbers.

    Returns
    -------
    pandas.DataFrame
        Same shape, all strings.
    """
    out = table.copy()
    for column in out.columns:
        out[column] = [_format_value(v.item() if hasattr(v, 'item') else v,
                                     digits) for v in out[column]]
    return out


def show(table, digits=2):
    """Display a table without its index and with readable numbers.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to show.
    digits : int
        Decimals for non-integer numbers.
    """
    display(HTML(formatted(table, digits).to_html(index=False)))
