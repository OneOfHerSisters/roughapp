from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import Any, Dict

from roughapp.core.missing import normalize_missing
from roughapp.core.utils import split_C_D


def _coerce_numeric(col: pd.Series) -> pd.Series:
    """Convert a column to numeric if every non-missing value is numeric.

    Textual missing markers force columns to object dtype; this recovers the
    original numeric type so numeric columns get mean-filled, not mode-filled.
    """
    converted = pd.to_numeric(col, errors="coerce")
    # Only accept the conversion if it didn't turn real values into NaN.
    if converted.isna().sum() == col.isna().sum():
        return converted
    return col


def _fill_series(col: pd.Series) -> pd.Series:
    """Fill one column: mean for numeric, most-common for nominal."""
    if col.isna().all():
        return col
    if is_numeric_dtype(col):
        return col.fillna(col.mean())
    mode = col.mode(dropna=True)
    if mode.empty:
        return col
    return col.fillna(mode.iloc[0])


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    by_class: bool = False,
    **_: Dict[str, Any],
) -> pd.DataFrame:
    """Fill missing values with the most common (nominal) or mean (numeric).

    RSES: *Complete with most common or mean value*.

    by_class=False: statistics computed over the whole column.
    by_class=True:  statistics computed within each decision class
                    (*... w.r.t. decision class*), which better preserves the
                    relationship between conditions and decision.

    The decision column itself is never filled — a row with a missing decision
    is left as-is (completion targets conditional attributes only).
    """
    C, _ = split_C_D(df, decision_col)
    out = df.copy()
    # Only conditional attributes are completed; the decision column is left
    # exactly as-is (a row with a missing decision is not touched here).
    out[C] = normalize_missing(df[C])
    # Recover numeric dtypes lost when textual markers forced object columns.
    out[C] = out[C].apply(_coerce_numeric)

    if by_class:
        out[C] = (
            out.groupby(decision_col, group_keys=False)[C]
            .apply(lambda g: g.apply(_fill_series))
        )
        # Classes that were entirely missing in some column fall back to global.
        out[C] = out[C].apply(_fill_series)
    else:
        out[C] = out[C].apply(_fill_series)

    return out.reset_index(drop=True)
