import pandas as pd
from typing import List, Tuple, Union, Any
from roughapp.core.utils import split_C_D


def ind_classes_table(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    # Build table of equivalence classes IND(C)
    # Rows = classes (unique combinations of conditional attributes)
    # Columns = decision values, values = counts
    C, D = split_C_D(df, decision_col)
    return df.groupby(C, dropna=False)[D].value_counts().unstack(fill_value=0)


def mask_for_class_key(
    df: pd.DataFrame, decision_col: str, key: Union[tuple, Any]
) -> pd.Series:
    # Get rows that match a specific equivalence class key
    C, _ = split_C_D(df, decision_col)
    key_vals = (key,) if not isinstance(key, tuple) else key
    # Check if each attribute matches
    conds = [df[c].eq(v) for c, v in zip(C, key_vals)]
    return pd.concat(conds, axis=1).all(axis=1)