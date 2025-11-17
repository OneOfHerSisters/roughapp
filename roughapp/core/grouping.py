import pandas as pd
from typing import List, Tuple, Union, Any
from roughapp.core.utils import split_C_D


def ind_classes_table(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    C, D = split_C_D(df, decision_col)
    grouped = df.groupby(C, dropna=False)
    decision_in_groups = grouped[D]
    counts = decision_in_groups.value_counts()
    return counts.unstack(fill_value=0)


def mask_for_class_key(
    df: pd.DataFrame,
    decision_col: str,
    key: Union[tuple, Any],
) -> pd.Series:
    C, _ = split_C_D(df, decision_col)

    if not isinstance(key, tuple):
        key_vals = (key,)
    else:
        key_vals = key

    if len(key_vals) != len(C):
        raise ValueError(
            f"Key length {len(key_vals)} doesn't match number of conditional attrs {len(C)}"
        )

    mask = pd.Series(True, index=df.index)

    for col, val in zip(C, key_vals):
        mask = mask & df[col].eq(val)

    return mask
