from __future__ import annotations

import pandas as pd
from typing import Any, Dict

from roughapp.core.missing import is_missing_mask


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    **_: Dict[str, Any],
) -> pd.DataFrame:
    """Remove every object (row) that has at least one missing value.

    RSES: *Complete / Remove objects with missing values*.  The simplest,
    lossiest strategy — safe when missing cells are rare, wasteful otherwise.

    ``decision_col`` is accepted for signature symmetry with the other
    completion strategies but is not used here (a missing marker anywhere in
    the row drops the row).
    """
    mask = is_missing_mask(df)
    keep = ~mask.any(axis=1)
    return df[keep].reset_index(drop=True)
