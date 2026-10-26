from __future__ import annotations

import pandas as pd
from typing import Any, Dict

from roughapp.core.utils import split_C_D


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    **_: Dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, frozenset]]:
    """Encode the generalised decision of each class as an integer code (GD).

    Every equivalence class of IND(C) collapses to one row, and the set of
    decisions occurring in that class — its generalised decision — is replaced
    by a natural number, so that equal sets get equal codes and different sets
    get different codes.  Consistent classes (a singleton set) and conflicting
    ones each receive their own code, and the resulting table is consistent.

    Returns (gd_df, mapping) where mapping[code] = frozenset of the original
    decisions, so the encoding is fully reversible.

    Azad et al. 2013.
    """
    C, _ = split_C_D(df, decision_col)

    records: list[dict[str, Any]] = []
    for key, group in df.groupby(C, dropna=False, sort=False):
        cond_vals: dict[str, Any] = (
            {C[0]: key} if len(C) == 1 else dict(zip(C, key))
        )
        cond_vals[decision_col] = frozenset(group[decision_col])
        records.append(cond_vals)

    out = pd.DataFrame(records)[C + [decision_col]].reset_index(drop=True)

    seen: dict[frozenset, int] = {}
    for fs in out[decision_col]:
        if fs not in seen:
            seen[fs] = len(seen)

    mapping: dict[int, frozenset] = {v: k for k, v in seen.items()}
    out = out.copy()
    out[decision_col] = out[decision_col].map(seen)
    return out, mapping
