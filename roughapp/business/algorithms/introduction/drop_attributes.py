import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, Literal, Optional

from roughapp.core.utils import split_C_D


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    mode: Literal["random", "explicit"] = "explicit",
    attrs: Optional[Iterable[str]] = None,
    n: int = 1,
    random_state: int = 42,
    **_: Dict[str, Any],
) -> pd.DataFrame:
    """Drop conditional attributes to introduce inconsistency.

    mode="explicit": drop the columns listed in *attrs*.
    mode="random":   drop *n* randomly chosen conditional columns.

    Neither mode changes the number of rows, so the denominator of γ is
    unaffected — contrast with duplicate_flip, which appends rows.
    """
    C, _ = split_C_D(df, decision_col)

    if mode == "random":
        if len(C) <= n:
            raise ValueError(f"Can't drop {n} attributes, only {len(C)} available")
        rng = np.random.default_rng(random_state)
        attrs_to_drop = rng.choice(C, size=n, replace=False).tolist()
    else:
        attrs_to_drop = list(attrs) if attrs is not None else []
        for a in attrs_to_drop:
            if a == decision_col:
                raise ValueError("Can't drop decision column")
            if a not in df.columns:
                raise ValueError(f"Column '{a}' not found")

    remaining = [c for c in df.columns if c not in attrs_to_drop]
    if not [c for c in remaining if c != decision_col]:
        raise ValueError("Need at least one conditional attribute left")

    return df[remaining].copy()
