import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    k: int = 1,
    new_value: Optional[Union[str, int]] = None,
    random_state: int = 42,
    **_: Dict[str, Any]
) -> pd.DataFrame:

    if k < 1:
        raise ValueError("k must be at least 1")

    rng = np.random.default_rng(random_state)
    out = df.copy()
    uniq = out[decision_col].unique().tolist()

    if len(uniq) < 2 and new_value is None:
        raise ValueError("Need at least 2 decision values to flip")

    idx = rng.choice(out.index, size=min(k, len(out)), replace=False)
    dup = out.loc[idx].copy()

    def alt(v: Union[str, int]) -> Union[str, int]:
        # new_value must differ from v, otherwise the duplicate has the same
        # decision as the original and no conflict is created.
        if new_value is not None and new_value != v:
            return new_value
        choices = [u for u in uniq if u != v]
        if not choices:
            raise ValueError(f"No alternative decision value for '{v}'")
        return rng.choice(choices)

    dup[decision_col] = dup[decision_col].apply(alt)
    return pd.concat([out, dup], ignore_index=True)
