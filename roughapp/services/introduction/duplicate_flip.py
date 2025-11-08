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
    # Duplicate some rows and change their decision to create inconsistency
    if k < 1:
        raise ValueError("k must be at least 1")
    
    rng = np.random.default_rng(random_state)
    out = df.copy()
    uniq = out[decision_col].unique().tolist()
    
    if len(uniq) < 2 and new_value is None:
        raise ValueError("Need at least 2 decision values to flip")
    
    # Pick k random rows
    idx = rng.choice(out.index, size=min(k, len(out)), replace=False)
    dup = out.loc[idx].copy()
    
    # Function to get a different decision value
    def alt(v: Union[str, int]) -> Union[str, int]:
        if new_value is not None:
            return new_value
        # If 2 values, just flip
        if len(uniq) == 2:
            if v == uniq[0]:
                return uniq[1]
            else:
                return uniq[0]
        # More values - pick random different one
        choices = [u for u in uniq if u != v]
        return rng.choice(choices)
    
    # Change decision in duplicated rows
    dup[decision_col] = dup[decision_col].apply(alt)
    return pd.concat([out, dup], ignore_index=True)