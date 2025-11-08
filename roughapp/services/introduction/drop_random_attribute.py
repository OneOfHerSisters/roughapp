import numpy as np
import pandas as pd
from typing import Dict, Any

from roughapp.core.utils import split_C_D


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    n: int = 1,
    random_state: int = 42,
    **_: Dict[str, Any]
) -> pd.DataFrame:
    # Remove random attributes to introduce inconsistency
    # When we remove attributes, some classes that were different become the same
    C, _ = split_C_D(df, decision_col)
    
    if len(C) <= n:
        raise ValueError(f"Can't drop {n} attributes, only {len(C)} available")
    
    rng = np.random.default_rng(random_state)
    # Choose n random attributes to remove
    attrs_to_drop = rng.choice(C, size=n, replace=False).tolist()
    
    # Return dataframe without those columns
    remaining = [c for c in df.columns if c not in attrs_to_drop]
    C_left = [c for c in remaining if c != decision_col]
    
    if not C_left:
        raise ValueError("Can't drop all conditional attributes")
    
    return df[remaining].copy()

