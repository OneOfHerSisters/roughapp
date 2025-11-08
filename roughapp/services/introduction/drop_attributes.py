import pandas as pd
from typing import Iterable, Dict, Any, Optional

def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    attrs: Optional[Iterable[str]] = None,
    **_: Dict[str, Any]
) -> pd.DataFrame:
    # Remove specified attributes to create inconsistency
    if attrs is None:
        attrs = []
    attrs = list(attrs)
    
    # Check that we're not dropping decision column
    for a in attrs:
        if a == decision_col:
            raise ValueError("Can't drop decision column")
        if a not in df.columns:
            raise ValueError(f"Column '{a}' not found")
    
    # Keep only columns we want
    remaining = [c for c in df.columns if c not in attrs]
    C_left = [c for c in remaining if c != decision_col]
    
    if not C_left:
        raise ValueError("Need at least one conditional attribute left")
    
    return df[remaining].copy()