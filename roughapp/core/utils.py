import pandas as pd
from typing import Tuple, List, Dict


def split_C_D(df: pd.DataFrame, decision_col: str) -> Tuple[List[str], str]:
    if decision_col not in df.columns:
        raise ValueError(f"Decision column '{decision_col}' not found.")

    C = [c for c in df.columns if c != decision_col]
    if not C:
        raise ValueError("No conditional attributes (C) found.")

    return C, decision_col


def dataset_stats(df: pd.DataFrame) -> Dict[str, int]:
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
