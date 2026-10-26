from __future__ import annotations

import pandas as pd

MISSING_TOKENS = {"?", "null", "missing", "nan", "none", ""}


def is_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    native = df.isna()
    text = df.apply(
        lambda col: col.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
    )
    return native | text


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[is_missing_mask(df)] = pd.NA
    return out


def missing_summary(df: pd.DataFrame) -> dict:
    mask = is_missing_mask(df)
    per_column = [
        {"column": col, "missing": int(mask[col].sum())}
        for col in df.columns
        if mask[col].any()
    ]
    total = int(mask.to_numpy().sum())
    rows_affected = int(mask.any(axis=1).sum())
    return {
        "total_missing": total,
        "rows_affected": rows_affected,
        "total_rows": int(len(df)),
        "per_column": per_column,
    }
