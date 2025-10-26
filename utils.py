from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Plik CSV jest pusty.")
    return df

def split_attributes(df: pd.DataFrame, decision_col: str) -> Tuple[List[str], str]:
    if decision_col not in df.columns:
        raise ValueError(f"Kolumna decyzji '{decision_col}' nie istnieje.")
    C = [c for c in df.columns if c != decision_col]
    if not C:
        raise ValueError("Brak atrybutów warunkowych (C).")
    return C, decision_col

def dataset_stats(df: pd.DataFrame) -> Dict[str, int]:
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

def find_inconsistencies(df: pd.DataFrame, decision_col: str) -> Dict[str, object]:
    C, D = split_attributes(df, decision_col)
    # grupowanie po C; jeśli w grupie jest >1 unikalna decyzja → klasa niespójna
    grp = df.groupby(C, dropna=False)[D].nunique()
    inconsistent_classes = grp[grp > 1]
    n_classes = int(inconsistent_classes.shape[0])

    # policz liczbę wierszy należących do klas niespójnych
    nu = df.groupby(C)[D].transform('nunique')
    inconsistent_rows = int((nu > 1).sum())

    # przykłady (max 5 klas)
    examples = []
    for key in inconsistent_classes.index[:5]:
        # key może być pojedynczą wartością lub krotką
        key_vals = (key,) if not isinstance(key, tuple) else key
        mask = np.logical_and.reduce([df[c].values == v for c, v in zip(C, key_vals)])
        examples.append(df.loc[mask, C + [D]].copy())

    return {
        "inconsistent_class_count": n_classes,
        "inconsistent_total_rows": inconsistent_rows,
        "examples": examples
    }

def introduce_inconsistency(
    df: pd.DataFrame, decision_col: str, k: int = 1,
    new_value: str | int | None = None, random_state: int = 42
) -> pd.DataFrame:
    """
    Dodaje k konfliktów: duplikuje wiersze i zmienia decyzję na inną.
    """
    rng = np.random.default_rng(random_state)
    C, D = split_attributes(df, decision_col)
    out = df.copy()
    uniq = out[D].unique().tolist()
    if len(uniq) < 2 and new_value is None:
        raise ValueError("Nie można wprowadzić niespójności: decyzja ma tylko jedną wartość.")
    idx = rng.choice(out.index, size=min(k, len(out)), replace=False)
    dup = out.loc[idx].copy()

    def alt(v):
        if new_value is not None:
            return new_value
        if len(uniq) == 2:
            return uniq[1] if v == uniq[0] else uniq[0]
        # wieloklasowo: wylosuj inną etykietę
        choices = [u for u in uniq if u != v]
        return rng.choice(choices)

    dup[D] = dup[D].apply(alt)
    return pd.concat([out, dup], ignore_index=True)