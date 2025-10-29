import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df


def split_attributes(df: pd.DataFrame, decision_col: str) -> Tuple[List[str], str]:
    if decision_col not in df.columns:
        raise ValueError(f"The decision column '{decision_col}' does not exist.")
    C = [c for c in df.columns if c != decision_col]
    if not C:
        raise ValueError("No conditional attributes (C) found.")
    return C, decision_col


def dataset_stats(df: pd.DataFrame) -> Dict[str, int]:
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}


def find_inconsistencies(df: pd.DataFrame, decision_col: str) -> Dict[str, object]:
    C, D = split_attributes(df, decision_col)

    # if a group has >1 unique decision then inconsistent class
    grp = df.groupby(C, dropna=False)[D].nunique()
    inconsistent_classes = grp[grp > 1]
    n_classes = int(inconsistent_classes.shape[0])

    nu = df.groupby(C)[D].transform('nunique')
    inconsistent_rows = int((nu > 1).sum())

    examples = []
    for key in inconsistent_classes.index[:5]:
        key_vals = (key,) if not isinstance(key, tuple) else key
        mask = np.logical_and.reduce([df[c].values == v for c, v in zip(C, key_vals)])
        examples.append(df.loc[mask, C + [D]].copy())

    return {
        "inconsistent_class_count": n_classes,
        "inconsistent_total_rows": inconsistent_rows,
        "examples": examples
    }


def introduce_inconsistency(
    df: pd.DataFrame,
    decision_col: str,
    k: int = 1,
    new_value: str | int | None = None,
    random_state: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)
    C, D = split_attributes(df, decision_col)
    out = df.copy()
    uniq = out[D].unique().tolist()

    if len(uniq) < 2 and new_value is None:
        raise ValueError("Cannot introduce inconsistency: only one decision value found.")

    # Randomly select k rows to duplicate
    idx = rng.choice(out.index, size=min(k, len(out)), replace=False)
    dup = out.loc[idx].copy()

    def alt(v):
        # selects an alternative decision value different from v
        if new_value is not None:
            return new_value
        if len(uniq) == 2:
            return uniq[1] if v == uniq[0] else uniq[0]
        #randomly choose a different value in multi class
        choices = [u for u in uniq if u != v]
        return rng.choice(choices)

    dup[D] = dup[D].apply(alt)

    return pd.concat([out, dup], ignore_index=True)