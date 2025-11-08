from __future__ import annotations

from typing import Dict, List, Any

import pandas as pd

from roughapp.core.grouping import ind_classes_table, mask_for_class_key


def find_inconsistencies(
    df: pd.DataFrame, decision_col: str
) -> Dict[str, Any]:
    # Find inconsistent classes (same conditional values, different decisions)
    table = ind_classes_table(df, decision_col)
    
    # Class is inconsistent if it has more than one decision value
    inconsistent_classes = table[(table > 0).sum(axis=1) > 1]
    n_classes = len(inconsistent_classes)
    
    # Count rows in inconsistent classes
    class_sizes = table.sum(axis=1)
    inconsistent_rows = int(class_sizes[inconsistent_classes.index].sum())

    # Get some examples (up to 5)
    examples: List[pd.DataFrame] = []
    for key in list(inconsistent_classes.index)[:5]:
        mask = mask_for_class_key(df, decision_col, key)
        examples.append(df.loc[mask].copy())

    return {
        "inconsistent_class_count": n_classes,
        "inconsistent_total_rows": inconsistent_rows,
        "examples": examples,
    }