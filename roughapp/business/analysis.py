from __future__ import annotations

from typing import Dict, List, Any

import pandas as pd

from roughapp.core.grouping import ind_classes_table, mask_for_class_key
from roughapp.core.metrics import alpha_rho_per_decision, accuracy_of_classification, quality_of_classification


def find_inconsistencies(df: pd.DataFrame, decision_col: str) -> Dict[str, Any]:
    table = ind_classes_table(df, decision_col)

    inconsistent_classes = table[(table > 0).sum(axis=1) > 1]
    n_classes = len(inconsistent_classes)

    class_sizes = table.sum(axis=1)
    inconsistent_rows = int(class_sizes[inconsistent_classes.index].sum())

    examples: List[pd.DataFrame] = []
    for key in list(inconsistent_classes.index)[:5]:
        mask = mask_for_class_key(df, decision_col, key)
        examples.append(df.loc[mask].copy())

    metrics = alpha_rho_per_decision(df, decision_col)

    # Exact duplicates: whole rows repeated (same conditions AND same decision).
    # These are redundancy, not conflict — a class with the same decision twice
    # is still consistent — but they bias frequency-based learners, so we report
    # them so the user can remove them.
    duplicate_rows = int(df.duplicated().sum())

    return {
        "inconsistent_class_count": n_classes,
        "inconsistent_total_rows": inconsistent_rows,
        "duplicate_rows": duplicate_rows,
        "accuracy": accuracy_of_classification(df, decision_col),
        "quality": quality_of_classification(df, decision_col),
        "per_decision": metrics.to_dict(orient="records"),
        "examples": examples,
    }
