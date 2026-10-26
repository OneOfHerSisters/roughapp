from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Literal, Optional

from roughapp.core.grouping import ind_classes_table, mask_for_class_key


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    tie_policy: Literal["min", "keep", "first", "random", "raise"] = "min",
    random_state: Optional[int] = None,
    **_: Dict[str, Any],
) -> tuple[pd.DataFrame, List[Any]]:
    rng = np.random.default_rng(random_state)
    table = ind_classes_table(df, decision_col)
    conflict_keys = table.index[(table > 0).sum(axis=1) > 1].tolist()

    out = df.copy()
    tied: List[Any] = []

    for key in conflict_keys:
        counts = table.loc[key]
        counts = counts[counts > 0]
        max_count = counts.max()
        candidates = counts[counts == max_count].index.tolist()

        if len(candidates) == 1:
            winner = candidates[0]
        elif tie_policy == "min":
            winner = min(candidates)
        elif tie_policy == "keep":
            tied.append(key)
            continue
        elif tie_policy == "first":
            winner = candidates[0]
        elif tie_policy == "random":
            idx = rng.integers(0, len(candidates))
            winner = candidates[idx]
        elif tie_policy == "raise":
            raise ValueError(
                f"Tie in equivalence class {key!r}: decisions {candidates} "
                f"all have count {max_count}"
            )
        else:
            raise ValueError(f"Unknown tie_policy: {tie_policy!r}")

        mask = mask_for_class_key(out, decision_col, key)
        out.loc[mask, decision_col] = winner

    out = out.drop_duplicates().reset_index(drop=True)
    return out, tied
