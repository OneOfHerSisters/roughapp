from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Literal, Optional

from roughapp.core.grouping import ind_classes_table, mask_for_class_key
from roughapp.core.metrics import alpha_rho_per_decision


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    tie_policy: Literal["keep", "random", "raise"] = "keep",
    random_state: Optional[int] = None,
    **_: Dict[str, Any],
) -> pd.DataFrame:
    """Qualitative removal (metoda jakościowa).

    The accuracy of approximation of each decision class is computed over the
    whole table; within every conflicting equivalence class the decision whose
    class has the highest accuracy is kept and the objects carrying the other,
    "less weighty" decisions are removed, so the class becomes consistent.
    When all decisions in a class tie on accuracy, the tie policy decides:
    leave the class unresolved (default), keep one at random, or raise.

    Przybyła-Kasperek, lecture notes "Systemy ekspertowe" (inconsistency
    removal, qualitative method).
    """
    rng = np.random.default_rng(random_state)

    alpha_df = alpha_rho_per_decision(df, decision_col)
    alpha_map: dict[Any, float] = dict(zip(alpha_df["decision"], alpha_df["alpha"]))

    table = ind_classes_table(df, decision_col)
    conflict_keys = table.index[(table > 0).sum(axis=1) > 1].tolist()

    rows_to_drop: list[int] = []

    for key in conflict_keys:
        counts = table.loc[key]
        decisions_in_class = counts[counts > 0].index.tolist()
        alphas = {d: alpha_map[d] for d in decisions_in_class}
        max_alpha = max(alphas.values())
        winners = [d for d, a in alphas.items() if a == max_alpha]
        losers = [d for d in decisions_in_class if d not in winners]

        if not losers:
            # All decisions share the same α — true tie
            if tie_policy == "keep":
                continue
            elif tie_policy == "random":
                idx = rng.integers(0, len(decisions_in_class))
                keep_d = decisions_in_class[idx]
                losers = [d for d in decisions_in_class if d != keep_d]
            elif tie_policy == "raise":
                raise ValueError(
                    f"Alpha tie in class {key!r}: {alphas}"
                )
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy!r}")

        class_mask = mask_for_class_key(df, decision_col, key)
        for loser_d in losers:
            drop_idx = df.index[class_mask & (df[decision_col] == loser_d)].tolist()
            rows_to_drop.extend(drop_idx)

    return df.drop(index=rows_to_drop).reset_index(drop=True)
