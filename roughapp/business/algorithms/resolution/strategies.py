from __future__ import annotations

import pandas as pd

from roughapp.business.algorithms.base import (
    ExperimentConfiguration,
    ResolutionResult,
    ResolutionStrategy,
)
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.metrics import (
    accuracy_of_classification,
    alpha_rho_per_decision,
    quality_of_classification,
)
from roughapp.business.algorithms.resolution import (
    generalized_decision,
    most_common_decision,
    rough_approximation,
)

_MAX_DETAIL_CLASSES = 10


def _after_metrics(df_after: pd.DataFrame, decision_col: str) -> dict:
    return {
        "gamma": quality_of_classification(df_after, decision_col),
        "alpha": accuracy_of_classification(df_after, decision_col),
        "rows": len(df_after),
    }


class MostCommonDecisionStrategy(ResolutionStrategy):
    name = "mcd"

    def __init__(self, tie_policy: str = "min", random_state: int | None = None):
        self.tie_policy = tie_policy
        self.random_state = random_state

    @classmethod
    def from_config(cls, config: ExperimentConfiguration) -> "MostCommonDecisionStrategy":
        return cls(tie_policy=config.tie_policy, random_state=config.random_state)

    def execute(self, table: DecisionTable) -> ResolutionResult:
        df_before = table.frame
        decision_col = table.decision_column
        result = most_common_decision.run(
            df_before, decision_col=decision_col,
            tie_policy=self.tie_policy, random_state=self.random_state,
        )
        df_after, tied_keys = result if isinstance(result, tuple) else (result, [])

        cond_cols = table.condition_columns
        tied_set = set(tied_keys)
        conflict_classes = []
        n_conflict_classes = 0
        for key, grp in df_before.groupby(cond_cols, sort=False):
            if grp[decision_col].nunique() <= 1:
                continue
            n_conflict_classes += 1
            if len(conflict_classes) >= _MAX_DETAIL_CLASSES:
                continue  # keep counting classes, stop collecting examples
            key_t = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=df_after.index)
            for col, val in zip(cond_cols, key_t):
                mask &= df_after[col] == val
            conflict_classes.append({
                "key": {col: str(v) for col, v in zip(cond_cols, key_t)},
                "counts": {str(k): int(v) for k, v in grp[decision_col].value_counts().items()},
                "tied": key_t in tied_set or key in tied_set,
                "chosen": [str(v) for v in df_after.loc[mask, decision_col].unique().tolist()],
            })

        return ResolutionResult(
            tables=(df_after,),
            metrics=_after_metrics(df_after, decision_col),
            details={
                "tied_count": len(tied_keys),
                "conflict_classes": conflict_classes,
                "more": n_conflict_classes > len(conflict_classes),
            },
        )


class RoughApproximationStrategy(ResolutionStrategy):
    name = "rough-approximation"

    def __init__(self, tie_policy: str = "keep", random_state: int | None = None):
        self.tie_policy = tie_policy
        self.random_state = random_state

    @classmethod
    def from_config(cls, config: ExperimentConfiguration) -> "RoughApproximationStrategy":
        return cls(tie_policy=config.qr_tie_policy, random_state=config.random_state)

    def execute(self, table: DecisionTable) -> ResolutionResult:
        df_before = table.frame
        decision_col = table.decision_column
        result = rough_approximation.run(
            df_before, decision_col=decision_col,
            tie_policy=self.tie_policy, random_state=self.random_state,
        )
        df_after = result[0] if isinstance(result, tuple) else result

        per_decision = alpha_rho_per_decision(df_before, decision_col)
        alpha_map = dict(zip(per_decision["decision"], per_decision["alpha"]))
        conflict_classes = []
        n_conflict_classes = 0
        for key, grp in df_before.groupby(table.condition_columns, sort=False):
            if grp[decision_col].nunique() <= 1:
                continue
            n_conflict_classes += 1
            if len(conflict_classes) >= _MAX_DETAIL_CLASSES:
                continue
            key_t = key if isinstance(key, tuple) else (key,)
            dec_counts = grp[decision_col].value_counts()
            alphas = {d: float(alpha_map.get(d, 0)) for d in dec_counts.index}
            max_a = max(alphas.values())
            is_tie = min(alphas.values()) == max_a
            conflict_classes.append({
                "key": {col: str(v) for col, v in zip(table.condition_columns, key_t)},
                "decisions": [
                    {"value": str(d), "count": int(dec_counts[d]), "alpha": round(alphas[d], 4)}
                    for d in dec_counts.index
                ],
                # kept: the decision(s) with the highest accuracy; the rest go
                "removed": [str(d) for d, a in alphas.items() if a < max_a],
                "tie": is_tie,
            })

        return ResolutionResult(
            tables=(df_after,),
            metrics=_after_metrics(df_after, decision_col),
            details={
                "conflict_classes": conflict_classes,
                "more": n_conflict_classes > len(conflict_classes),
            },
        )


class GeneralizedDecisionStrategy(ResolutionStrategy):
    name = "gd"

    @classmethod
    def from_config(cls, config: ExperimentConfiguration) -> "GeneralizedDecisionStrategy":
        return cls()

    def execute(self, table: DecisionTable) -> ResolutionResult:
        df_after, mapping = generalized_decision.run(
            table.frame, decision_col=table.decision_column
        )
        n_codes = len(mapping)
        n_conflict = sum(1 for v in mapping.values() if len(v) > 1)
        return ResolutionResult(
            tables=(df_after,),
            metrics={
                "gamma": 1.0, "alpha": 1.0, "rows": len(df_after),
                "n_gd_codes": n_codes, "n_gd_conflict_codes": n_conflict,
            },
            details={
                "total_codes": n_codes,
                "conflict_codes": n_conflict,
                "full_mapping": [
                    {"code": k, "decisions": sorted(list(v)), "conflict": len(v) > 1}
                    for k, v in sorted(mapping.items())
                ],
            },
        )
