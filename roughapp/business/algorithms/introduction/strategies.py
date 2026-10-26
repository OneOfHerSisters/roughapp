"""Introduction strategies as classes.

The validation and file-name-suffix logic that used to live in the datasets
router (_introduce_transform) moves next to the algorithms it belongs to.
"""
from __future__ import annotations

from roughapp.business.algorithms.base import IntroductionResult, IntroductionStrategy
from roughapp.business.exceptions import ValidationError
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.metrics import quality_of_classification
from roughapp.business.algorithms.introduction import drop_attributes as _drop
from roughapp.business.algorithms.introduction import duplicate_flip as _flip
from roughapp.business.algorithms.introduction.introduce_until_quality import run as _until_quality


class IntroduceUntilQualityStrategy(IntroductionStrategy):
    name = "until_quality"

    def __init__(self, target_gamma: float, random_state: int | None = None):
        if target_gamma is None:
            raise ValidationError("target_gamma is required for until_quality")
        self.target_gamma = target_gamma
        self.random_state = random_state

    def execute(self, table: DecisionTable) -> IntroductionResult:
        # Introduction only lowers γ (it adds conflicts). If the table is
        # already at or below the target, the loop would do nothing and we
        # would silently save an identical copy with a misleading name.
        gamma_current = quality_of_classification(table.frame, table.decision_column)
        if gamma_current <= self.target_gamma:
            raise ValidationError(
                f"Dataset γ ({gamma_current:.4f}) is already at or below the target "
                f"({self.target_gamma}) — introduction can only lower γ; choose a smaller target."
            )
        df_dirty, gamma_actual = _until_quality(
            table.frame, decision_col=table.decision_column,
            target_gamma=self.target_gamma, random_state=self.random_state,
        )
        return IntroductionResult(
            table=table.with_frame(df_dirty),
            gamma_actual=gamma_actual,
            suffix=f"g{self.target_gamma}",
        )


class DuplicateFlipStrategy(IntroductionStrategy):
    name = "duplicate_flip"

    def __init__(self, k: int = 1, random_state: int | None = None):
        self.k = k
        self.random_state = random_state if random_state is not None else 42

    def execute(self, table: DecisionTable) -> IntroductionResult:
        df_dirty = _flip.run(
            table.frame, decision_col=table.decision_column,
            k=self.k, random_state=self.random_state,
        )
        return IntroductionResult(
            table=table.with_frame(df_dirty),
            gamma_actual=quality_of_classification(df_dirty, table.decision_column),
            suffix=f"flip{self.k}",
        )


class DropAttributesStrategy(IntroductionStrategy):
    name = "drop_attributes"

    def __init__(self, attrs: list[str]):
        if not attrs:
            raise ValidationError("attrs (list of column names) is required for drop_attributes")
        self.attrs = list(attrs)

    def execute(self, table: DecisionTable) -> IntroductionResult:
        df_dirty = _drop.run(
            table.frame, decision_col=table.decision_column,
            mode="explicit", attrs=self.attrs,
        )
        return IntroductionResult(
            table=table.with_frame(df_dirty),
            gamma_actual=quality_of_classification(df_dirty, table.decision_column),
            suffix=f"drop_{'_'.join(self.attrs)}",
        )
