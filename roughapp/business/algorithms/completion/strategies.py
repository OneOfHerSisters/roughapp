"""Missing-value completion strategies as classes."""
from __future__ import annotations

from roughapp.business.algorithms.base import CompletionResult, CompletionStrategy
from roughapp.business.exceptions import ValidationError
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.metrics import quality_of_classification
from roughapp.core.missing import missing_summary
from roughapp.business.algorithms.completion import fill_values, remove_incomplete


def _require_missing(table: DecisionTable) -> None:
    if missing_summary(table.frame)["total_missing"] == 0:
        raise ValidationError("Dataset has no missing values to complete")


def _result(table: DecisionTable, df_clean, suffix: str) -> CompletionResult:
    if len(df_clean) == 0:
        raise ValidationError("Completion removed every row")
    return CompletionResult(
        table=table.with_frame(df_clean),
        gamma_actual=quality_of_classification(df_clean, table.decision_column),
        suffix=suffix,
    )


class RemoveIncompleteStrategy(CompletionStrategy):
    name = "remove"

    def execute(self, table: DecisionTable) -> CompletionResult:
        _require_missing(table)
        df_clean = remove_incomplete.run(table.frame, decision_col=table.decision_column)
        return _result(table, df_clean, "no_missing")


class FillValuesStrategy(CompletionStrategy):
    name = "fill"

    def __init__(self, by_class: bool = False):
        self.by_class = by_class

    def execute(self, table: DecisionTable) -> CompletionResult:
        _require_missing(table)
        df_clean = fill_values.run(
            table.frame, decision_col=table.decision_column, by_class=self.by_class
        )
        return _result(table, df_clean, "filled_by_class" if self.by_class else "filled")
