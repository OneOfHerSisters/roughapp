"""DecisionTable — the domain object every algorithm works on.

A thin, validating wrapper around a pandas DataFrame.  It guarantees two
invariants that used to be re-checked (or forgotten) at every call site:
the table is not empty, and the decision column actually exists.  It also
carries the column-type map used for CSV round-tripping and tree encoding,
so callers stop threading (df, decision_col, column_types) triples around.
"""
from __future__ import annotations

import pandas as pd

from roughapp.business.exceptions import (
    DecisionColumnNotFoundError,
    EmptyDecisionTableError,
)
from roughapp.core.csv_table import ColumnTypes, default_column_types


class DecisionTable:
    def __init__(
        self,
        frame: pd.DataFrame,
        decision_column: str,
        column_types: ColumnTypes | None = None,
    ) -> None:
        if frame.empty:
            raise EmptyDecisionTableError()
        if decision_column not in frame.columns:
            raise DecisionColumnNotFoundError(decision_column)
        self._frame = frame.copy(deep=True)
        self._decision_column = decision_column
        self._column_types: ColumnTypes = (
            dict(column_types) if column_types is not None
            else default_column_types(frame.columns)
        )

    @property
    def decision_column(self) -> str:
        return self._decision_column

    @property
    def row_count(self) -> int:
        return len(self._frame)

    @property
    def column_count(self) -> int:
        return len(self._frame.columns)

    @property
    def condition_columns(self) -> list[str]:
        return [c for c in self._frame.columns if c != self._decision_column]

    @property
    def column_types(self) -> ColumnTypes:
        return dict(self._column_types)

    @property
    def frame(self) -> pd.DataFrame:
        """The internal frame, shared without copying.

        Read-only by convention: algorithms must copy before mutating
        (which the existing strategy functions already do).  Use
        :meth:`to_dataframe` when a caller needs a frame it may change.
        """
        return self._frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)

    def gamma(self) -> float:
        """Quality of classification γ of this table."""
        from roughapp.core.metrics import quality_of_classification
        return quality_of_classification(self._frame, self._decision_column)

    def alpha(self) -> float:
        """Accuracy of approximation α of this table."""
        from roughapp.core.metrics import accuracy_of_classification
        return accuracy_of_classification(self._frame, self._decision_column)

    def with_frame(self, frame: pd.DataFrame) -> "DecisionTable":
        """A new table with the same decision column and (pruned) types."""
        types = {c: t for c, t in self._column_types.items() if c in frame.columns}
        for c in frame.columns:
            types.setdefault(c, "c")
        return DecisionTable(frame, self._decision_column, types)
