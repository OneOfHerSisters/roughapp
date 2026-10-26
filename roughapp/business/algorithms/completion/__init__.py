"""Missing-value completion algorithms (pure functions + strategy classes)."""
from roughapp.business.algorithms.completion import fill_values, remove_incomplete
from roughapp.business.algorithms.completion.strategies import (
    FillValuesStrategy,
    RemoveIncompleteStrategy,
)

__all__ = [
    "remove_incomplete",
    "fill_values",
    "RemoveIncompleteStrategy",
    "FillValuesStrategy",
]
