"""Resolution algorithms.

Each pure-function module exposes ``run(df, *, decision_col, ...)`` and is
individually unit-tested; strategies.py wraps them into the
ResolutionStrategy classes used by the runner.
"""
from roughapp.business.algorithms.resolution import (
    generalized_decision,
    most_common_decision,
    rough_approximation,
)
from roughapp.business.algorithms.resolution.strategies import (
    GeneralizedDecisionStrategy,
    MostCommonDecisionStrategy,
    RoughApproximationStrategy,
)

__all__ = [
    "most_common_decision",
    "rough_approximation",
    "generalized_decision",
    "MostCommonDecisionStrategy",
    "RoughApproximationStrategy",
    "GeneralizedDecisionStrategy",
]
