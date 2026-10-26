"""Abstract interfaces and result objects for the algorithm families.

Each family gets its own interface and its own result type — a resolution
strategy, an introduction method, a completion method, and a tree builder
answer different questions, so forcing them through one universal
``Algorithm`` interface would only blur that.

Polymorphism happens at the ``execute``/``build`` call sites: the runner
and the services work with the base types and never branch on concrete
strategy names.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.models.decision_tree import DecisionTree


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of one resolution strategy on one table.

    ``tables`` are ready-to-save output frames — one per strategy.
    ``metrics`` holds the numeric after-measures under their canonical
    metric names; ``details`` is the strategy-specific JSON payload shown
    in the UI.
    """
    tables: tuple[pd.DataFrame, ...]
    metrics: dict[str, float | int | None]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntroductionResult:
    table: DecisionTable
    gamma_actual: float
    suffix: str  # used in the derived dataset's file name, e.g. "g0.5"


@dataclass(frozen=True)
class CompletionResult:
    table: DecisionTable
    gamma_actual: float
    suffix: str


@dataclass(frozen=True)
class TreeBuildResult:
    depth: int
    avg_depth: float
    leaves: int
    internal_nodes: int
    text: str

    @property
    def nodes(self) -> int:
        return self.leaves + self.internal_nodes

    def stats(self) -> dict[str, float | int]:
        return {"depth": self.depth, "avg_depth": self.avg_depth, "nodes": self.nodes}


class ResolutionStrategy(ABC):
    """One way of removing inconsistency from a decision table.

    Parameters (tie policies, seeds, limits) are constructor arguments, so
    the runner can execute any strategy through the same one-argument call.
    """
    name: str

    @abstractmethod
    def execute(self, table: DecisionTable) -> ResolutionResult: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: "ExperimentConfiguration") -> "ResolutionStrategy":
        """Pick this strategy's own parameters out of the shared config."""


class IntroductionStrategy(ABC):
    name: str

    @abstractmethod
    def execute(self, table: DecisionTable) -> IntroductionResult: ...


class CompletionStrategy(ABC):
    name: str

    @abstractmethod
    def execute(self, table: DecisionTable) -> CompletionResult: ...


class TreeBuilder(ABC):
    name: str

    @abstractmethod
    def build(self, table: DecisionTable, max_depth: int | None = None) -> TreeBuildResult:
        """Summary statistics and text rendering of the induced tree."""

    @abstractmethod
    def build_tree(self, table: DecisionTable, max_depth: int | None = None) -> "DecisionTree":
        """The structured node model, for rule extraction and inspection."""


@dataclass(frozen=True)
class ExperimentConfiguration:
    """All user-settable parameters of an experiment run."""
    strategies: tuple[str, ...]
    tie_policy: str = "min"
    qr_tie_policy: str = "keep"
    random_state: int | None = None
