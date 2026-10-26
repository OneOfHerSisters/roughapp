"""ExperimentRunner: run selected resolution strategies over one table.

The runner is deliberately narrow.  It selects strategies from the registry,
executes them in the requested order, measures wall time, and returns result
objects.  It does not touch SQL, files, HTTP, or commits — persisting the
outcome is the ExperimentService's job.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from roughapp.business.algorithms import resolution_registry
from roughapp.business.algorithms.base import (
    ExperimentConfiguration,
    ResolutionResult,
)
from roughapp.business.exceptions import UnknownStrategyError, ValidationError
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.registries import Registry


@dataclass(frozen=True)
class StrategyOutcome:
    strategy: str
    result: ResolutionResult
    time_ms: int


class ExperimentRunner:
    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry or resolution_registry

    def run(
        self,
        table: DecisionTable,
        configuration: ExperimentConfiguration,
    ) -> list[StrategyOutcome]:
        unknown = [s for s in configuration.strategies if s not in self._registry]
        if unknown:
            raise UnknownStrategyError(unknown)

        outcomes: list[StrategyOutcome] = []
        for name in configuration.strategies:
            strategy = self._registry.get_class(name).from_config(configuration)
            t0 = time.monotonic()
            try:
                result = strategy.execute(table)
            except ValueError as e:
                raise ValidationError(str(e)) from e
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            outcomes.append(StrategyOutcome(strategy=name, result=result, time_ms=elapsed_ms))
        return outcomes
