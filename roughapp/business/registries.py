"""Explicit registries for algorithm families.

One small generic registry class, instantiated once per family in
roughapp.business.algorithms.  Registration is explicit — no module
scanning, no import side effects.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from roughapp.business.exceptions import UnknownStrategyError, ValidationError

T = TypeVar("T")


class StrategyAlreadyRegisteredError(ValidationError):
    def __init__(self, name: str):
        super().__init__(f"Strategy '{name}' is already registered")


class Registry(Generic[T]):
    """Maps a public name to an algorithm class and instantiates it."""

    def __init__(self, family: str) -> None:
        self._family = family
        self._classes: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"{cls.__name__} has no 'name' attribute")
        if name in self._classes:
            raise StrategyAlreadyRegisteredError(name)
        self._classes[name] = cls
        return cls

    def names(self) -> list[str]:
        return list(self._classes)

    def __contains__(self, name: str) -> bool:
        return name in self._classes

    def get_class(self, name: str) -> type[T]:
        cls = self._classes.get(name)
        if cls is None:
            raise UnknownStrategyError([name])
        return cls

    def create(self, name: str, **parameters) -> T:
        return self.get_class(name)(**parameters)
