"""Unit of Work: the transaction boundary handle given to business services.

Repositories never commit; a business service finishes a scenario with one
commit() (or rollback() on failure).  This is a thin wrapper over the
SQLAlchemy session on purpose — the session already *is* a unit of work,
the class only hides it from the business layer.
"""
from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session


class UnitOfWork(Protocol):
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def flush(self) -> None: ...


class SqlAlchemyUnitOfWork:
    def __init__(self, session: Session) -> None:
        self._session = session

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def flush(self) -> None:
        self._session.flush()
