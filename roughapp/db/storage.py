"""File storage behind an interface.

Services talk to FileStorage in terms of *storage keys* (the relative-ish
paths persisted in the database) and never assemble absolute paths
themselves.  LocalFileStorage is the only real implementation today; an
S3FileStorage would be one new class with the same Protocol, injected in
roughapp.api.dependencies — no service changes.

Note on transactions: PostgreSQL cannot roll back the filesystem.  The
services therefore write files before commit and compensate (delete the
file) when the database operation fails; deletions run after a successful
commit and are best-effort.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from roughapp.business.exceptions import StorageError
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.csv_table import (
    ColumnTypes,
    column_types_for_columns,
    read_csv,
    read_csv_table,
    write_csv_table,
)


class FileStorage(Protocol):
    def save_bytes(self, content: bytes, key: str) -> str: ...
    def save_table(self, table: DecisionTable, key: str) -> str: ...
    def read_table(self, key: str, decision_column: str) -> DecisionTable: ...
    def read_frame(self, key: str) -> pd.DataFrame: ...
    def delete(self, key: str) -> None: ...
    def upload_key(self, record_id: int, filename: str) -> str: ...
    def tree_artifact_key(self, tree_run_id: int) -> str: ...


class LocalFileStorage:
    """CSV files under data/uploads; tree text under data/tree_artifacts."""

    def __init__(
        self,
        upload_dir: str | Path = "data/uploads",
        tree_artifacts_dir: str | Path = "data/tree_artifacts",
    ) -> None:
        self._upload_dir = Path(upload_dir)
        self._tree_artifacts_dir = Path(tree_artifacts_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._tree_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -- key construction ---------------------------------------------------
    def upload_key(self, record_id: int, filename: str) -> str:
        # The record's own id prefixes the file name, so two users uploading
        # "data.csv" can never collide on disk.
        return str(self._upload_dir / f"{record_id}_{filename}")

    def tree_artifact_key(self, tree_run_id: int) -> str:
        return str(self._tree_artifacts_dir / f"{tree_run_id}.txt")

    # -- IO -------------------------------------------------------------------
    def save_bytes(self, content: bytes, key: str) -> str:
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        Path(key).write_bytes(content)
        return key

    def save_table(self, table: DecisionTable, key: str) -> str:
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        types: ColumnTypes = column_types_for_columns(table.frame.columns, table.column_types)
        write_csv_table(table.frame, Path(key), types)
        return key

    def read_table(self, key: str, decision_column: str) -> DecisionTable:
        try:
            df, types = read_csv_table(key)
        except Exception as e:
            raise StorageError(f"Could not read stored table: {e}") from e
        return DecisionTable(df, decision_column, types)

    def read_frame(self, key: str) -> pd.DataFrame:
        try:
            return read_csv(key)
        except Exception as e:
            raise StorageError(f"Could not read stored table: {e}") from e

    def delete(self, key: str) -> None:
        try:
            Path(key).unlink(missing_ok=True)
        except OSError:
            pass  # best-effort: the DB row is the source of truth
