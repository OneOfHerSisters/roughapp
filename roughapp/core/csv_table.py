from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Union

import pandas as pd

PathLike = Union[str, Path]

TYPE_INDICATOR_CHARS = frozenset({"c", "n", "i", "r", "l"})
CATEGORICAL_TYPES = frozenset({"c", "l"})
NUMERIC_TYPES = frozenset({"n", "i", "r"})
ColumnTypes = Dict[str, str]


def default_column_types(columns) -> ColumnTypes:
    return {str(c): "c" for c in columns}


def is_type_row(row) -> bool:
    vals = {str(v).strip().lower() for v in row}
    return bool(vals) and vals.issubset(TYPE_INDICATOR_CHARS)


def types_from_row(columns, row) -> ColumnTypes:
    return {
        str(col): str(val).strip().lower()
        for col, val in zip(columns, row)
    }


def read_csv_table(path_or_bytes) -> tuple[pd.DataFrame, ColumnTypes]:
    if isinstance(path_or_bytes, (str, Path)):
        raw = pd.read_csv(path_or_bytes)
    else:
        raw = pd.read_csv(io.BytesIO(path_or_bytes))

    column_types = default_column_types(raw.columns)
    if len(raw) > 0 and is_type_row(raw.iloc[0]):
        column_types = types_from_row(raw.columns, raw.iloc[0])
        raw = raw.iloc[1:].reset_index(drop=True)
    return raw, column_types


def read_csv(path_or_bytes) -> pd.DataFrame:
    return read_csv_table(path_or_bytes)[0]


def write_csv_table(df: pd.DataFrame, path: PathLike, column_types: ColumnTypes) -> None:
    cols = list(df.columns)
    types = {c: column_types.get(c, "c") for c in cols}
    type_row = pd.DataFrame([{c: types[c] for c in cols}])

    out = df.copy()
    for c in cols:
        if types[c] in CATEGORICAL_TYPES:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(x))

    pd.concat([type_row, out], ignore_index=True).to_csv(path, index=False)


def column_types_for_columns(columns, source_types: ColumnTypes | None = None) -> ColumnTypes:
    if source_types is None:
        return default_column_types(columns)
    return {c: source_types.get(c, "c") for c in columns}
