"""Legacy dict-based experiment API (offline scripts only).

Kept for experiments_batch.py; the web application uses
business.runner.ExperimentRunner via ExperimentService.
The actual work is done by roughapp.business.runner.ExperimentRunner and the
strategy classes; this module only adapts their result objects to the flat
dicts the old interface returned, and handles saving result CSVs to a
directory for the offline script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from roughapp.business.algorithms.base import ExperimentConfiguration
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.runner import ExperimentRunner, StrategyOutcome
from roughapp.core.csv_table import (
    ColumnTypes,
    column_types_for_columns,
    default_column_types,
    write_csv_table,
)
from roughapp.core.metrics import accuracy_of_classification, quality_of_classification
from roughapp.db.models import ALL_STRATEGIES as _ALL

ALL_STRATEGIES = list(_ALL)


def save_outcome_files(
    outcome: StrategyOutcome,
    save_dir: Path,
    column_types: ColumnTypes,
) -> list[str]:
    """Write the outcome's tables as CSV files; returns the paths in order."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tables = outcome.result.tables
    paths: list[str] = []
    for i, df_out in enumerate(tables):
        name = f"{outcome.strategy}_{i}.csv" if len(tables) > 1 else f"{outcome.strategy}.csv"
        path = save_dir / name
        write_csv_table(df_out, path, column_types_for_columns(df_out.columns, column_types))
        paths.append(str(path))
    return paths


def _outcome_to_row(
    outcome: StrategyOutcome,
    gamma_before: float,
    alpha_before: float,
    rows_before: int,
    paths: list[str] | None,
) -> dict:
    m = outcome.result.metrics
    rows_after = m.get("rows")
    return {
        "strategy": outcome.strategy,
        "gamma_before": gamma_before,
        "gamma_after": m.get("gamma"),
        "alpha_before": alpha_before,
        "alpha_after": m.get("alpha"),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": (rows_before - rows_after) if rows_after is not None else 0,
        "n_gd_codes": m.get("n_gd_codes"),
        "n_gd_conflict_codes": m.get("n_gd_conflict_codes"),
        "time_ms": outcome.time_ms,
        "details": outcome.result.details,
        "result_path": paths[0] if paths else None,
    }


def run_experiment(
    df: pd.DataFrame,
    decision_col: str,
    strategies: list[str] | None = None,
    random_state: int | None = None,
    tie_policy: str = "min",
    qr_tie_policy: str = "keep",
    save_dir: Path | None = None,
    column_types: ColumnTypes | None = None,
) -> list[dict]:
    """Run the selected resolution strategies and collect before/after metrics.

    Strategy parameters are passed through to the strategies that accept them:
    *tie_policy* to MCD, *qr_tie_policy* to rough approximation, and
    *random_state* to MCD/QR when tie-breaking is "random".
    """
    selected = [s for s in (strategies or ALL_STRATEGIES) if s in ALL_STRATEGIES]
    types = column_types or default_column_types(df.columns)
    table = DecisionTable(df, decision_col, types)

    config = ExperimentConfiguration(
        strategies=tuple(selected),
        tie_policy=tie_policy,
        qr_tie_policy=qr_tie_policy,
        random_state=random_state,
    )
    outcomes = ExperimentRunner().run(table, config)

    gamma_before = quality_of_classification(df, decision_col)
    alpha_before = accuracy_of_classification(df, decision_col)
    rows_before = len(df)

    results = []
    for outcome in outcomes:
        paths = save_outcome_files(outcome, save_dir, types) if save_dir else None
        results.append(_outcome_to_row(outcome, gamma_before, alpha_before, rows_before, paths))
    return results
