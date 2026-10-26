"""Rebuild inconsistent benchmark tables from Azad et al. (2013, RSFDGrC).

The authors do not publish ready-made CSV files.  They start from UCI tables,
fill missing values, remove identifier columns, then drop additional conditional
attributes (suffix ``-N`` = number of attributes removed) until rows with the
same conditions can carry different decisions.

Table row counts in the paper count *equivalence classes* (MVD view).  We store
a minimal inconsistent table: one physical row per distinct decision within each
class, so resolution strategies work on standard single-valued rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from roughapp.core.grouping import ind_classes_table
from roughapp.core.utils import split_C_D

from roughapp.core.csv_table import read_csv_table, write_csv_table, column_types_for_columns


@dataclass(frozen=True)
class AzadSpec:
    name: str
    source: str
    drop: Tuple[str, ...]
    classes: int
    attrs: int
    spectrum: Dict[int, int]


# Table 1 — Azad, Chikalov, Moshkov (2013), RSFDGrC.
# kr-vs-kp-* and spect-test-* need raw UCI sources not bundled in this repo.
AZAD_BENCHMARKS: Tuple[AzadSpec, ...] = (
    AzadSpec("balance-scale-1", "data/modified_balance-scale.csv", ("left-weight",), 125, 3, {1: 45, 2: 50, 3: 30}),
    AzadSpec("breast-cancer-1", "data/modified_breast-cancer.csv", ("tumor-size",), 193, 8, {1: 169, 2: 24}),
    AzadSpec("breast-cancer-5", "data/modified_breast-cancer.csv",
             ("inv-nodes", "node-caps", "deg-malig", "breast-quad", "irradiat"), 98, 4, {1: 58, 2: 40}),
    AzadSpec("cars-1", "data/modified_cars.csv", ("buying",), 432, 5, {1: 258, 2: 161, 3: 13}),
    AzadSpec("flags-5", "data/modified_flags.csv", ("zone", "language", "religion", "circles", "sunstars"), 171, 21, {1: 159, 2: 12}),
    AzadSpec("hayes-roth-data-1", "data/modified_hayes-roth-data.csv", ("marital status",), 39, 3, {1: 22, 2: 13, 3: 4}),
    AzadSpec("lymphography-5", "data/modified_lymphography.csv",
             ("lymphatics", "changes in node", "changes in stru", "special forms", "no of nodes in"), 122, 13, {1: 113, 2: 9}),
    AzadSpec("mushroom-5", "data/modified_mushroom.csv",
             ("odor", "gill-size", "stalk-root", "stalk-surface-below-ring", "habitat"), 4078, 17, {1: 4048, 2: 30}),
    AzadSpec("nursery-4", "data/modified_nursery.csv", ("parents", "housing", "finance", "social"), 240, 4, {1: 97, 2: 96, 3: 47}),
    AzadSpec("nursery-1", "data/modified_nursery.csv", ("parents",), 4320, 7, {1: 2858, 2: 1460, 3: 2}),
    AzadSpec("teeth-1", "data/modified_Teeth.csv", ("Top incisors",), 22, 7, {1: 12, 2: 10}),
    AzadSpec("teeth-5", "data/modified_Teeth.csv",
             ("Bottom incisors", "Top canines", "Bottom canines", "Top premolars", "Bottom molars"), 14, 3, {1: 6, 2: 3, 3: 3, 4: 1, 5: 1}),
    AzadSpec("tic-tac-toe-4", "data/modified_tic-tac-toe.csv",
             ("top-right-square", "middle-middle-square", "bottom-left-square", "bottom-right-square"), 231, 5, {1: 102, 2: 129}),
    AzadSpec("tic-tac-toe-3", "data/modified_tic-tac-toe.csv",
             ("middle-middle-square", "bottom-left-square", "bottom-right-square"), 449, 6, {1: 300, 2: 149}),
    AzadSpec("zoo-data-5", "data/modified_zoo-data.csv", ("feathers", "backbone", "breathes", "legs", "tail"), 42, 11, {1: 36, 2: 6}),
)


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _resolve_column(columns: Iterable[str], name: str) -> Optional[str]:
    target = _norm(name)
    cols = list(columns)
    for c in cols:
        if _norm(c) == target:
            return c
    for c in cols:
        cn = _norm(c)
        if target in cn or cn in target:
            return c
    return None


def to_minimal_inconsistent(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    """One row per (equivalence class, decision) pair."""
    C, _ = split_C_D(df, decision_col)
    rows: list[pd.Series] = []
    for _, group in df.groupby(C, dropna=False, sort=False):
        for decision in group[decision_col].unique():
            rows.append(group[group[decision_col] == decision].iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def spectrum(df: pd.DataFrame, decision_col: str) -> Tuple[int, Dict[int, int]]:
    table = ind_classes_table(df, decision_col)
    spec: Dict[int, int] = {}
    for key in table.index:
        k = int((table.loc[key] > 0).sum())
        spec[k] = spec.get(k, 0) + 1
    return len(table), spec


def build_azad_table(spec: AzadSpec, *, root: Path | None = None) -> pd.DataFrame:
    root = root or Path(".")
    src = root / spec.source
    if not src.exists():
        raise FileNotFoundError(src)

    df, source_types = read_csv_table(src)
    decision_col = df.columns[-1]
    drop_cols = [_resolve_column(df.columns, a) for a in spec.drop]
    if any(c is None for c in drop_cols):
        missing = [a for a, c in zip(spec.drop, drop_cols) if c is None]
        raise ValueError(f"{spec.name}: column(s) not found: {missing}")

    reduced = df.drop(columns=drop_cols)  # type: ignore[arg-type]
    return to_minimal_inconsistent(reduced, decision_col)


def verify_azad_table(df: pd.DataFrame, spec: AzadSpec) -> dict:
    decision_col = df.columns[-1]
    n_classes, spec_counts = spectrum(df, decision_col)
    return {
        "name": spec.name,
        "rows": len(df),
        "attrs": len(df.columns) - 1,
        "classes": n_classes,
        "spectrum": spec_counts,
        "classes_ok": n_classes == spec.classes,
        "attrs_ok": len(df.columns) - 1 == spec.attrs,
        "spectrum_ok": spec_counts == spec.spectrum,
    }


def write_azad_benchmarks(out_dir: Path | str = "data/azad", *, root: Path | None = None) -> List[dict]:
    root = root or Path(".")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports: List[dict] = []

    for spec in AZAD_BENCHMARKS:
        df = build_azad_table(spec, root=root)
        path = out / f"{spec.name}.csv"
        _, source_types = read_csv_table(root / spec.source)
        write_csv_table(df, path, column_types_for_columns(df.columns, source_types))
        report = verify_azad_table(df, spec)
        report["path"] = str(path)
        reports.append(report)

    return reports
