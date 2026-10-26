"""CSV export of strategy-specific resolution explanations (full, not UI-truncated)."""
from __future__ import annotations

import io

import pandas as pd

from roughapp.business.algorithms.base import ExperimentConfiguration
from roughapp.business.algorithms.resolution import (
    generalized_decision,
    most_common_decision,
    rough_approximation,
)
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.metrics import alpha_rho_per_decision


def _profile(cond_cols: list[str], key) -> str:
    key_t = key if isinstance(key, tuple) else (key,)
    return ", ".join(f"{col}={v}" for col, v in zip(cond_cols, key_t))


def _mcd_csv(table: DecisionTable, tie_policy: str, random_state: int | None) -> str:
    df_before = table.frame
    decision_col = table.decision_column
    cond_cols = table.condition_columns
    df_after, tied_keys = most_common_decision.run(
        df_before,
        decision_col=decision_col,
        tie_policy=tie_policy,
        random_state=random_state,
    )
    tied_set = set(tied_keys)
    rows: list[dict[str, str]] = []
    for key, grp in df_before.groupby(cond_cols, sort=False):
        if grp[decision_col].nunique() <= 1:
            continue
        key_t = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=df_after.index)
        for col, val in zip(cond_cols, key_t):
            mask &= df_after[col] == val
        chosen = [str(v) for v in df_after.loc[mask, decision_col].unique().tolist()]
        tied = key_t in tied_set or key in tied_set
        counts_str = ", ".join(
            f"dec {k}: {v}×" for k, v in grp[decision_col].value_counts().items()
        )
        if tied:
            outcome = f"TIE — kept all (dec {', '.join(chosen)})"
        else:
            outcome = f"chose {chosen[0]}" if len(chosen) == 1 else f"chose {', '.join(chosen)}"
        rows.append({
            "profile": _profile(cond_cols, key),
            "decisions": counts_str,
            "outcome": outcome,
        })
    buf = io.StringIO()
    pd.DataFrame(rows, columns=["profile", "decisions", "outcome"]).to_csv(buf, index=False)
    return buf.getvalue()


def _qr_csv(table: DecisionTable, tie_policy: str, random_state: int | None) -> str:
    df_before = table.frame
    decision_col = table.decision_column
    cond_cols = table.condition_columns
    rough_approximation.run(
        df_before,
        decision_col=decision_col,
        tie_policy=tie_policy,
        random_state=random_state,
    )
    per_decision = alpha_rho_per_decision(df_before, decision_col)
    alpha_map = dict(zip(per_decision["decision"], per_decision["alpha"]))
    rows: list[dict[str, str | int | float | bool]] = []
    for key, grp in df_before.groupby(cond_cols, sort=False):
        if grp[decision_col].nunique() <= 1:
            continue
        profile = _profile(cond_cols, key)
        dec_counts = grp[decision_col].value_counts()
        alphas = {d: float(alpha_map.get(d, 0)) for d in dec_counts.index}
        min_a, max_a = min(alphas.values()), max(alphas.values())
        is_tie = min_a == max_a
        removed = {d for d, a in alphas.items() if a == min_a and not is_tie}
        for d in dec_counts.index:
            if is_tie:
                status = "tie, kept"
            elif d in removed:
                status = "removed (lowest α)"
            else:
                status = "kept"
            rows.append({
                "profile": profile,
                "decision": str(d),
                "count": int(dec_counts[d]),
                "alpha": round(alphas[d], 4),
                "status": status,
                "class_tie": is_tie,
            })
    buf = io.StringIO()
    pd.DataFrame(
        rows,
        columns=["profile", "decision", "count", "alpha", "status", "class_tie"],
    ).to_csv(buf, index=False)
    return buf.getvalue()



def _gd_csv(table: DecisionTable) -> str:
    decision_col = table.decision_column
    _, mapping = generalized_decision.run(table.frame, decision_col=decision_col)
    rows = [
        {
            "code": code,
            "decisions": "{" + ", ".join(str(v) for v in sorted(list(decisions))) + "}",
            "conflict": len(decisions) > 1,
        }
        for code, decisions in sorted(mapping.items())
    ]
    buf = io.StringIO()
    pd.DataFrame(rows, columns=["code", "decisions", "conflict"]).to_csv(buf, index=False)
    return buf.getvalue()


def details_csv(table: DecisionTable, strategy: str, config: ExperimentConfiguration) -> str:
    if strategy == "mcd":
        return _mcd_csv(table, config.tie_policy, config.random_state)
    if strategy == "rough-approximation":
        return _qr_csv(table, config.qr_tie_policy, config.random_state)
    if strategy == "gd":
        return _gd_csv(table)
    raise ValueError(f"Unknown strategy: {strategy}")
