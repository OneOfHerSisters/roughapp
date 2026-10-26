"""Display names for experiment result datasets."""
from __future__ import annotations

_STRATEGY_LABELS = {
    "mcd": "MCD",
    "rough-approximation": "Rough approx.",
    "gd": "GD",
}


def format_experiment_result_name(
    input_name: str,
    strategy: str,
    experiment_id: int,
) -> str:
    strat = _STRATEGY_LABELS.get(strategy, strategy)
    return f"{input_name} · {strat} (exp #{experiment_id})"
