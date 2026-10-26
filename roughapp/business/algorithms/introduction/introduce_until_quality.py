import numpy as np
import pandas as pd
from typing import Optional

from roughapp.core.metrics import quality_of_classification
from roughapp.business.algorithms.introduction.duplicate_flip import run as duplicate_flip


def run(
    df: pd.DataFrame,
    *,
    decision_col: str,
    target_gamma: float,
    random_state: Optional[int] = None,
    max_iter: int = 10_000,
) -> tuple[pd.DataFrame, float]:
    """Introduce conflicts one row at a time until γ ≤ target_gamma.

    Each iteration appends one duplicate with a flipped decision (via
    duplicate_flip), so |U| grows monotonically and γ converges.  Because the
    flip step is discrete, the returned γ is the first value that crossed the
    target — it will typically be slightly *below* target_gamma, not equal.

    Returns the modified dataframe and the achieved γ.
    """
    if not (0.0 <= target_gamma <= 1.0):
        raise ValueError("target_gamma must be in [0, 1]")

    rng = np.random.default_rng(random_state)
    gamma = quality_of_classification(df, decision_col)
    it = 0

    while gamma > target_gamma and it < max_iter:
        seed = int(rng.integers(0, 2**31))
        df = duplicate_flip(df, decision_col=decision_col, k=1, random_state=seed)
        gamma = quality_of_classification(df, decision_col)
        it += 1

    return df, gamma
