"""Tests for resolution strategies against the thesis Table 1 fixture.

Conflict class: {x3, x9} — both have (a=1,b=1,c=1,d=0) but dec=0 (x3) and dec=2 (x9).
γ_before = 7/9.
α(X0)=3/5=0.60, α(X1)=2/2=1.0, α(X2)=2/4=0.50.
"""

import pytest
import pandas as pd

from roughapp.core.metrics import quality_of_classification
from roughapp.business.algorithms.resolution import (
    most_common_decision,
    rough_approximation,
    generalized_decision,
)

DCOL = "dec"
CONFLICT_KEY = (1, 1, 1, 0)  # (a, b, c, d) for the conflict class


# ── helpers ──────────────────────────────────────────────────────────────────

def _conflict_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["a"] == 1) & (df["b"] == 1) & (df["c"] == 1) & (df["d"] == 0)]


def _is_consistent(df: pd.DataFrame) -> bool:
    return quality_of_classification(df, DCOL) == 1.0


# ── baseline ─────────────────────────────────────────────────────────────────

def test_fixture_gamma(table1):
    assert quality_of_classification(table1, DCOL) == pytest.approx(7 / 9)


# ── MCD ──────────────────────────────────────────────────────────────────────

class TestMCD:
    def test_tie_keep_leaves_conflict(self, table1):
        result, tied = most_common_decision.run(table1, decision_col=DCOL, tie_policy="keep")
        # Both decisions still present in the conflict class
        decisions_in_class = set(_conflict_rows(result)[DCOL].unique())
        assert decisions_in_class == {0, 2}
        # Exactly one tied class recorded
        assert len(tied) == 1
        assert tied[0] == CONFLICT_KEY
        # γ unchanged
        assert quality_of_classification(result, DCOL) == pytest.approx(7 / 9)

    def test_tie_keep_does_not_mutate_input(self, table1):
        original = table1.copy()
        most_common_decision.run(table1, decision_col=DCOL, tie_policy="keep")
        pd.testing.assert_frame_equal(table1, original)

    def test_tie_first_resolves_to_smaller_value(self, table1):
        # Tied between 0 and 2; "first" picks 0 (smaller / first in sorted columns)
        result, tied = most_common_decision.run(table1, decision_col=DCOL, tie_policy="first")
        assert tied == []
        conflict = _conflict_rows(result)
        assert set(conflict[DCOL].unique()) == {0}
        assert _is_consistent(result)
        assert len(result) == 8  # one duplicate row dropped

    def test_tie_random_resolves(self, table1):
        result, tied = most_common_decision.run(
            table1, decision_col=DCOL, tie_policy="random", random_state=0
        )
        assert tied == []
        assert len(_conflict_rows(result)[DCOL].unique()) == 1
        assert _is_consistent(result)

    def test_tie_raise_raises(self, table1):
        with pytest.raises(ValueError, match="Tie in equivalence class"):
            most_common_decision.run(table1, decision_col=DCOL, tie_policy="raise")

    def test_no_conflict_table_unchanged(self):
        df = pd.DataFrame({"a": [0, 1], "b": [0, 1], "dec": [0, 1]})
        result, tied = most_common_decision.run(df, decision_col="dec")
        assert tied == []
        pd.testing.assert_frame_equal(result, df)


# ── Rough approximation ───────────────────────────────────────────────────────

class TestRoughApproximation:
    def test_removes_x9(self, table1):
        # α(X0)=0.60 > α(X2)=0.50 → x9 (dec=2) removed, x3 (dec=0) kept
        result = rough_approximation.run(table1, decision_col=DCOL)
        assert len(result) == 8
        conflict = _conflict_rows(result)
        # Only x3 remains in the class
        assert len(conflict) == 1
        assert conflict.iloc[0][DCOL] == 0

    def test_result_is_consistent(self, table1):
        result = rough_approximation.run(table1, decision_col=DCOL)
        assert _is_consistent(result)

    def test_does_not_mutate_input(self, table1):
        original = table1.copy()
        rough_approximation.run(table1, decision_col=DCOL)
        pd.testing.assert_frame_equal(table1, original)

    def test_no_conflict_table_unchanged(self):
        df = pd.DataFrame({"a": [0, 1], "b": [0, 1], "dec": [0, 1]})
        result = rough_approximation.run(df, decision_col="dec")
        pd.testing.assert_frame_equal(result, df)



# ── GD ────────────────────────────────────────────────────────────────────────

class TestGD:
    def test_row_count(self, table1):
        result, _ = generalized_decision.run(table1, decision_col=DCOL)
        assert len(result) == 8

    def test_four_unique_codes(self, table1):
        result, mapping = generalized_decision.run(table1, decision_col=DCOL)
        unique_codes = result[DCOL].nunique()
        assert unique_codes == 4
        assert len(mapping) == 4

    def test_mapping_is_bijective(self, table1):
        result, mapping = generalized_decision.run(table1, decision_col=DCOL)
        # Every code in the df must appear in mapping
        for code in result[DCOL].unique():
            assert code in mapping
        # All mapping values are frozensets
        for code, fs in mapping.items():
            assert isinstance(fs, frozenset)
        # No two codes map to the same frozenset
        assert len(set(mapping.values())) == len(mapping)

    def test_conflict_class_maps_to_two_element_set(self, table1):
        result, mapping = generalized_decision.run(table1, decision_col=DCOL)
        conflict = result[
            (result["a"] == 1) & (result["b"] == 1)
            & (result["c"] == 1) & (result["d"] == 0)
        ]
        assert len(conflict) == 1
        code = conflict.iloc[0][DCOL]
        assert mapping[code] == frozenset({0, 2})

    def test_singleton_codes_distinct_from_conflict_code(self, table1):
        result, mapping = generalized_decision.run(table1, decision_col=DCOL)
        conflict_code = result[
            (result["a"] == 1) & (result["b"] == 1)
            & (result["c"] == 1) & (result["d"] == 0)
        ].iloc[0][DCOL]
        for code, fs in mapping.items():
            if code != conflict_code:
                assert len(fs) == 1

    def test_decision_column_is_integer(self, table1):
        result, _ = generalized_decision.run(table1, decision_col=DCOL)
        assert result[DCOL].dtype == int or all(
            isinstance(v, int) for v in result[DCOL]
        )

    def test_does_not_mutate_input(self, table1):
        original = table1.copy()
        generalized_decision.run(table1, decision_col=DCOL)
        pd.testing.assert_frame_equal(table1, original)
