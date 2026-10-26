"""Tests for missing-value detection and completion strategies.

Fixture ``incomplete`` has three missing markers (mixed ``?`` / ``NULL``)
across two conditional columns and one numeric column, spanning both decision
classes so the by-class fill can diverge from the global fill.
"""

import pandas as pd
import pytest

from roughapp.core.missing import (
    is_missing_mask,
    normalize_missing,
    missing_summary,
    MISSING_TOKENS,
)
from roughapp.business.algorithms.completion import remove_incomplete, fill_values

DCOL = "dec"


@pytest.fixture
def incomplete() -> pd.DataFrame:
    """5 objects; missing in `a` (row1), `b` (row2), `num` (row2).

    Class P rows: 0,1,4 ; class N rows: 2,3.
    Column `num` non-missing values: 10,20,_,40,50 → global mean 30,
    but within class N (rows 2,3) the only present value is 40.
    """
    return pd.DataFrame(
        {
            "a":   ["x", "?", "x", "y", "y"],
            "b":   ["1", "2", "NULL", "2", "1"],
            "num": ["10", "20", "?", "40", "50"],
            "dec": ["P", "P", "N", "N", "P"],
        }
    )


# ── core.missing ──────────────────────────────────────────────────────────────

class TestMissingDetection:
    def test_tokens_case_insensitive(self):
        df = pd.DataFrame({"x": ["a", "?", "NuLl", "Missing", ""]})
        mask = is_missing_mask(df)
        assert mask["x"].tolist() == [False, True, True, True, True]

    def test_native_nan_detected(self):
        df = pd.DataFrame({"x": ["a", None, float("nan")]})
        assert is_missing_mask(df)["x"].tolist() == [False, True, True]

    def test_normalize_replaces_with_na(self, incomplete):
        norm = normalize_missing(incomplete)
        assert norm.isna().sum().sum() == 3
        # A real value is left untouched
        assert norm.loc[0, "a"] == "x"

    def test_summary_counts(self, incomplete):
        s = missing_summary(incomplete)
        assert s["total_missing"] == 3
        assert s["rows_affected"] == 2
        assert s["total_rows"] == 5
        per = {c["column"]: c["missing"] for c in s["per_column"]}
        assert per == {"a": 1, "b": 1, "num": 1}

    def test_summary_clean_table(self):
        df = pd.DataFrame({"a": ["x", "y"], "dec": ["P", "N"]})
        s = missing_summary(df)
        assert s["total_missing"] == 0
        assert s["per_column"] == []

    def test_all_tokens_recognized(self):
        df = pd.DataFrame({"x": sorted(MISSING_TOKENS)})
        assert is_missing_mask(df)["x"].all()


# ── remove_incomplete ─────────────────────────────────────────────────────────

class TestRemoveIncomplete:
    def test_drops_rows_with_missing(self, incomplete):
        result = remove_incomplete.run(incomplete, decision_col=DCOL)
        assert len(result) == 3
        assert is_missing_mask(result).to_numpy().sum() == 0

    def test_index_reset(self, incomplete):
        result = remove_incomplete.run(incomplete, decision_col=DCOL)
        assert list(result.index) == list(range(len(result)))

    def test_does_not_mutate_input(self, incomplete):
        original = incomplete.copy()
        remove_incomplete.run(incomplete, decision_col=DCOL)
        pd.testing.assert_frame_equal(incomplete, original)

    def test_clean_table_unchanged(self):
        df = pd.DataFrame({"a": ["x", "y"], "dec": ["P", "N"]})
        result = remove_incomplete.run(df, decision_col="dec")
        pd.testing.assert_frame_equal(result, df)


# ── fill_values ───────────────────────────────────────────────────────────────

class TestFillValues:
    def test_no_missing_after_fill(self, incomplete):
        result = fill_values.run(incomplete, decision_col=DCOL)
        assert is_missing_mask(result).to_numpy().sum() == 0
        assert len(result) == len(incomplete)

    def test_numeric_filled_with_global_mean(self, incomplete):
        result = fill_values.run(incomplete, decision_col=DCOL)
        # (10+20+40+50)/4 = 30
        assert result.loc[2, "num"] == pytest.approx(30.0)

    def test_by_class_uses_class_statistic(self, incomplete):
        result = fill_values.run(incomplete, decision_col=DCOL, by_class=True)
        # Row 2 is class N; the only present num in class N is 40
        assert result.loc[2, "num"] == pytest.approx(40.0)

    def test_nominal_filled_with_mode(self, incomplete):
        result = fill_values.run(incomplete, decision_col=DCOL)
        # `a` present values x,x,y,y → mode is x (tie broken to first). Row 1
        # was the missing one. `a` is non-numeric so mode-fill applies.
        assert result.loc[1, "a"] == "x"

    def test_decision_column_never_filled(self):
        df = pd.DataFrame({"a": ["x", "y"], "dec": ["P", "?"]})
        result = fill_values.run(df, decision_col="dec")
        # decision '?' is left as-is (not treated as a conditional attribute)
        assert result.loc[1, "dec"] == "?"

    def test_does_not_mutate_input(self, incomplete):
        original = incomplete.copy()
        fill_values.run(incomplete, decision_col=DCOL, by_class=True)
        pd.testing.assert_frame_equal(incomplete, original)
