import pandas as pd

from roughapp.core.csv_table import read_csv as _read_csv
from roughapp.business.algorithms import id3 as tree_builder


def test_tree_builder_string_dtype_columns():
    """pandas 2.x uses StringDtype; categorical columns must stay nominal."""
    df = _read_csv("data/modified_breast-cancer.csv")
    decision_col = df.columns[-1]
    result = tree_builder.build(df, decision_col)
    assert result["stats"]["depth"] > 0
    assert result["stats"]["leaves"] > 0


def test_id3_is_the_only_registered_algorithm():
    from roughapp.business.algorithms import tree_builder_registry
    from roughapp.db.reference_seed import TREE_ALGORITHM_ROWS
    assert tree_builder_registry.names() == ["id3"]
    assert [name for name, _ in TREE_ALGORITHM_ROWS] == ["id3"]


def test_id3_branches_on_every_attribute_value():
    """One edge per value: a single attribute that fully determines the
    decision yields one internal node and one leaf per value."""
    df = pd.DataFrame({
        "colour": ["red", "green", "blue", "red", "green", "blue"],
        "dec": ["a", "b", "c", "a", "b", "c"],
    })
    stats = tree_builder.build(df, "dec")["stats"]
    assert stats["depth"] == 1
    assert stats["leaves"] == 3
    assert stats["internal_nodes"] == 1


def test_id3_stops_on_indiscernible_rows():
    """Conflicting rows cannot be separated, so the node becomes a leaf."""
    df = pd.DataFrame({"a": ["x", "x"], "dec": ["0", "1"]})
    stats = tree_builder.build(df, "dec")["stats"]
    assert stats["internal_nodes"] == 0
    assert stats["leaves"] == 1
