import pandas as pd

from roughapp.core.csv_table import (
    is_type_row,
    read_csv_table,
    write_csv_table,
    default_column_types,
)
from roughapp.business.algorithms import id3 as tree_builder


def test_is_type_row():
    assert is_type_row(["c", "c", "n"])
    assert not is_type_row(["c", "x"])


def test_read_csv_table_strips_type_row(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b,dec\nc,c,c\n1,2,y\n", encoding="utf-8")
    df, types = read_csv_table(path)
    assert len(df) == 1
    assert types == {"a": "c", "b": "c", "dec": "c"}


def test_write_csv_table_roundtrip_preserves_categorical(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "dec": ["x", "y"]})
    types = {"a": "c", "dec": "c"}
    path = tmp_path / "out.csv"
    write_csv_table(df, path, types)
    back, back_types = read_csv_table(path)
    assert back_types == types
    assert not pd.api.types.is_numeric_dtype(back["a"])
    assert list(back["a"]) == ["1", "2"]


def test_tree_builder_honours_categorical_type_for_numeric_values():
    df_str = pd.DataFrame({"a": ["1", "2", "3", "1"], "dec": ["x", "x", "y", "y"]})
    df_int = pd.DataFrame({"a": [1, 2, 3, 1], "dec": ["x", "x", "y", "y"]})
    stats_str = tree_builder.build(df_str, "dec")["stats"]
    stats_int = tree_builder.build(df_int, "dec")["stats"]
    assert stats_str == stats_int


def test_builtin_breast_cancer_reads_type_row():
    df, types = read_csv_table("data/modified_breast-cancer.csv")
    assert len(df) > 0
    assert all(t == "c" for t in types.values())
