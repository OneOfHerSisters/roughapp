"""Azad et al. (2013) benchmark tables rebuilt from local UCI-derived CSVs."""
from pathlib import Path

import pytest

from roughapp.core.csv_table import read_csv as _read_csv
from roughapp.business.algorithms import id3 as tree_builder
from roughapp.business.datasets.azad import AZAD_BENCHMARKS, build_azad_table, verify_azad_table
from roughapp.business.algorithms.resolution import most_common_decision

ROOT = Path(__file__).resolve().parents[1]

EXACT_SPECTRUM = {
    "balance-scale-1", "cars-1", "flags-5", "lymphography-5", "mushroom-5",
    "nursery-1", "nursery-4", "tic-tac-toe-3", "tic-tac-toe-4", "zoo-data-5",
}


@pytest.mark.parametrize("spec", AZAD_BENCHMARKS, ids=lambda s: s.name)
def test_azad_table_matches_paper_classes(spec):
    df = build_azad_table(spec, root=ROOT)
    report = verify_azad_table(df, spec)
    assert report["classes_ok"], f"{spec.name}: classes {report['classes']} != {spec.classes}"
    assert report["attrs_ok"], f"{spec.name}: attrs {report['attrs']} != {spec.attrs}"


@pytest.mark.parametrize(
    "spec",
    [s for s in AZAD_BENCHMARKS if s.name in EXACT_SPECTRUM],
    ids=lambda s: s.name,
)
def test_azad_spectrum_exact(spec):
    df = build_azad_table(spec, root=ROOT)
    report = verify_azad_table(df, spec)
    assert report["spectrum_ok"], f"{spec.name}: {report['spectrum']} != {spec.spectrum}"


def test_mushroom5_tree_is_small_and_mcd_helps():
    """Paper reports depth ~7–8 (their greedy U tree); sklearn CART differs but stays small."""
    df = build_azad_table(next(s for s in AZAD_BENCHMARKS if s.name == "mushroom-5"), root=ROOT)
    col = df.columns[-1]
    raw = tree_builder.build(df, col)["stats"]
    mcd_df, _ = most_common_decision.run(df, decision_col=col, tie_policy="min")
    mcd = tree_builder.build(mcd_df, col)["stats"]

    assert raw["depth"] < 25
    assert raw["internal_nodes"] + raw["leaves"] < 300
    assert mcd["depth"] <= raw["depth"]
    assert mcd["internal_nodes"] + mcd["leaves"] <= raw["internal_nodes"] + raw["leaves"]


def test_azad_csv_files_exist_after_build():
    missing = [s.name for s in AZAD_BENCHMARKS if not (ROOT / "data" / "azad" / f"{s.name}.csv").exists()]
    if missing:
        pytest.skip(f"Run scripts/build_azad_datasets.py first; missing: {missing[:3]}…")
    df = _read_csv(ROOT / "data/azad/mushroom-5.csv")
    assert len(df) == 4108
