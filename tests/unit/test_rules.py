"""Unit tests for the structured tree model and rule extraction."""
import pandas as pd
import pytest

from roughapp.business.algorithms import rule_extractor_registry, tree_builder_registry
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.models.decision_tree import LeafNode, NominalNode


@pytest.fixture
def consistent_table() -> DecisionTable:
    # perfectly separable: a=1 -> P, a=2 -> N
    df = pd.DataFrame({
        "a": ["1", "1", "2", "2"],
        "b": ["x", "y", "x", "y"],
        "dec": ["P", "P", "N", "N"],
    })
    return DecisionTable(df, "dec", {"a": "c", "b": "c", "dec": "c"})


def test_build_tree_structure(consistent_table):
    tree = tree_builder_registry.create("id3").build_tree(consistent_table)
    assert isinstance(tree.root, NominalNode)
    assert tree.root.attribute == "a"  # b carries no information
    assert set(tree.root.children) == {"1", "2"}
    assert all(isinstance(c, LeafNode) for c in tree.root.children.values())


def test_extract_rules_from_paths(consistent_table):
    builder = tree_builder_registry.create("id3")
    extractor = rule_extractor_registry.create("tree-paths")
    rules = extractor.extract(builder.build_tree(consistent_table))

    assert len(rules) == 2  # two leaves
    conclusions = {r.conclusion for r in rules.rules}
    assert conclusions == {"P", "N"}
    for r in rules.rules:
        assert r.confidence == 1.0
        assert r.support == 2
        # every edge is a descriptor (attribute = value)
        assert all(c.attribute in {"a", "b"} for c in r.conditions)
        assert all(c.operator == "==" for c in r.conditions)


def test_rules_sorted_by_confidence(table1):
    table = DecisionTable(table1, "dec")
    builder = tree_builder_registry.create("id3")
    rules = rule_extractor_registry.create("tree-paths").extract(builder.build_tree(table))
    confidences = [r.confidence for r in rules.rules]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(r.support for r in rules.rules) == 9  # every object lands in one leaf
