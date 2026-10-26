"""Tree builders.

Id3TreeBuilder adapts the ID3 implementation in roughapp.business.algorithms.id3
to the TreeBuilder interface: ``build`` returns the summary statistics and the
printable tree, ``build_tree`` returns the structured DecisionNode/LeafNode
model consumed by rule extraction.  A second algorithm would be one new class
registered in tree_builder_registry, matching a new row in the tree_algorithms
lookup table.
"""
from __future__ import annotations

from roughapp.business.algorithms.base import TreeBuilder, TreeBuildResult
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.models.decision_tree import DecisionTree
from roughapp.business.algorithms import id3 as _id3


class Id3TreeBuilder(TreeBuilder):
    name = "id3"

    def build(self, table: DecisionTable, max_depth: int | None = None) -> TreeBuildResult:
        raw = _id3.build(table.frame, table.decision_column, max_depth=max_depth)
        stats = raw["stats"]
        return TreeBuildResult(
            depth=stats["depth"],
            avg_depth=stats["avg_depth"],
            leaves=stats["leaves"],
            internal_nodes=stats["internal_nodes"],
            text=raw["text"],
        )

    def build_tree(self, table: DecisionTable, max_depth: int | None = None) -> DecisionTree:
        return _id3.build_tree(table.frame, table.decision_column, max_depth=max_depth)
