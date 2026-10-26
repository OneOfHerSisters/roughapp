"""Structured decision-tree model, independent of any learning library.

Adapters (Id3TreeBuilder.build_tree) convert an induced tree into these
nodes, so rule extraction and any future consumers never depend on how the
tree was built.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field


class TreeNode(ABC):
    pass


@dataclass(frozen=True)
class NominalNode(TreeNode):
    """Internal node querying a nominal attribute.

    One child per value of the attribute, keyed by that value, which is how
    decision trees over decision tables are defined: a path spells out a
    conjunction of descriptors (a = v).
    """
    attribute: str
    children: dict[str, TreeNode] = field(default_factory=dict)


@dataclass(frozen=True)
class LeafNode(TreeNode):
    decision: str
    support: int        # training objects that reach this leaf
    confidence: float   # fraction of them carrying the majority decision


@dataclass(frozen=True)
class DecisionTree:
    root: TreeNode
