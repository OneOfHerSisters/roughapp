"""ID3 decision-tree induction for decision tables with nominal attributes.

Quinlan's ID3: at every node the attribute with the highest information gain
is selected, and the node branches on *each* value of that attribute — one
edge per value.  Nothing is encoded and no thresholds are used, which matches
the way decision trees are defined over decision tables in the rough-set
literature: an attribute is queried once per path, so the depth of a tree is
the number of attributes consulted.

Attributes are treated as nominal, values compared for equality only; a used
attribute is not queried again further down the same path.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from roughapp.business.models.decision_tree import (
    DecisionTree,
    LeafNode,
    NominalNode,
    TreeNode,
)


def _entropy(labels: pd.Series) -> float:
    counts = labels.value_counts()
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counts.to_numpy() if c > 0
    )


def _leaf(labels: pd.Series) -> LeafNode:
    """Majority decision; ties broken by the smallest value for determinism."""
    counts = labels.value_counts()
    best = max(counts.max(), 0)
    winners = sorted(str(v) for v in counts[counts == best].index)
    support = int(counts.sum())
    return LeafNode(
        decision=winners[0],
        support=support,
        confidence=round(best / support, 4) if support else 0.0,
    )


def _grow(
    df: pd.DataFrame,
    decision_col: str,
    attributes: list[str],
    depth: int,
    max_depth: int | None,
) -> TreeNode:
    labels = df[decision_col]

    if labels.nunique() <= 1 or not attributes:
        return _leaf(labels)
    if max_depth is not None and depth >= max_depth:
        return _leaf(labels)

    base = _entropy(labels)
    total = len(df)
    best_gain = 0.0
    best_attr: str | None = None
    best_groups: dict[Any, pd.DataFrame] | None = None

    for attr in attributes:  # column order decides ties, keeping runs reproducible
        groups = {val: grp for val, grp in df.groupby(attr, sort=False, dropna=False)}
        if len(groups) < 2:
            continue  # constant here: querying it separates nothing
        remainder = sum(
            (len(grp) / total) * _entropy(grp[decision_col]) for grp in groups.values()
        )
        gain = base - remainder
        if gain > best_gain:
            best_gain, best_attr, best_groups = gain, attr, groups

    if best_attr is None or best_groups is None:
        # No attribute yields information: the rows are indiscernible but carry
        # different decisions (an inconsistent class), so the node is a leaf.
        return _leaf(labels)

    remaining = [a for a in attributes if a != best_attr]
    children = {
        str(val): _grow(grp, decision_col, remaining, depth + 1, max_depth)
        for val, grp in best_groups.items()
    }
    return NominalNode(attribute=best_attr, children=children)


def _walk_stats(root: TreeNode) -> dict:
    leaves = 0
    internal = 0
    max_depth = 0
    weighted_depth = 0
    objects = 0

    def walk(node: TreeNode, depth: int) -> None:
        nonlocal leaves, internal, max_depth, weighted_depth, objects
        if isinstance(node, LeafNode):
            leaves += 1
            max_depth = max(max_depth, depth)
            weighted_depth += node.support * depth
            objects += node.support
            return
        assert isinstance(node, NominalNode)
        internal += 1
        for child in node.children.values():
            walk(child, depth + 1)

    walk(root, 0)
    return {
        "depth": max_depth,
        # Averaged over objects, as the average depth is defined for decision
        # tables: the mean number of attributes queried per row.
        "avg_depth": round(weighted_depth / objects, 2) if objects else 0.0,
        "leaves": leaves,
        "internal_nodes": internal,
    }


def _render(root: TreeNode) -> str:
    lines: list[str] = []

    def walk(node: TreeNode, indent: str) -> None:
        if isinstance(node, LeafNode):
            lines.append(
                f"{indent}--> {node.decision} "
                f"(support {node.support}, confidence {node.confidence})"
            )
            return
        assert isinstance(node, NominalNode)
        for value, child in node.children.items():
            lines.append(f"{indent}{node.attribute} = {value}")
            walk(child, indent + "|   ")

    walk(root, "")
    return "\n".join(lines)


def build_tree(
    df: pd.DataFrame, decision_col: str, max_depth: int | None = None
) -> DecisionTree:
    attributes = [c for c in df.columns if c != decision_col]
    frame = df.copy()
    for col in attributes:
        frame[col] = frame[col].astype(str)
    return DecisionTree(root=_grow(frame, decision_col, attributes, 0, max_depth))


def build(
    df: pd.DataFrame, decision_col: str, max_depth: int | None = None
) -> dict:
    """Return {'stats': {depth, avg_depth, leaves, internal_nodes}, 'text': ...}."""
    if df.empty:
        empty = {"depth": 0, "avg_depth": 0.0, "leaves": 0, "internal_nodes": 0}
        return {"stats": empty, "text": ""}
    tree = build_tree(df, decision_col, max_depth)
    return {"stats": _walk_stats(tree.root), "text": _render(tree.root)}
