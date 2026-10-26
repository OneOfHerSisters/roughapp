"""Rule extraction from decision trees.

Building a tree and turning it into rules are separate responsibilities:
TreeBuilder produces a DecisionTree, a RuleExtractor consumes one.  The
extractor below walks every root-to-leaf path and emits one rule per leaf.

Because nodes branch on nominal attribute values, every edge contributes one
descriptor (a = v) and a root-to-leaf path is exactly a decision rule in the
rough-set sense.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from roughapp.business.models.decision_tree import (
    DecisionTree,
    LeafNode,
    NominalNode,
    TreeNode,
)


@dataclass(frozen=True)
class RuleCondition:
    attribute: str
    operator: str  # "==", "!=", "<=", ">"
    value: object


@dataclass(frozen=True)
class DecisionRule:
    conditions: tuple[RuleCondition, ...]
    conclusion: object
    support: int
    confidence: float


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[DecisionRule, ...]

    def __len__(self) -> int:
        return len(self.rules)


class RuleExtractor(ABC):
    name: str

    @abstractmethod
    def extract(self, tree: DecisionTree) -> RuleSet: ...


class TreePathRuleExtractor(RuleExtractor):
    """One rule per leaf: the conjunction of descriptors along its path."""
    name = "tree-paths"

    def extract(self, tree: DecisionTree) -> RuleSet:
        rules: list[DecisionRule] = []

        def walk(node: TreeNode, conditions: tuple[RuleCondition, ...]) -> None:
            if isinstance(node, LeafNode):
                rules.append(DecisionRule(
                    conditions=conditions,
                    conclusion=node.decision,
                    support=node.support,
                    confidence=node.confidence,
                ))
                return
            assert isinstance(node, NominalNode)
            for value, child in node.children.items():
                walk(child, conditions + (
                    RuleCondition(node.attribute, "==", value),
                ))

        walk(tree.root, ())
        # most reliable rules first
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return RuleSet(rules=tuple(rules))
