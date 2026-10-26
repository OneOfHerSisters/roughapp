"""Algorithm families and their registries.

Every registration is explicit and lives here — the single place to look
when adding a new strategy (see docs/architecture.md).
"""
from roughapp.business.registries import Registry
from roughapp.business.algorithms.base import (
    CompletionResult,
    CompletionStrategy,
    IntroductionResult,
    IntroductionStrategy,
    ResolutionResult,
    ResolutionStrategy,
    TreeBuilder,
    TreeBuildResult,
)
from roughapp.business.algorithms import completion, introduction, resolution, rules, trees
from roughapp.business.algorithms.rules import RuleExtractor, RuleSet

resolution_registry: Registry[ResolutionStrategy] = Registry("resolution")
resolution_registry.register(resolution.MostCommonDecisionStrategy)
resolution_registry.register(resolution.RoughApproximationStrategy)
resolution_registry.register(resolution.GeneralizedDecisionStrategy)

introduction_registry: Registry[IntroductionStrategy] = Registry("introduction")
introduction_registry.register(introduction.IntroduceUntilQualityStrategy)
introduction_registry.register(introduction.DuplicateFlipStrategy)
introduction_registry.register(introduction.DropAttributesStrategy)

completion_registry: Registry[CompletionStrategy] = Registry("completion")
completion_registry.register(completion.RemoveIncompleteStrategy)
completion_registry.register(completion.FillValuesStrategy)

tree_builder_registry: Registry[TreeBuilder] = Registry("tree-builder")
tree_builder_registry.register(trees.Id3TreeBuilder)

rule_extractor_registry: Registry[RuleExtractor] = Registry("rule-extractor")
rule_extractor_registry.register(rules.TreePathRuleExtractor)

__all__ = [
    "resolution_registry",
    "introduction_registry",
    "completion_registry",
    "tree_builder_registry",
    "rule_extractor_registry",
    "RuleExtractor",
    "RuleSet",
    "ResolutionStrategy",
    "ResolutionResult",
    "IntroductionStrategy",
    "IntroductionResult",
    "CompletionStrategy",
    "CompletionResult",
    "TreeBuilder",
    "TreeBuildResult",
]
