"""Unit tests for the business algorithm layer (no DB, no HTTP)."""
import pandas as pd
import pytest

from roughapp.business.algorithms import (
    completion_registry,
    introduction_registry,
    resolution_registry,
    tree_builder_registry,
)
from roughapp.business.algorithms.base import ExperimentConfiguration
from roughapp.business.exceptions import (
    DecisionColumnNotFoundError,
    EmptyDecisionTableError,
    UnknownStrategyError,
    ValidationError,
)
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.registries import Registry, StrategyAlreadyRegisteredError
from roughapp.business.runner import ExperimentRunner


@pytest.fixture
def table(table1) -> DecisionTable:
    return DecisionTable(table1, "dec")


class TestDecisionTable:
    def test_validates_empty(self):
        with pytest.raises(EmptyDecisionTableError):
            DecisionTable(pd.DataFrame({"a": []}), "a")

    def test_validates_decision_column(self, table1):
        with pytest.raises(DecisionColumnNotFoundError):
            DecisionTable(table1, "nope")

    def test_internal_frame_protected_from_source_mutation(self, table1):
        t = DecisionTable(table1, "dec")
        table1.loc[0, "a"] = 99
        assert t.frame.loc[0, "a"] == 0

    def test_to_dataframe_returns_copy(self, table):
        df = table.to_dataframe()
        df.loc[0, "a"] = 99
        assert table.frame.loc[0, "a"] == 0

    def test_properties(self, table):
        assert table.row_count == 9
        assert table.column_count == 5
        assert table.condition_columns == ["a", "b", "c", "d"]


class TestRegistry:
    def test_rejects_duplicates(self):
        reg = Registry("test")

        class A:
            name = "a"

        reg.register(A)
        with pytest.raises(StrategyAlreadyRegisteredError):
            reg.register(A)

    def test_unknown_name(self):
        reg = Registry("test")
        with pytest.raises(UnknownStrategyError):
            reg.create("nope")

    def test_all_families_registered(self):
        assert set(resolution_registry.names()) == {
            "mcd", "rough-approximation", "gd",
        }
        assert set(introduction_registry.names()) == {
            "until_quality", "duplicate_flip", "drop_attributes",
        }
        assert set(completion_registry.names()) == {"remove", "fill"}
        assert "id3" in tree_builder_registry.names()


class TestResolutionStrategies:
    """γ/α on Table 1 from the thesis: γ = 7/9, conflict class {x3, x9}."""

    def test_polymorphic_execution(self, table):
        # every strategy runs through the same base-type call
        for name in resolution_registry.names():
            strategy = resolution_registry.get_class(name).from_config(
                ExperimentConfiguration(strategies=(name,))
            )
            result = strategy.execute(table)
            assert result.tables, name
            assert "rows" in result.metrics, name

    def test_mcd_metrics_and_details(self, table):
        r = resolution_registry.create("mcd", tie_policy="min").execute(table)
        assert r.metrics["gamma"] == 1.0
        assert r.metrics["rows"] == 8  # one of the two conflicting rows removed
        assert r.details["conflict_classes"][0]["counts"] == {"0": 1, "2": 1}


    def test_gd_counts_codes(self, table):
        r = resolution_registry.create("gd").execute(table)
        assert r.metrics["n_gd_codes"] >= r.metrics["n_gd_conflict_codes"] >= 1


class TestIntroductionStrategies:
    def test_until_quality_rejects_high_target(self, table):
        strat = introduction_registry.create("until_quality", target_gamma=0.9)
        with pytest.raises(ValidationError):
            strat.execute(table)

    def test_until_quality(self, table):
        strat = introduction_registry.create("until_quality", target_gamma=0.5, random_state=42)
        res = strat.execute(table)
        assert res.gamma_actual <= 0.5
        assert res.suffix == "g0.5"

    def test_drop_attributes_requires_attrs(self):
        with pytest.raises(ValidationError):
            introduction_registry.create("drop_attributes", attrs=[])


class TestCompletionStrategies:
    def test_requires_missing_values(self, table):
        with pytest.raises(ValidationError):
            completion_registry.create("fill").execute(table)

    def test_fill(self):
        df = pd.DataFrame({"a": ["1", "?", "1"], "dec": ["P", "P", "N"]})
        res = completion_registry.create("fill").execute(DecisionTable(df, "dec"))
        assert res.suffix == "filled"
        assert (res.table.frame["a"] == "?").sum() == 0


class TestTreeBuilder:
    def test_id3_stats(self, table):
        result = tree_builder_registry.create("id3").build(table)
        assert result.depth >= 1
        assert result.nodes == result.leaves + result.internal_nodes
        assert "-->" in result.text


class TestExperimentRunner:
    def test_runner_rejects_unknown(self, table):
        with pytest.raises(UnknownStrategyError):
            ExperimentRunner().run(table, ExperimentConfiguration(strategies=("greedy",)))

    def test_runner_all_strategies(self, table):
        config = ExperimentConfiguration(
            strategies=("mcd", "rough-approximation", "gd"),
            random_state=42,
        )
        outcomes = ExperimentRunner().run(table, config)
        assert [o.strategy for o in outcomes] == list(config.strategies)
        assert all(o.time_ms >= 0 for o in outcomes)
        gd = next(o for o in outcomes if o.strategy == "gd")
        assert gd.result.metrics["gamma"] == 1.0
