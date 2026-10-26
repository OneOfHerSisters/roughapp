"""ExperimentService: create, inspect, export and delete experiments.

Coordinates the runner (pure computation), the repositories (persistence),
and the file storage (result CSVs).  The transaction pattern for create is:
flush to get the experiment id → run strategies → write result files →
insert result rows → one commit; on failure everything is rolled back and
the result directory is removed (the filesystem cannot join the DB
transaction, so compensation is explicit).
"""
from __future__ import annotations

import io

import pandas as pd

from roughapp.business.algorithms import resolution_registry
from roughapp.business.algorithms.base import ExperimentConfiguration
from roughapp.business.exceptions import (
    ExperimentNotFoundError,
    ResultNotFoundError,
    UnknownPolicyError,
    UnknownStrategyError,
    ValidationError,
)
from roughapp.business.details_export import details_csv as build_details_csv
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.ownership import ActorContext
from roughapp.business.policies import OwnerAccessPolicy
from roughapp.business.runner import ExperimentRunner
from roughapp.business.services.dataset_service import DatasetService
from roughapp.db.models import (
    Dataset,
    Experiment,
    ExperimentMetric,
    ExperimentResult,
)
from roughapp.db.repositories import (
    DatasetRepository,
    ExperimentRepository,
    ReferenceRepository,
)
from roughapp.db.storage import FileStorage
from roughapp.db.unit_of_work import UnitOfWork


class ExperimentService:
    def __init__(
        self,
        experiments: ExperimentRepository,
        datasets: DatasetRepository,
        references: ReferenceRepository,
        dataset_service: DatasetService,
        storage: FileStorage,
        owner_policy: OwnerAccessPolicy,
        unit_of_work: UnitOfWork,
        runner: ExperimentRunner | None = None,
    ) -> None:
        self._experiments = experiments
        self._datasets = datasets
        self._references = references
        self._dataset_service = dataset_service
        self._storage = storage
        self._owner_policy = owner_policy
        self._uow = unit_of_work
        self._runner = runner or ExperimentRunner()

    # -- access ---------------------------------------------------------------

    def get(self, actor: ActorContext, experiment_id: int) -> Experiment:
        exp = self._experiments.get(experiment_id)
        if exp is None or not self._owner_policy.can_view(actor, exp):
            raise ExperimentNotFoundError(experiment_id)
        return exp

    def list_for(self, actor: ActorContext) -> list[Experiment]:
        return self._experiments.list_for(actor)

    def list_for_comparison(self, actor: ActorContext) -> list[Experiment]:
        return self._experiments.list_for(actor, newest_first_by="created_at")

    def get_result(self, exp: Experiment, strategy: str) -> ExperimentResult:
        r = next((x for x in exp.results if x.strategy == strategy), None)
        if r is None:
            raise ResultNotFoundError("Strategy result not found")
        return r

    # -- create ---------------------------------------------------------------

    def create(
        self,
        actor: ActorContext,
        dataset_id: int,
        strategies: list[str] | None,
        random_state: int | None,
        tie_policy: str,
        qr_tie_policy: str,
    ) -> Experiment:
        ds = self._dataset_service.get_visible(actor, dataset_id)

        # Validate up front: unknown names would leave an experiment with
        # zero results; duplicates would violate the (experiment, strategy)
        # unique constraint. Deduplicate preserving order.
        if strategies is not None:
            unknown = [s for s in strategies if s not in resolution_registry]
            if unknown:
                raise UnknownStrategyError(unknown)
            strategies = list(dict.fromkeys(strategies))
            if not strategies:
                raise ValidationError("Select at least one strategy")
        else:
            strategies = resolution_registry.names()

        mcd_policy_id = self._references.mcd_policy_id(tie_policy)
        if mcd_policy_id is None:
            raise UnknownPolicyError(f"Unknown MCD tie policy: {tie_policy}")
        qr_policy_id = self._references.qr_policy_id(qr_tie_policy)
        if qr_policy_id is None:
            raise UnknownPolicyError(f"Unknown QR tie policy: {qr_tie_policy}")

        exp = self._experiments.add(Experiment(
            dataset_id=ds.id,
            user_id=actor.user_id,
            guest_session_id=actor.guest_session_id,
            random_state=random_state,
            mcd_tie_policy_id=mcd_policy_id,
            qr_tie_policy_id=qr_policy_id,
        ))
        created_paths: list[str] = []
        try:
            table = self._dataset_service.load_table(ds)
            outcomes = self._runner.run(table, ExperimentConfiguration(
                strategies=tuple(strategies),
                tie_policy=tie_policy,
                qr_tie_policy=qr_tie_policy,
                random_state=random_state,
            ))

            self._sync_dataset_input_metrics(ds, table)

            for outcome in outcomes:
                strategy_id = self._references.strategy_id(outcome.strategy)
                if strategy_id is None:
                    raise UnknownStrategyError([outcome.strategy])
                er = ExperimentResult(
                    strategy_id=strategy_id,
                    time_ms=outcome.time_ms,
                    details=outcome.result.details,
                )
                exp.results.append(er)
                er.metrics = [
                    ExperimentMetric(metric_name=name, value=float(value))
                    for name, value in outcome.result.metrics.items()
                    if value is not None
                ]
                self._uow.flush()
                out_table = table.with_frame(outcome.result.tables[0])
                out_ds = self._dataset_service.register_experiment_result(
                    actor,
                    input_dataset=ds,
                    result=er,
                    table=out_table,
                    experiment_id=exp.id,
                    strategy=outcome.strategy,
                )
                created_paths.append(out_ds.storage_path)
            self._uow.commit()
        except Exception:
            self._uow.rollback()
            for path in created_paths:
                self._storage.delete(path)
            raise
        return exp

    @staticmethod
    def _sync_dataset_input_metrics(ds: Dataset, table: DecisionTable) -> None:
        """Persist γ/α/row count on the dataset when missing or stale."""
        ds.rows = table.row_count
        ds.gamma_actual = round(table.gamma(), 4)
        ds.alpha_actual = round(table.alpha(), 4)

    # -- delete -----------------------------------------------------------------

    def delete(self, actor: ActorContext, experiment_id: int) -> None:
        exp = self.get(actor, experiment_id)
        self._experiments.delete(exp)
        self._uow.commit()

    # -- result access -----------------------------------------------------------

    def preview_result(self, actor: ActorContext, experiment_id: int, strategy: str, offset: int, limit: int) -> dict:
        exp = self.get(actor, experiment_id)
        r = self.get_result(exp, strategy)
        if r.result_path:
            df = self._storage.read_frame(r.result_path)
        else:
            raise ResultNotFoundError("No saved result for this strategy")
        page = df.iloc[offset: offset + limit]
        return {
            "columns": list(df.columns),
            "rows": page.astype(str).values.tolist(),
            "total": len(df),
            "offset": offset,
            "limit": limit,
        }

    def result_download(self, actor: ActorContext, experiment_id: int, strategy: str) -> dict:
        """Payload for the download endpoint: the strategy's single CSV path."""
        exp = self.get(actor, experiment_id)
        r = self.get_result(exp, strategy)
        if r.result_path:
            return {"kind": "csv", "path": r.result_path}
        raise ResultNotFoundError("No saved result for this strategy")

    def summary_rows(self, actor: ActorContext, experiment_id: int) -> list[dict]:
        exp = self.get(actor, experiment_id)
        ds = exp.dataset
        return [{
            "dataset": ds.name if ds else "",
            "strategy": r.strategy,
            "gamma_before": ds.gamma_actual if ds else None,
            "gamma_after": r.gamma_after,
            "alpha_before": ds.alpha_actual if ds else None,
            "alpha_after": r.alpha_after,
            "rows_before": ds.rows if ds else None,
            "rows_after": r.rows_after,
            "rows_removed": self.rows_removed(ds, r),
            "time_ms": r.time_ms,
        } for r in exp.results]

    def summary_csv(self, actor: ActorContext, experiment_id: int) -> str:
        buf = io.StringIO()
        pd.DataFrame(self.summary_rows(actor, experiment_id)).to_csv(buf, index=False)
        return buf.getvalue()

    def details_csv(self, actor: ActorContext, experiment_id: int, strategy: str) -> str:
        exp = self.get(actor, experiment_id)
        self.get_result(exp, strategy)
        ds = exp.dataset
        if ds is None:
            raise ResultNotFoundError("Experiment dataset not found")
        table = self._dataset_service.load_table(ds)
        config = ExperimentConfiguration(
            strategies=(strategy,),
            tie_policy=exp.tie_policy or "min",
            qr_tie_policy=exp.qr_tie_policy or "keep",
            random_state=exp.random_state,
        )
        return build_details_csv(table, strategy, config)

    @staticmethod
    def rows_removed(ds: Dataset | None, r: ExperimentResult) -> int:
        """Derived: rows of the input table minus rows kept after resolution."""
        if ds is None or ds.rows is None or r.rows_after is None:
            return 0
        return ds.rows - r.rows_after
