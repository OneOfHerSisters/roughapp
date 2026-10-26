"""Repositories: every SQLAlchemy query of the application lives here.

Business services receive repositories as constructor arguments and depend
only on their method signatures; the ORM models returned are used as the
domain objects directly (they already carry the derived properties such as
ExperimentResult.gamma_after).  Repositories never call commit() — the
transaction boundary belongs to the service via UnitOfWork.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, joinedload

from roughapp.business.dataset_provenance import KIND_DERIVED
from roughapp.business.ownership import ActorContext
from roughapp.db.models import (
    Dataset,
    DatasetProvenance,
    Experiment,
    ExperimentResult,
    McdTiePolicy,
    QrTiePolicy,
    Strategy,
    TreeAlgorithm,
    TreeRun,
)

_EXPERIMENT_LOAD = (
    joinedload(Experiment.dataset),
    joinedload(Experiment.mcd_policy),
    joinedload(Experiment.qr_policy),
    joinedload(Experiment.results).joinedload(ExperimentResult.output_dataset),
    joinedload(Experiment.results).joinedload(ExperimentResult.strategy_ref),
)


class DatasetRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, dataset_id: int) -> Dataset | None:
        return self._session.get(Dataset, dataset_id)

    def add(self, dataset: Dataset) -> Dataset:
        self._session.add(dataset)
        self._session.flush()  # assign the id (needed for the storage key)
        return dataset

    def delete(self, dataset: Dataset) -> None:
        self._session.delete(dataset)

    def list_visible(self, actor: ActorContext) -> list[Dataset]:
        """Builtins plus the actor's own datasets, in id order."""
        load = (
            joinedload(Dataset.provenance).joinedload(DatasetProvenance.source),
            joinedload(Dataset.produced_by_result)
            .joinedload(ExperimentResult.experiment),
            joinedload(Dataset.produced_by_result)
            .joinedload(ExperimentResult.strategy_ref),
        )
        q = self._session.query(Dataset).options(*load).order_by(Dataset.id)
        builtin = Dataset.user_id.is_(None) & Dataset.guest_session_id.is_(None)
        if actor.user_id is not None:
            return q.filter(builtin | (Dataset.user_id == actor.user_id)).all()
        if actor.guest_session_id is not None:
            return q.filter(builtin | (Dataset.guest_session_id == actor.guest_session_id)).all()
        return q.filter(builtin).all()

    def has_experiments(self, dataset_id: int) -> bool:
        return (
            self._session.query(Experiment).filter_by(dataset_id=dataset_id).first()
            is not None
        )

    def experiments_using(self, dataset_id: int) -> list[Experiment]:
        """Experiments whose *input* is this dataset (they cannot outlive it)."""
        return (
            self._session.query(Experiment)
            .filter_by(dataset_id=dataset_id)
            .order_by(Experiment.id)
            .all()
        )

    def delete_experiment(self, experiment: Experiment) -> None:
        """Delete a run record together with the dataset it depends on.

        ORM cascades take the results and metrics; the output datasets
        themselves are independent data and survive (FK is SET NULL).
        """
        self._session.delete(experiment)

    def find_derived_by_name(
        self, actor: ActorContext, source_dataset_id: int, name: str,
    ) -> Dataset | None:
        return (
            self._session.query(Dataset)
            .join(DatasetProvenance, Dataset.id == DatasetProvenance.dataset_id)
            .filter(
                Dataset.user_id == actor.user_id,
                Dataset.guest_session_id == actor.guest_session_id,
                DatasetProvenance.kind == KIND_DERIVED,
                DatasetProvenance.source_dataset_id == source_dataset_id,
                Dataset.name == name,
            )
            .first()
        )

    def find_output_dataset(self, result_id: int) -> Dataset | None:
        result = (
            self._session.query(ExperimentResult)
            .options(
                joinedload(ExperimentResult.output_dataset).joinedload(Dataset.provenance),
            )
            .filter_by(id=result_id)
            .first()
        )
        return result.output_dataset if result else None

    def derived_child_ids(self, root_id: int) -> list[int]:
        """Datasets produced as derived copies from *root_id* (transitive, excludes root)."""
        seen: set[int] = set()
        frontier = {root_id}
        while frontier:
            current = frontier.pop()
            child_ids = [
                row[0]
                for row in self._session.query(DatasetProvenance.dataset_id)
                .filter(
                    DatasetProvenance.source_dataset_id == current,
                    DatasetProvenance.kind == KIND_DERIVED,
                )
                .all()
            ]
            for child_id in child_ids:
                if child_id not in seen:
                    seen.add(child_id)
                    frontier.add(child_id)
        return sorted(seen)

    def deletion_plan(self, dataset_id: int) -> "DatasetDeletionPlan":
        """Files to remove when THIS dataset goes.

        Derived datasets deliberately survive their source (their provenance
        link is severed by ON DELETE SET NULL), so the plan covers only the
        dataset's own CSV and its tree artifacts.
        """
        storage_paths: list[str] = []
        ds = self._session.get(Dataset, dataset_id)
        if ds and ds.storage_path:
            storage_paths.append(ds.storage_path)
        tree_artifacts = [
            row[0]
            for row in self._session.query(TreeRun.artifact_path)
            .filter(TreeRun.dataset_id == dataset_id, TreeRun.artifact_path.isnot(None))
            .all()
            if row[0]
        ]
        return DatasetDeletionPlan(
            storage_paths=storage_paths,
            tree_artifacts=tree_artifacts,
        )


@dataclass(frozen=True)
class DatasetDeletionPlan:
    storage_paths: list[str]
    tree_artifacts: list[str]


def apply_dataset_deletion_plan(plan: DatasetDeletionPlan, storage) -> None:
    """Remove on-disk files listed in a dataset deletion plan (best-effort)."""
    for path in plan.storage_paths:
        storage.delete(path)
    for artifact in plan.tree_artifacts:
        storage.delete(artifact)


class ExperimentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, experiment_id: int) -> Experiment | None:
        return (
            self._session.query(Experiment)
            .options(*_EXPERIMENT_LOAD)
            .filter_by(id=experiment_id)
            .first()
        )

    def add(self, experiment: Experiment) -> Experiment:
        self._session.add(experiment)
        self._session.flush()
        return experiment

    def delete(self, experiment: Experiment) -> None:
        self._session.delete(experiment)

    def list_for(self, actor: ActorContext, newest_first_by: str = "id") -> list[Experiment]:
        """Experiments owned by the actor (independent of dataset visibility)."""
        if actor.user_id is not None:
            owner_filter = Experiment.user_id == actor.user_id
        elif actor.guest_session_id is not None:
            owner_filter = Experiment.guest_session_id == actor.guest_session_id
        else:
            return []
        order = Experiment.id.desc() if newest_first_by == "id" else Experiment.created_at.desc()
        return (
            self._session.query(Experiment)
            .options(*_EXPERIMENT_LOAD)
            .filter(owner_filter)
            .order_by(order)
            .all()
        )


class TreeRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, tree_run_id: int) -> "TreeRun | None":
        from roughapp.db.models import TreeRun

        return self._session.get(TreeRun, tree_run_id)

    def delete(self, run: "TreeRun") -> None:
        self._session.delete(run)

    def list_for(self, actor: ActorContext) -> list["TreeRun"]:
        """Tree runs the actor may see, newest first."""
        from roughapp.db.models import TreeRun

        load = (
            joinedload(TreeRun.dataset)
            .joinedload(Dataset.provenance),
            joinedload(TreeRun.dataset)
            .joinedload(Dataset.produced_by_result)
            .joinedload(ExperimentResult.experiment),
            joinedload(TreeRun.algorithm),
        )

        builtin = Dataset.user_id.is_(None) & Dataset.guest_session_id.is_(None)
        if actor.user_id is not None:
            own_dataset = Dataset.user_id == actor.user_id
        elif actor.guest_session_id is not None:
            own_dataset = Dataset.guest_session_id == actor.guest_session_id
        else:
            own_dataset = None

        q = (
            self._session.query(TreeRun)
            .options(*load)
            .join(Dataset, TreeRun.dataset_id == Dataset.id)
        )
        if own_dataset is None:
            q = q.filter(builtin)
        else:
            q = q.filter(builtin | own_dataset)

        runs = q.all()
        if actor.user_id is not None:
            runs = [r for r in runs if r.user_id == actor.user_id]
        elif actor.guest_session_id is not None:
            runs = [r for r in runs if r.guest_session_id == actor.guest_session_id]
        else:
            runs = [r for r in runs if r.user_id is None and r.guest_session_id is None]
        runs.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        return runs


class ReferenceRepository:
    """Read access to the lookup tables (strategies, policies, algorithms)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def strategy_id(self, name: str) -> int | None:
        row = self._session.query(Strategy).filter_by(name=name).first()
        return row.id if row else None

    def mcd_policy_id(self, name: str) -> int | None:
        row = self._session.query(McdTiePolicy).filter_by(name=name).first()
        return row.id if row else None

    def qr_policy_id(self, name: str) -> int | None:
        row = self._session.query(QrTiePolicy).filter_by(name=name).first()
        return row.id if row else None

    def tree_algorithm_id(self, name: str) -> int | None:
        row = self._session.query(TreeAlgorithm).filter_by(name=name).first()
        return row.id if row else None

    def list_tree_algorithms(self) -> list[dict]:
        rows = self._session.query(TreeAlgorithm).order_by(TreeAlgorithm.name).all()
        return [{"name": r.name, "description": r.description} for r in rows]
