"""DatasetService: every user scenario that starts from a dataset."""
from __future__ import annotations

from pathlib import Path

from roughapp.business.algorithms import completion_registry, introduction_registry
from roughapp.business.dataset_naming import format_experiment_result_name
from roughapp.business.dataset_provenance import (
    KIND_DERIVED,
    KIND_EXPERIMENT_OUTPUT,
    KIND_UPLOAD,
)
from roughapp.business.exceptions import (
    CascadeRequiredError,
    DatasetNotFoundError,
    DecisionColumnNotFoundError,
    ValidationError,
)
from roughapp.business.models.decision_table import DecisionTable
from roughapp.business.ownership import ActorContext
from roughapp.business.policies import DatasetAccessPolicy
from roughapp.core.missing import missing_summary
from roughapp.db.models import Dataset, DatasetProvenance, ExperimentResult
from roughapp.db.repositories import DatasetRepository, ExperimentRepository, apply_dataset_deletion_plan
from roughapp.db.storage import FileStorage
from roughapp.db.unit_of_work import UnitOfWork
from roughapp.business.analysis import find_inconsistencies


class DatasetService:
    def __init__(
        self,
        repository: DatasetRepository,
        experiments: ExperimentRepository,
        storage: FileStorage,
        access_policy: DatasetAccessPolicy,
        unit_of_work: UnitOfWork,
    ) -> None:
        self._repository = repository
        self._experiments = experiments
        self._storage = storage
        self._policy = access_policy
        self._uow = unit_of_work

    def list_visible(self, actor: ActorContext) -> list[Dataset]:
        return self._repository.list_visible(actor)

    def get_visible(self, actor: ActorContext, dataset_id: int) -> Dataset:
        ds = self._repository.get(dataset_id)
        if ds is None:
            raise DatasetNotFoundError(dataset_id)
        self._policy.require_view(actor, ds)
        return ds

    def get_output_dataset(
        self,
        actor: ActorContext,
        result_id: int,
    ) -> Dataset:
        ds = self._repository.find_output_dataset(result_id)
        if ds is None:
            raise DatasetNotFoundError(result_id)
        self._policy.require_view(actor, ds)
        return ds

    def load_table(self, dataset: Dataset) -> DecisionTable:
        return self._storage.read_table(dataset.storage_path, dataset.decision_col)

    def preview(self, actor: ActorContext, dataset_id: int, offset: int, limit: int) -> dict:
        ds = self.get_visible(actor, dataset_id)
        df = self._storage.read_frame(ds.storage_path)
        page = df.iloc[offset: offset + limit]
        return {
            "columns": list(df.columns),
            "rows": page.astype(str).values.tolist(),
            "total": len(df),
            "offset": offset,
            "limit": limit,
        }

    def analyze(self, actor: ActorContext, dataset_id: int, decision_col: str | None = None) -> dict:
        ds = self.get_visible(actor, dataset_id)
        df = self._storage.read_frame(ds.storage_path)
        col = decision_col or ds.decision_col
        try:
            result = find_inconsistencies(df, col)
        except ValueError as e:
            raise ValidationError(str(e))
        per_decision = result["per_decision"]
        return {
            "decision_col": col,
            "inconsistent_class_count": result["inconsistent_class_count"],
            "inconsistent_total_rows": result["inconsistent_total_rows"],
            "duplicate_rows": result["duplicate_rows"],
            "missing": missing_summary(df),
            "accuracy": result["accuracy"],
            "quality": result["quality"],
            "per_decision": per_decision.to_dict(orient="records")
            if hasattr(per_decision, "to_dict") else per_decision,
        }

    def remove_duplicates(self, actor: ActorContext, dataset_id: int) -> Dataset:
        """Drop exact duplicate rows and save the result as a new dataset.

        A duplicate here is a whole row repeated (identical conditions and the
        same decision), so removing it changes neither the conflict structure
        nor $\\gamma$; it only removes redundancy. Mirrors the introduce and
        complete flows: the source dataset is never modified.
        """
        ds = self.get_visible(actor, dataset_id)
        table = self.load_table(ds)
        deduped = table.frame.drop_duplicates().reset_index(drop=True)
        if len(deduped) == len(table.frame):
            raise ValidationError("Dataset has no duplicate rows to remove")
        new_table = table.with_frame(deduped)
        return self.save_derived(
            actor, source=ds, table=new_table,
            gamma_actual=new_table.gamma(), suffix="deduped",
        )

    def attribute_importance(self, actor: ActorContext, dataset_id: int) -> dict:
        ds = self.get_visible(actor, dataset_id)
        try:
            table = self.load_table(ds)
            gamma_full = table.gamma()
            result = []
            for col in table.condition_columns:
                reduced = table.with_frame(table.frame.drop(columns=[col]))
                gamma_reduced = reduced.gamma()
                result.append({
                    "attribute": col,
                    "importance": round(gamma_full - gamma_reduced, 4),
                    "gamma_without": round(gamma_reduced, 4),
                })
            result.sort(key=lambda x: x["importance"], reverse=True)
            return {"gamma_full": round(gamma_full, 4), "attributes": result}
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(str(e))

    def missing_summary(self, actor: ActorContext, dataset_id: int) -> dict:
        ds = self.get_visible(actor, dataset_id)
        df = self._storage.read_frame(ds.storage_path)
        return missing_summary(df)

    def upload(self, actor: ActorContext, filename: str, content: bytes, decision_col: str) -> Dataset:
        if not filename or not filename.endswith(".csv"):
            raise ValidationError("Only CSV files are accepted")
        try:
            df = self._read_uploaded(content)
        except Exception:
            raise ValidationError("Could not parse CSV")
        if decision_col not in df.columns:
            raise DecisionColumnNotFoundError(decision_col)
        table = DecisionTable(df, decision_col)

        ds = Dataset(
            name=filename,
            storage_path="",
            rows=table.row_count,
            cols=table.column_count,
            decision_col=decision_col,
            gamma_actual=round(table.gamma(), 4),
            alpha_actual=round(table.alpha(), 4),
            user_id=actor.user_id,
            guest_session_id=actor.guest_session_id,
        )
        self._repository.add(ds)
        key = self._storage.upload_key(ds.id, filename)
        self._storage.save_bytes(content, key)
        ds.storage_path = key
        ds.provenance = DatasetProvenance(kind=KIND_UPLOAD)
        try:
            self._uow.commit()
        except Exception:
            self._uow.rollback()
            self._storage.delete(key)
            raise
        return ds

    def delete(self, actor: ActorContext, dataset_id: int, cascade: bool = False) -> None:
        """Delete a dataset the actor owns.

        Deletion follows the provenance graph in one direction only: tables
        derived from this one are independent data and always survive (their
        provenance link is severed). Experiments that used this dataset as
        input cannot outlive it; without ``cascade`` their list is returned
        in a 409 manifest so the client can confirm, with ``cascade`` they
        are deleted together with the dataset. Their output tables are data
        and survive as well.
        """
        ds = self._repository.get(dataset_id)
        if ds is None:
            raise DatasetNotFoundError(dataset_id)
        self._policy.require_delete(actor, ds)

        experiments = self._repository.experiments_using(dataset_id)
        if experiments and not cascade:
            raise CascadeRequiredError(
                "Dataset is used as the input of experiments; deleting it "
                "deletes those experiment records as well (their result "
                "tables are kept).",
                manifest={
                    "experiments": [
                        {
                            "id": e.id,
                            "created_at": e.created_at.isoformat() if e.created_at else "",
                            "strategies": [r.strategy for r in e.results],
                        }
                        for e in experiments
                    ],
                },
            )

        plan = self._repository.deletion_plan(dataset_id)
        for e in experiments:
            self._repository.delete_experiment(e)
        self._repository.delete(ds)
        self._uow.commit()

        apply_dataset_deletion_plan(plan, self._storage)

    def introduce(
        self,
        actor: ActorContext,
        dataset_id: int,
        method: str,
        *,
        target_gamma: float | None = None,
        k: int | None = None,
        attrs: list[str] | None = None,
        random_state: int | None = None,
    ) -> Dataset:
        ds = self.get_visible(actor, dataset_id)
        strategy = self._build_introduction(method, target_gamma, k, attrs, random_state)
        result = strategy.execute(self.load_table(ds))
        return self.save_derived(
            actor, source=ds, table=result.table,
            gamma_actual=result.gamma_actual, suffix=result.suffix,
        )

    def complete(self, actor: ActorContext, dataset_id: int, method: str, by_class: bool = False) -> Dataset:
        ds = self.get_visible(actor, dataset_id)
        if method == "fill":
            strategy = completion_registry.create("fill", by_class=by_class)
        elif method == "remove":
            strategy = completion_registry.create("remove")
        else:
            raise ValidationError(f"Unknown method: {method}")
        result = strategy.execute(self.load_table(ds))
        return self.save_derived(
            actor, source=ds, table=result.table,
            gamma_actual=result.gamma_actual, suffix=result.suffix,
        )

    def save_derived(
        self,
        actor: ActorContext,
        *,
        source: Dataset,
        table: DecisionTable,
        gamma_actual: float | None,
        suffix: str,
        alpha_actual: float | None = None,
        display_name: str | None = None,
    ) -> Dataset:
        new_name = f"{Path(source.storage_path).stem}_{suffix}.csv"
        if alpha_actual is None and gamma_actual is not None:
            alpha_actual = round(table.alpha(), 4)

        new_ds = Dataset(
            name=display_name or new_name,
            storage_path="",
            rows=table.row_count,
            cols=table.column_count,
            decision_col=source.decision_col,
            gamma_actual=round(gamma_actual, 4) if gamma_actual is not None else None,
            alpha_actual=alpha_actual,
            user_id=actor.user_id,
            guest_session_id=actor.guest_session_id,
        )
        self._repository.add(new_ds)
        key = self._storage.upload_key(new_ds.id, new_name)
        self._storage.save_table(table, key)
        new_ds.storage_path = key
        new_ds.provenance = DatasetProvenance(
            kind=KIND_DERIVED,
            source_dataset_id=source.id,
        )
        try:
            self._uow.commit()
        except Exception:
            self._uow.rollback()
            self._storage.delete(key)
            raise
        return new_ds

    def register_experiment_result(
        self,
        actor: ActorContext,
        *,
        input_dataset: Dataset,
        result: ExperimentResult,
        table: DecisionTable,
        experiment_id: int,
        strategy: str,
    ) -> Dataset:
        """One dataset row + one CSV for the strategy's output table."""
        existing = self._repository.find_output_dataset(result.id)
        if existing is not None:
            return existing

        name = format_experiment_result_name(
            input_dataset.name,
            strategy,
            experiment_id,
        )
        stem = Path(input_dataset.name).stem
        file_name = f"{stem}_exp{experiment_id}_{strategy}.csv"

        ds = Dataset(
            name=name,
            storage_path="",
            rows=table.row_count,
            cols=table.column_count,
            decision_col=input_dataset.decision_col,
            gamma_actual=round(table.gamma(), 4),
            alpha_actual=round(table.alpha(), 4),
            user_id=actor.user_id,
            guest_session_id=actor.guest_session_id,
        )
        self._repository.add(ds)
        key = self._storage.upload_key(ds.id, file_name)
        self._storage.save_table(table, key)
        ds.storage_path = key
        ds.provenance = DatasetProvenance(
            kind=KIND_EXPERIMENT_OUTPUT,
            source_dataset_id=input_dataset.id,
        )
        result.output_dataset = ds
        return ds

    @staticmethod
    def _read_uploaded(content: bytes):
        from roughapp.core.csv_table import read_csv
        return read_csv(content)

    @staticmethod
    def _build_introduction(method, target_gamma, k, attrs, random_state):
        if method == "until_quality":
            return introduction_registry.create(
                "until_quality", target_gamma=target_gamma, random_state=random_state
            )
        if method == "duplicate_flip":
            return introduction_registry.create(
                "duplicate_flip", k=k or 1, random_state=random_state
            )
        if method == "drop_attributes":
            return introduction_registry.create("drop_attributes", attrs=attrs or [])
        raise ValidationError(f"Unknown method: {method}")
