"""TreeService: decision trees over datasets (including experiment outputs).

Every buildable table is a dataset row. Experiment outputs are materialized
as datasets linked via experiment_results.output_dataset_id + dataset_provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roughapp.business.algorithms import tree_builder_registry
from roughapp.business.algorithms.base import TreeBuilder, TreeBuildResult
from roughapp.business.dataset_provenance import origin_from_dataset, provenance_parent_map
from roughapp.business.models.decision_table import DecisionTable
from roughapp.core.csv_table import read_csv_table
from roughapp.business.exceptions import ResultNotFoundError, ValidationError
from roughapp.business.ownership import ActorContext
from roughapp.business.services.dataset_service import DatasetService
from roughapp.business.services.experiment_service import ExperimentService
from roughapp.db.models import Dataset, ExperimentResult, TreeMetric, TreeRun
from roughapp.db.repositories import ReferenceRepository, TreeRunRepository
from roughapp.db.storage import FileStorage
from roughapp.db.unit_of_work import UnitOfWork

_TREE_METRIC_NAMES = ("depth", "avg_depth", "nodes")


def _params_hash(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stats_payload(stats: dict) -> dict:
    return {k: stats[k] for k in _TREE_METRIC_NAMES if k in stats}


def _build_payload(run: TreeRun, built: TreeBuildResult, **extra) -> dict:
    return {
        "tree_run_id": run.id,
        "text": built.text,
        **_stats_payload(built.stats()),
        **extra,
    }


def _run_metrics(run: TreeRun | None) -> dict:
    if run is None:
        return {}
    return {m.metric_name: m.value for m in run.metrics}


def _owner_match(run: TreeRun, actor: ActorContext) -> bool:
    if actor.user_id is not None:
        return run.user_id == actor.user_id
    if actor.guest_session_id is not None:
        return run.guest_session_id == actor.guest_session_id
    return run.user_id is None and run.guest_session_id is None


def _owned_runs_with_artifact(holder, actor: ActorContext) -> list[TreeRun]:
    return [
        r for r in holder.tree_runs
        if _owner_match(r, actor) and r.artifact_path
    ]


def _built_algorithm_names(holder, actor: ActorContext) -> list[str]:
    names: list[str] = []
    for run in _owned_runs_with_artifact(holder, actor):
        algo = run.algorithm.name if run.algorithm else None
        if algo and algo not in names:
            names.append(algo)
    return names


def _display_run(holder, actor: ActorContext, prefer_algo: str = "id3") -> TreeRun | None:
    runs = _owned_runs_with_artifact(holder, actor)
    for run in runs:
        if run.algorithm and run.algorithm.name == prefer_algo:
            return run
    return runs[0] if runs else None


def _find_owned_run(
    holder,
    actor: ActorContext,
    algorithm_id: int,
    params_hash: str,
) -> TreeRun | None:
    return next(
        (
            r for r in holder.tree_runs
            if r.tree_algorithm_id == algorithm_id
            and r.params_hash == params_hash
            and _owner_match(r, actor)
        ),
        None,
    )


def _source_entry(
    holder,
    actor: ActorContext,
    *,
    source_key: str,
    kind: str,
    label: str,
    table_name: str,
    origin: dict,
    dataset_id: int | None,
    experiment_id: int | None,
    strategy: str | None,
    rows: int | None,
    gamma: float | None,
) -> dict:
    built_algorithms = _built_algorithm_names(holder, actor)
    run = _display_run(holder, actor)
    metrics = _run_metrics(run)
    return {
        "source_key": source_key,
        "kind": kind,
        "label": label,
        "table_name": table_name,
        "origin": origin,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "strategy": strategy,
        "rows": rows,
        "gamma": gamma,
        "built_algorithms": built_algorithms,
        "tree_built": len(built_algorithms) > 0,
        "tree_run_id": run.id if run else None,
        "depth": int(metrics["depth"]) if metrics.get("depth") is not None else None,
        "avg_depth": metrics.get("avg_depth"),
        "nodes": int(metrics["nodes"]) if metrics.get("nodes") is not None else None,
    }


def _dataset_origin(
    ds: Dataset,
    name_by_id: dict[int, str],
    parent_of: dict[int, int | None] | None = None,
) -> dict:
    return origin_from_dataset(ds, name_by_id, parent_of)


class TreeService:
    def __init__(
        self,
        experiments: ExperimentService,
        datasets: DatasetService,
        references: ReferenceRepository,
        tree_runs: TreeRunRepository,
        storage: FileStorage,
        unit_of_work: UnitOfWork,
        builder: TreeBuilder | None = None,
    ) -> None:
        self._experiments = experiments
        self._datasets = datasets
        self._references = references
        self._tree_runs = tree_runs
        self._storage = storage
        self._uow = unit_of_work
        self._builder = builder or tree_builder_registry.create("id3")

    # -- registry -----------------------------------------------------------

    # -- sources & registry ---------------------------------------------------

    def list_tree_sources(self, actor: ActorContext) -> list[dict]:
        """Datasets the actor can induce a tree on."""
        sources: list[dict] = []
        visible = self._datasets.list_visible(actor)
        name_by_id = {ds.id: ds.name for ds in visible}
        parent_of = provenance_parent_map(visible)

        for ds in visible:
            origin = _dataset_origin(ds, name_by_id, parent_of)
            sources.append(_source_entry(
                ds, actor,
                source_key=f"dataset:{ds.id}",
                kind="dataset",
                label=ds.name,
                table_name=ds.name,
                origin=origin,
                dataset_id=ds.id,
                experiment_id=origin.get("experiment_id"),
                strategy=origin.get("strategy"),
                rows=ds.rows,
                gamma=ds.gamma_actual,
            ))

        sources.sort(key=lambda s: s["label"])
        return sources

    def list_tree_runs(self, actor: ActorContext) -> list[dict]:
        visible = self._datasets.list_visible(actor)
        name_by_id = {ds.id: ds.name for ds in visible}
        parent_of = provenance_parent_map(visible)
        rows = []
        for run in self._tree_runs.list_for(actor):
            if run.artifact_path is None or not _owner_match(run, actor):
                continue
            ds = run.dataset
            if ds is None:
                continue
            metrics = {m.metric_name: m.value for m in run.metrics}
            origin = _dataset_origin(ds, name_by_id, parent_of)
            rows.append({
                "tree_run_id": run.id,
                "created_at": run.created_at.isoformat() if run.created_at else "",
                "algorithm": run.algorithm.name if run.algorithm else "",
                "label": ds.name,
                "table_name": ds.name,
                "origin": origin,
                "source": {
                    "kind": "dataset",
                    "dataset_id": ds.id,
                    "dataset_name": ds.name,
                    "strategy": origin.get("strategy"),
                    "experiment_id": origin.get("experiment_id"),
                },
                "rows": ds.rows,
                "gamma": ds.gamma_actual,
                "depth": int(metrics["depth"]) if metrics.get("depth") is not None else None,
                "avg_depth": metrics.get("avg_depth"),
                "nodes": int(metrics["nodes"]) if metrics.get("nodes") is not None else None,
            })
        return rows

    # -- build ----------------------------------------------------------------

    def list_algorithms(self) -> list[dict]:
        return self._references.list_tree_algorithms()

    def _try_build(self, key: str | None, decision_col: str | None) -> TreeBuildResult | None:
        if not key:
            return None
        try:
            df, types = read_csv_table(key)
            dc = decision_col or str(df.columns[-1])
            return self._builder.build(DecisionTable(df, dc, types))
        except Exception:
            return None

    def _result_storage_key(self, r: ExperimentResult) -> str:
        if r.result_path:
            return r.result_path
        raise ResultNotFoundError("No saved result for this strategy")

    def _upsert_run(
        self,
        actor: ActorContext,
        *,
        dataset: Dataset,
        stats: dict,
        artifact_text: str | None = None,
        algorithm_name: str = "id3",
    ) -> TreeRun:
        algorithm_id = self._references.tree_algorithm_id(algorithm_name)
        if algorithm_id is None:
            raise ValidationError(f"Tree algorithm '{algorithm_name}' is not seeded")
        params_hash = _params_hash({})
        run = _find_owned_run(dataset, actor, algorithm_id, params_hash)
        if run is None:
            run = TreeRun(
                dataset_id=dataset.id,
                tree_algorithm_id=algorithm_id,
                params_json={},
                params_hash=params_hash,
                user_id=actor.user_id,
                guest_session_id=actor.guest_session_id,
            )
            dataset.tree_runs.append(run)
            self._uow.flush()
        for metric_name in _TREE_METRIC_NAMES:
            value = stats.get(metric_name)
            existing = next((m for m in run.metrics if m.metric_name == metric_name), None)
            if existing:
                existing.value = value
            elif value is not None:
                run.metrics.append(TreeMetric(metric_name=metric_name, value=float(value)))
        if artifact_text:
            if run.artifact_path:
                self._storage.delete(run.artifact_path)
            key = self._storage.tree_artifact_key(run.id)
            self._storage.save_bytes(artifact_text.encode("utf-8"), key)
            run.artifact_path = key
        return run

    def preview_dataset_tree(
        self, actor: ActorContext, dataset_id: int, max_depth: int | None = None,
    ) -> dict:
        ds = self._datasets.get_visible(actor, dataset_id)
        table = self._datasets.load_table(ds)
        try:
            built = self._builder.build(table, max_depth=max_depth)
        except Exception as e:
            raise ValidationError(str(e))
        return {"stats": built.stats(), "text": built.text}

    def build_dataset_tree(
        self, actor: ActorContext, dataset_id: int, algorithm_name: str = "id3",
    ) -> dict:
        ds = self._datasets.get_visible(actor, dataset_id)
        table = self._datasets.load_table(ds)
        try:
            built = tree_builder_registry.create(algorithm_name).build(table)
        except Exception as e:
            raise ValidationError(str(e))
        run = self._upsert_run(
            actor, dataset=ds, stats=built.stats(), artifact_text=built.text,
            algorithm_name=algorithm_name,
        )
        self._uow.commit()
        return _build_payload(run, built, dataset_id=ds.id, dataset_name=ds.name)

    def build_result_tree(
        self,
        actor: ActorContext,
        experiment_id: int,
        strategy: str,
        algorithm_name: str = "id3",
    ) -> dict:
        exp = self._experiments.get(actor, experiment_id)
        r = self._experiments.get_result(exp, strategy)
        ds = self._datasets.get_output_dataset(actor, r.id)
        built = self.build_dataset_tree(actor, ds.id, algorithm_name=algorithm_name)
        return {
            **built,
            "experiment_id": experiment_id,
            "strategy": strategy,
        }

    def preview_result_tree(
        self, actor: ActorContext, experiment_id: int, strategy: str,
    ) -> dict:
        exp = self._experiments.get(actor, experiment_id)
        r = self._experiments.get_result(exp, strategy)
        decision_col = exp.dataset.decision_col if exp.dataset else ""
        key = self._result_storage_key(r)
        built = self._try_build(key, decision_col or None)
        if not built:
            raise ValidationError(f"Could not induce a tree for strategy '{strategy}'")
        return {**_stats_payload(built.stats()), "text": built.text}

    def build_from_source(self, actor: ActorContext, spec: dict) -> dict:
        kind = spec.get("kind")
        algorithm = spec.get("algorithm") or "id3"
        if algorithm not in tree_builder_registry:
            raise ValidationError(f"Tree algorithm '{algorithm}' is not available")
        if kind == "dataset":
            if not spec.get("dataset_id"):
                raise ValidationError("dataset_id is required")
            return self.build_dataset_tree(
                actor, int(spec["dataset_id"]), algorithm_name=algorithm,
            )
        if kind == "result":
            if not spec.get("experiment_id") or not spec.get("strategy"):
                raise ValidationError("experiment_id and strategy are required")
            exp = self._experiments.get(actor, int(spec["experiment_id"]))
            r = self._experiments.get_result(exp, spec["strategy"])
            ds = self._datasets.get_output_dataset(actor, r.id)
            return self.build_dataset_tree(
                actor, ds.id, algorithm_name=algorithm,
            )
        raise ValidationError("kind must be 'dataset' or 'result'")

    def preview_experiment_tree_metrics(self, actor: ActorContext, experiment_id: int) -> dict:
        exp = self._experiments.get(actor, experiment_id)
        ds = exp.dataset
        decision_col = ds.decision_col if ds else ""
        before_built = self._try_build(ds.storage_path if ds else None, decision_col or None)
        before = before_built.stats() if before_built else {
            "depth": None, "avg_depth": None, "nodes": None,
        }
        strategies: dict[str, dict] = {}
        for r in exp.results:
            key = None
            try:
                key = self._result_storage_key(r)
            except (ResultNotFoundError, ValidationError):
                pass
            built = self._try_build(key, decision_col) if key else None
            strategies[r.strategy] = _stats_payload(built.stats()) if built else {
                "depth": None, "avg_depth": None, "nodes": None,
            }
        return {"before": _stats_payload(before), "strategies": strategies}

    # -- text access ----------------------------------------------------------

    def _assert_run_access(self, actor: ActorContext, run: TreeRun) -> None:
        if not _owner_match(run, actor):
            raise ResultNotFoundError("Saved tree not found")
        if run.dataset_id is None:
            raise ResultNotFoundError("Saved tree not found")
        self._datasets.get_visible(actor, run.dataset_id)

    def get_tree_text(self, actor: ActorContext, tree_run_id: int) -> dict:
        run = self._tree_runs.get(tree_run_id)
        if run is None or not run.artifact_path:
            raise ResultNotFoundError("Saved tree not found")
        self._assert_run_access(actor, run)
        try:
            text = Path(run.artifact_path).read_text(encoding="utf-8")
        except OSError as e:
            raise ResultNotFoundError("Tree file is missing on disk") from e
        return {"tree_run_id": run.id, "text": text}

    def delete_tree_run(self, actor: ActorContext, tree_run_id: int) -> None:
        run = self._tree_runs.get(tree_run_id)
        if run is None:
            raise ResultNotFoundError("Saved tree not found")
        self._assert_run_access(actor, run)
        artifact = run.artifact_path
        self._tree_runs.delete(run)
        self._uow.commit()
        if artifact:
            self._storage.delete(artifact)

    def get_dataset_tree_text(self, actor: ActorContext, dataset_id: int) -> dict:
        ds = self._datasets.get_visible(actor, dataset_id)
        id3_id = self._references.tree_algorithm_id("id3")
        run = _find_owned_run(ds, actor, id3_id, _params_hash({})) if id3_id else None
        if run is None or not run.artifact_path:
            raise ResultNotFoundError("No tree built for this dataset")
        return self.get_tree_text(actor, run.id)

    def get_result_tree_text(
        self, actor: ActorContext, experiment_id: int, strategy: str,
    ) -> dict:
        exp = self._experiments.get(actor, experiment_id)
        r = self._experiments.get_result(exp, strategy)
        ds = self._datasets.get_output_dataset(actor, r.id)
        return self.get_dataset_tree_text(actor, ds.id)

    def dataset_tree_metric(self, ds: Dataset | None, metric_name: str):
        if not ds or not ds.tree_runs:
            return None
        run = next((r for r in ds.tree_runs if r.artifact_path), None)
        if run is None:
            return None
        value = next((m.value for m in run.metrics if m.metric_name == metric_name), None)
        if value is None:
            return None
        return int(value) if metric_name in {"depth", "nodes"} else value

    save_dataset_tree = build_dataset_tree
    save_experiment_result_tree = build_result_tree
