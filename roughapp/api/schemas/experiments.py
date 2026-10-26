"""Request/response schemas of the experiments router.

The response models still expose gamma_before / gamma_after-style fields,
assembled from the dataset record, experiment_metrics, and tree_metrics —
the storage layout changed, the API contract did not.
"""
from typing import List, Optional

from pydantic import BaseModel

from roughapp.api.schemas.datasets import _tree_metric
from roughapp.db.models import Dataset, Experiment, ExperimentResult


class ExperimentRequest(BaseModel):
    dataset_id: int
    strategies: Optional[List[str]] = None
    random_state: Optional[int] = None
    tie_policy: str = "min"           # MCD: min|keep|first|random|raise
    qr_tie_policy: str = "keep"       # rough-approximation: keep|random|raise


class ResultOut(BaseModel):
    strategy: str
    gamma_after: Optional[float]
    alpha_after: Optional[float]
    rows_after: Optional[int]
    rows_removed: int
    depth_after: Optional[int] = None
    avg_depth_after: Optional[float] = None
    nodes_after: Optional[int] = None
    n_gd_codes: Optional[int]
    n_gd_conflict_codes: Optional[int] = None
    time_ms: int
    details: Optional[dict] = None
    has_result: bool = False

    @classmethod
    def from_model(cls, r: ExperimentResult, ds: Dataset | None) -> "ResultOut":
        rows_removed = 0
        if ds is not None and ds.rows is not None and r.rows_after is not None:
            rows_removed = ds.rows - r.rows_after
        return cls(
            strategy=r.strategy,
            gamma_after=r.gamma_after,
            alpha_after=r.alpha_after,
            rows_after=r.rows_after,
            rows_removed=rows_removed,
            depth_after=r.depth_after,
            avg_depth_after=r.avg_depth_after,
            nodes_after=r.nodes_after,
            n_gd_codes=r.n_gd_codes,
            n_gd_conflict_codes=r.n_gd_conflict_codes,
            time_ms=r.time_ms,
            details=r.details,
            has_result=bool(r.result_path),
        )


def dataset_before_fields(ds: Dataset | None) -> dict:
    """Input-table metrics live on the dataset, not duplicated on experiments."""
    if not ds:
        return {
            "gamma_before": None, "alpha_before": None, "rows_before": None,
            "depth_before": None, "avg_depth_before": None, "nodes_before": None,
        }
    return {
        "gamma_before": ds.gamma_actual,
        "alpha_before": ds.alpha_actual,
        "rows_before": ds.rows,
        "depth_before": _tree_metric(ds, "depth"),
        "avg_depth_before": _tree_metric(ds, "avg_depth"),
        "nodes_before": _tree_metric(ds, "nodes"),
    }


class ExperimentOut(BaseModel):
    id: int
    dataset_id: int
    dataset_name: str
    random_state: Optional[int]
    tie_policy: Optional[str] = None
    qr_tie_policy: Optional[str] = None
    created_at: str
    gamma_before: Optional[float] = None
    alpha_before: Optional[float] = None
    rows_before: Optional[int] = None
    depth_before: Optional[int] = None
    avg_depth_before: Optional[float] = None
    nodes_before: Optional[int] = None
    results: List[ResultOut]

    @classmethod
    def from_model(cls, exp: Experiment) -> "ExperimentOut":
        return cls(
            id=exp.id,
            dataset_id=exp.dataset_id,
            dataset_name=exp.dataset.name if exp.dataset else "",
            random_state=exp.random_state,
            tie_policy=exp.tie_policy,
            qr_tie_policy=exp.qr_tie_policy,
            created_at=exp.created_at.isoformat() if exp.created_at else "",
            **dataset_before_fields(exp.dataset),
            results=[ResultOut.from_model(r, exp.dataset) for r in exp.results],
        )


class TreeMetricsOut(BaseModel):
    before: dict
    strategies: dict[str, dict]


def comparison_row(exp: Experiment) -> dict:
    before = dataset_before_fields(exp.dataset)
    return {
        "experiment_id": exp.id,
        "dataset_name": exp.dataset.name if exp.dataset else "",
        "created_at": exp.created_at.isoformat() if exp.created_at else "",
        "random_state": exp.random_state,
        "tie_policy": exp.tie_policy,
        "qr_tie_policy": exp.qr_tie_policy,
        "strategies_run": [r.strategy for r in exp.results],
        "before": {
            "depth": before["depth_before"],
            "avg_depth": before["avg_depth_before"],
            "nodes": before["nodes_before"],
            "gamma": before["gamma_before"],
            "alpha": before["alpha_before"],
            "rows": before["rows_before"],
        },
        "strategies": {
            r.strategy: {
                "depth_after": r.depth_after,
                "avg_depth_after": r.avg_depth_after,
                "nodes_after": r.nodes_after,
                "gamma_after": r.gamma_after,
                "rows_after": r.rows_after,
            }
            for r in exp.results
        },
    }
