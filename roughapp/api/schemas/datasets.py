"""Request/response schemas of the datasets router."""
from typing import List, Optional

from pydantic import BaseModel

from roughapp.business.dataset_provenance import provenance_out
from roughapp.db.models import Dataset


def _tree_metric(ds: Dataset, metric_name: str):
    """Saved tree stats only (artifact_path set)."""
    run = ds.tree_runs[0] if ds.tree_runs else None
    if not run or not run.artifact_path:
        return None
    value = next((m.value for m in run.metrics if m.metric_name == metric_name), None)
    if value is None:
        return None
    return int(value) if metric_name in {"depth", "nodes"} else value


class ProvenanceOut(BaseModel):
    kind: str
    source_dataset_id: Optional[int] = None


class DatasetOut(BaseModel):
    id: int
    name: str
    rows: int
    cols: int
    decision_col: str
    created_at: str
    storage_path: str
    provenance: Optional[ProvenanceOut] = None
    gamma_actual: Optional[float] = None
    alpha_actual: Optional[float] = None
    depth_actual: Optional[int] = None
    avg_depth_actual: Optional[float] = None
    nodes_actual: Optional[int] = None

    @classmethod
    def from_model(cls, ds: Dataset) -> "DatasetOut":
        return cls(
            id=ds.id,
            name=ds.name,
            rows=ds.rows or 0,
            cols=ds.cols or 0,
            decision_col=ds.decision_col,
            created_at=ds.created_at.isoformat() if ds.created_at else "",
            storage_path=ds.storage_path,
            provenance=provenance_out(ds.provenance),
            gamma_actual=ds.gamma_actual,
            alpha_actual=ds.alpha_actual,
            depth_actual=_tree_metric(ds, "depth"),
            avg_depth_actual=_tree_metric(ds, "avg_depth"),
            nodes_actual=_tree_metric(ds, "nodes"),
        )


class IntroduceRequest(BaseModel):
    method: str = "until_quality"    # until_quality | duplicate_flip | drop_attributes
    target_gamma: Optional[float] = None    # until_quality
    k: Optional[int] = None                 # duplicate_flip: rows to flip
    attrs: Optional[List[str]] = None       # drop_attributes: explicit column list
    random_state: Optional[int] = None


class CompleteRequest(BaseModel):
    method: str = "fill"    # remove | fill
    by_class: bool = False  # fill: compute stats within each decision class
