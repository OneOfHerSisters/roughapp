"""Provenance kinds and display helpers for datasets."""
from __future__ import annotations

from roughapp.db.models import Dataset, DatasetProvenance

KIND_UPLOAD = "upload"
KIND_DERIVED = "derived"
KIND_EXPERIMENT_OUTPUT = "experiment_output"


def _root_name(
    dataset_id: int,
    name_by_id: dict[int, str],
    parent_of: dict[int, int | None],
) -> str:
    """Walk provenance to the upload root; used to group a lineage in the UI."""
    seen: set[int] = set()
    cur: int | None = dataset_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        parent = parent_of.get(cur)
        if parent is None:
            return name_by_id.get(cur, f"#{cur}")
        cur = parent
    return name_by_id.get(dataset_id, f"#{dataset_id}")


def origin_from_dataset(
    ds: Dataset,
    name_by_id: dict[int, str],
    parent_of: dict[int, int | None] | None = None,
) -> dict:
    """Build the origin payload for tree sources / tree runs.

    ``group_name`` is the upload-root name so a derived table and the experiment
    outputs built from it sort in one family. ``cluster_dataset_id`` is the
    table that "owns" the cluster row (the derived/upload itself, or the input
    of an experiment output), so outputs sit directly under their input table.
    """
    parents = parent_of if parent_of is not None else {}
    prov = ds.provenance
    if prov is None:
        return {
            "type": "upload",
            "input_dataset_name": ds.name,
            "input_dataset_id": ds.id,
            "group_name": ds.name,
            "cluster_dataset_id": ds.id,
            "label": "Original upload",
        }
    if prov.kind == KIND_EXPERIMENT_OUTPUT:
        source_id = prov.source_dataset_id
        source_name = (
            name_by_id.get(source_id, f"#{source_id}")
            if source_id is not None
            else ds.name.split(" · ")[0] if " · " in ds.name else ds.name
        )
        result = ds.produced_by_result
        exp_id = result.experiment_id if result else None
        strategy = result.strategy if result else None
        root_id = source_id if source_id is not None else ds.id
        return {
            "type": "experiment_result",
            "input_dataset_name": source_name,
            "input_dataset_id": source_id,
            "group_name": _root_name(root_id, name_by_id, parents),
            "cluster_dataset_id": source_id if source_id is not None else ds.id,
            "label": ds.name if exp_id is None else ds.name,
            "experiment_id": exp_id,
            "strategy": strategy,
        }
    if prov.kind == KIND_DERIVED and prov.source_dataset_id is not None:
        parent_name = name_by_id.get(prov.source_dataset_id, f"#{prov.source_dataset_id}")
        return {
            "type": "derived",
            "input_dataset_name": parent_name,
            "input_dataset_id": prov.source_dataset_id,
            "group_name": _root_name(ds.id, name_by_id, parents),
            "cluster_dataset_id": ds.id,
            "label": f"Derived from {parent_name}",
            "parent_dataset_id": prov.source_dataset_id,
        }
    return {
        "type": "upload",
        "input_dataset_name": ds.name,
        "input_dataset_id": ds.id,
        "group_name": ds.name,
        "cluster_dataset_id": ds.id,
        "label": "Original upload",
    }


def provenance_parent_map(datasets: list[Dataset]) -> dict[int, int | None]:
    """dataset_id → immediate source_dataset_id (None for uploads / unknown)."""
    out: dict[int, int | None] = {}
    for ds in datasets:
        prov = ds.provenance
        if prov is None or prov.kind == KIND_UPLOAD:
            out[ds.id] = None
        else:
            out[ds.id] = prov.source_dataset_id
    return out


def provenance_out(prov: DatasetProvenance | None) -> dict | None:
    if prov is None:
        return None
    return {
        "kind": prov.kind,
        "source_dataset_id": prov.source_dataset_id,
    }
