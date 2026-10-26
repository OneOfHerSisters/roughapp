"""Datasets router: HTTP in, service call, response out.

Heavy handlers are declared as plain `def` (or off-load with
asyncio.to_thread) so pandas work runs in the thread pool and the event
loop stays free — see the System Design chapter, "Handling Long-Running
Computations".
"""
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from roughapp.api.dependencies import (
    ServiceContainer,
    get_actor_context,
    get_or_create_actor,
    get_services,
)
from roughapp.api.schemas.datasets import CompleteRequest, DatasetOut, IntroduceRequest
from roughapp.business.ownership import ActorContext

router = APIRouter(prefix="/datasets")


@router.post("", response_model=DatasetOut)
async def upload_dataset(
    file: UploadFile = File(...),
    decision_col: str = Form(...),
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    contents = await file.read()
    ds = await asyncio.to_thread(
        services.datasets.upload, actor, file.filename or "", contents, decision_col
    )
    return DatasetOut.from_model(ds)


@router.get("", response_model=List[DatasetOut])
async def list_datasets(
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return [DatasetOut.from_model(ds) for ds in services.datasets.list_visible(actor)]


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(
    dataset_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return DatasetOut.from_model(services.datasets.get_visible(actor, dataset_id))


@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: int,
    offset: int = 0,
    limit: int = 100,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.datasets.preview(actor, dataset_id, offset, limit)


@router.get("/{dataset_id}/analyze")
def analyze_dataset(
    dataset_id: int,
    decision_col: Optional[str] = None,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.datasets.analyze(actor, dataset_id, decision_col)


@router.get("/{dataset_id}/attribute-importance")
def attribute_importance(
    dataset_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    # Plain `def`: γ is recomputed once per attribute (CPU-bound pandas loop),
    # so this runs in the thread pool instead of blocking the event loop.
    return services.datasets.attribute_importance(actor, dataset_id)


@router.get("/{dataset_id}/tree")
def preview_dataset_tree(
    dataset_id: int,
    max_depth: Optional[int] = None,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.preview_dataset_tree(actor, dataset_id, max_depth)


@router.post("/{dataset_id}/tree/build")
def build_dataset_tree(
    dataset_id: int,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.build_dataset_tree(actor, dataset_id)


@router.post("/{dataset_id}/tree/save")
def save_dataset_tree(
    dataset_id: int,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.build_dataset_tree(actor, dataset_id)


@router.get("/{dataset_id}/tree/text")
def get_dataset_tree_text(
    dataset_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.get_dataset_tree_text(actor, dataset_id)


@router.get("/{dataset_id}/missing-summary")
def get_missing_summary(
    dataset_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.datasets.missing_summary(actor, dataset_id)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: int,
    cascade: bool = False,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    """Without cascade, a dataset used by experiments answers 409 with a
    manifest of those experiments; retrying with cascade=true deletes them
    too (their result tables survive as independent datasets)."""
    services.datasets.delete(actor, dataset_id, cascade=cascade)


@router.post("/{dataset_id}/introduce", response_model=DatasetOut)
async def introduce_dataset(
    dataset_id: int,
    req: IntroduceRequest,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    ds = await asyncio.to_thread(
        services.datasets.introduce, actor, dataset_id, req.method,
        target_gamma=req.target_gamma, k=req.k, attrs=req.attrs,
        random_state=req.random_state,
    )
    return DatasetOut.from_model(ds)


@router.post("/{dataset_id}/complete", response_model=DatasetOut)
async def complete_dataset(
    dataset_id: int,
    req: CompleteRequest,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    ds = await asyncio.to_thread(
        services.datasets.complete, actor, dataset_id, req.method, req.by_class
    )
    return DatasetOut.from_model(ds)


@router.post("/{dataset_id}/deduplicate", response_model=DatasetOut)
async def deduplicate_dataset(
    dataset_id: int,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    """Remove exact duplicate rows, saving the result as a new dataset."""
    ds = await asyncio.to_thread(services.datasets.remove_duplicates, actor, dataset_id)
    return DatasetOut.from_model(ds)
