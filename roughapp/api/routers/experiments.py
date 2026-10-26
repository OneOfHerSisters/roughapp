"""Experiments router: HTTP in, service call, response out.

Handlers that trigger pandas/sklearn computation are plain `def` on purpose:
FastAPI runs them in the thread pool, so the event loop keeps serving other
requests while an experiment runs (see Section "Handling Long-Running
Computations" of the thesis).
"""
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse

from roughapp.api.dependencies import (
    ServiceContainer,
    get_actor_context,
    get_or_create_actor,
    get_services,
)
from roughapp.api.schemas.experiments import (
    ExperimentOut,
    ExperimentRequest,
    TreeMetricsOut,
    comparison_row,
)
from roughapp.business.ownership import ActorContext

router = APIRouter(prefix="/experiments")


@router.post("", response_model=ExperimentOut)
def create_experiment(
    req: ExperimentRequest,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    exp = services.experiments.create(
        actor,
        dataset_id=req.dataset_id,
        strategies=req.strategies,
        random_state=req.random_state,
        tie_policy=req.tie_policy,
        qr_tie_policy=req.qr_tie_policy,
    )
    return ExperimentOut.from_model(exp)


@router.get("", response_model=List[ExperimentOut])
async def list_experiments(
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return [ExperimentOut.from_model(e) for e in services.experiments.list_for(actor)]


@router.get("/comparison")
def get_comparison(
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return [comparison_row(e) for e in services.experiments.list_for_comparison(actor)]


@router.get("/{experiment_id}", response_model=ExperimentOut)
async def get_experiment(
    experiment_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return ExperimentOut.from_model(services.experiments.get(actor, experiment_id))


@router.post("/{experiment_id}/tree-metrics", response_model=TreeMetricsOut)
def preview_tree_metrics(
    experiment_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return TreeMetricsOut(**services.trees.preview_experiment_tree_metrics(actor, experiment_id))


@router.post("/{experiment_id}/results/{strategy}/tree-metrics")
def preview_result_tree(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.preview_result_tree(actor, experiment_id, strategy)


@router.post("/{experiment_id}/results/{strategy}/tree/build")
def build_result_tree(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.build_result_tree(actor, experiment_id, strategy)


@router.post("/{experiment_id}/results/{strategy}/tree/save")
def save_result_tree(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.build_result_tree(actor, experiment_id, strategy)


@router.get("/{experiment_id}/results/{strategy}/tree/text")
def get_result_tree_text(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.get_result_tree_text(actor, experiment_id, strategy)


@router.post("/{experiment_id}/tree/build")
def build_original_tree(
    experiment_id: int,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    exp = services.experiments.get(actor, experiment_id)
    return services.trees.build_dataset_tree(actor, exp.dataset_id)


@router.post("/{experiment_id}/tree/save")
def save_original_tree(
    experiment_id: int,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    exp = services.experiments.get(actor, experiment_id)
    return services.trees.build_dataset_tree(actor, exp.dataset_id)


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    services.experiments.delete(actor, experiment_id)


@router.get("/{experiment_id}/results/{strategy}/preview")
def preview_result(
    experiment_id: int,
    strategy: str,
    offset: int = 0,
    limit: int = 100,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.experiments.preview_result(actor, experiment_id, strategy, offset, limit)


@router.get("/{experiment_id}/results/{strategy}/csv")
async def download_result(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    download = services.experiments.result_download(actor, experiment_id, strategy)
    return FileResponse(
        download["path"], media_type="text/csv",
        filename=f"exp{experiment_id}_{strategy}.csv",
    )


@router.get("/{experiment_id}/results/{strategy}/details/csv")
async def download_details_csv(
    experiment_id: int,
    strategy: str,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    csv_text = services.experiments.details_csv(actor, experiment_id, strategy)
    return StreamingResponse(
        iter([csv_text]), media_type="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename=exp{experiment_id}_{strategy}_details.csv"
            ),
        },
    )


@router.get("/{experiment_id}/csv")
async def download_summary_csv(
    experiment_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    csv_text = services.experiments.summary_csv(actor, experiment_id)
    return StreamingResponse(
        iter([csv_text]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.csv"},
    )
