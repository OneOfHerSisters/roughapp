"""Tree registry: sources, builds, and comparison."""
from fastapi import APIRouter, Depends

from roughapp.api.dependencies import (
    ServiceContainer,
    get_actor_context,
    get_or_create_actor,
    get_services,
)
from roughapp.api.schemas.trees import TreeBuildRequest
from roughapp.business.ownership import ActorContext

router = APIRouter(prefix="/trees")


@router.get("/algorithms")
async def list_tree_algorithms(
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.list_algorithms()


@router.get("/sources")
async def list_tree_sources(
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.list_tree_sources(actor)


@router.post("/build")
def build_tree(
    req: TreeBuildRequest,
    actor: ActorContext = Depends(get_or_create_actor),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.build_from_source(actor, req.model_dump())


@router.get("")
async def list_trees(
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.list_tree_runs(actor)


@router.delete("/{tree_run_id}", status_code=204)
def delete_tree(
    tree_run_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    services.trees.delete_tree_run(actor, tree_run_id)


@router.get("/{tree_run_id}/text")
def get_tree_text(
    tree_run_id: int,
    actor: ActorContext = Depends(get_actor_context),
    services: ServiceContainer = Depends(get_services),
):
    return services.trees.get_tree_text(actor, tree_run_id)
