"""FastAPI dependency wiring: request → actor context and service instances.

This is the composition root: the only place that knows which concrete
repository, storage, and unit-of-work implementations the services get.
Swapping LocalFileStorage for an S3 implementation is a one-line change in
ServiceContainer.

FastAPI caches a dependency per request, so `get_services` builds the whole
object graph exactly once per request and every endpoint (and service)
shares the same repositories and unit of work over the same session.
"""
from dataclasses import dataclass

from fastapi import Depends, Response
from sqlalchemy.orm import Session

from roughapp.auth import RequestOwner, ensure_owner, get_request_owner_optional
from roughapp.business.ownership import ActorContext
from roughapp.business.policies import DatasetAccessPolicy, OwnerAccessPolicy
from roughapp.business.services import DatasetService, ExperimentService, TreeService
from roughapp.db.engine import get_db
from roughapp.db.repositories import (
    DatasetRepository,
    ExperimentRepository,
    ReferenceRepository,
    TreeRunRepository,
)
from roughapp.db.storage import LocalFileStorage
from roughapp.db.unit_of_work import SqlAlchemyUnitOfWork

_storage = LocalFileStorage()
_policy = DatasetAccessPolicy()
_owner_policy = OwnerAccessPolicy()


@dataclass(frozen=True)
class ServiceContainer:
    datasets: DatasetService
    experiments: ExperimentService
    trees: TreeService


def build_services(db: Session) -> ServiceContainer:
    """One dependency graph per request/session."""
    uow = SqlAlchemyUnitOfWork(db)
    dataset_repo = DatasetRepository(db)
    reference_repo = ReferenceRepository(db)

    dataset_service = DatasetService(
        repository=dataset_repo,
        experiments=ExperimentRepository(db),
        storage=_storage,
        access_policy=_policy,
        unit_of_work=uow,
    )
    experiment_service = ExperimentService(
        experiments=ExperimentRepository(db),
        datasets=dataset_repo,
        references=reference_repo,
        dataset_service=dataset_service,
        storage=_storage,
        owner_policy=_owner_policy,
        unit_of_work=uow,
    )
    tree_service = TreeService(
        experiments=experiment_service,
        datasets=dataset_service,
        references=reference_repo,
        tree_runs=TreeRunRepository(db),
        storage=_storage,
        unit_of_work=uow,
    )
    return ServiceContainer(
        datasets=dataset_service,
        experiments=experiment_service,
        trees=tree_service,
    )


def get_services(db: Session = Depends(get_db)) -> ServiceContainer:
    return build_services(db)


def _to_actor(owner: RequestOwner) -> ActorContext:
    return ActorContext(
        user_id=owner.user.id if owner.user else None,
        guest_session_id=owner.guest_session.id if owner.guest_session else None,
    )


def get_actor_context(
    owner: RequestOwner = Depends(get_request_owner_optional),
) -> ActorContext:
    """Actor for read endpoints: may be anonymous."""
    return _to_actor(owner)


def get_or_create_actor(
    response: Response,
    db: Session = Depends(get_db),
    owner: RequestOwner = Depends(get_request_owner_optional),
) -> ActorContext:
    """Actor for write endpoints: opens a guest session (cookie) if needed."""
    return _to_actor(ensure_owner(owner, db, response))

