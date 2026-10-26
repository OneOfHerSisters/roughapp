"""Access policy: the single place that answers "may this actor do that?".

Replaces the _check_dataset_access / _require_dataset_owner helpers that
were duplicated across routers.  The policy works on plain ownership facts
(user_id / guest_session_id of a record), so it is unit-testable without a
database, and adding team-based access later means new rules here — not
edits in every router.
"""
from __future__ import annotations

from typing import Protocol

from roughapp.business.exceptions import (
    BuiltinDatasetModificationError,
    DatasetNotFoundError,
)
from roughapp.business.ownership import ActorContext


class Owned(Protocol):
    """Anything with the two nullable owner columns (Dataset works as-is)."""
    user_id: int | None
    guest_session_id: int | None


class OwnerAccessPolicy:
    """Strict private ownership — no public resources (experiments, etc.)."""

    def can_view(self, actor: ActorContext, resource: Owned) -> bool:
        if resource.user_id is not None:
            return actor.user_id == resource.user_id
        if resource.guest_session_id is not None:
            return actor.guest_session_id == resource.guest_session_id
        return False


class DatasetAccessPolicy:
    @staticmethod
    def _is_builtin(resource: Owned) -> bool:
        return resource.user_id is None and resource.guest_session_id is None

    def can_view(self, actor: ActorContext, resource: Owned) -> bool:
        if self._is_builtin(resource):
            return True  # builtin datasets are public
        if resource.user_id is not None:
            return actor.user_id == resource.user_id
        return actor.guest_session_id == resource.guest_session_id

    def can_modify(self, actor: ActorContext, resource: Owned) -> bool:
        # builtins are read-only for everyone; private data only for the owner
        return not self._is_builtin(resource) and self.can_view(actor, resource)

    can_delete = can_modify

    def require_view(self, actor: ActorContext, resource: Owned) -> None:
        """Not-found on failure: a 404 for someone else's private resource,
        so ids cannot be probed to confirm that the resource exists."""
        if not self.can_view(actor, resource):
            raise DatasetNotFoundError()

    def require_modify(self, actor: ActorContext, resource: Owned) -> None:
        self.require_view(actor, resource)
        if self._is_builtin(resource):
            raise BuiltinDatasetModificationError()

    def require_delete(self, actor: ActorContext, resource: Owned) -> None:
        self.require_modify(actor, resource)
