"""Who is acting: the actor context passed from presentation to business.

ActorContext is a plain value object — it does not know how the ids were
obtained (JWT, cookie) and carries the hook for future team memberships
without forcing an Owner class hierarchy whose subclasses would differ only
by an id.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TeamRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


@dataclass(frozen=True)
class TeamMembership:
    team_id: int
    role: TeamRole


@dataclass(frozen=True)
class ActorContext:
    user_id: int | None = None
    guest_session_id: int | None = None
    team_memberships: tuple[TeamMembership, ...] = ()

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None

    @property
    def is_guest(self) -> bool:
        return self.user_id is None and self.guest_session_id is not None

    @property
    def is_anonymous(self) -> bool:
        return self.user_id is None and self.guest_session_id is None
