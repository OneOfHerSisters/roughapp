"""DatasetAccessPolicy unit tests — no database, plain value objects."""
from dataclasses import dataclass

import pytest

from roughapp.business.exceptions import (
    BuiltinDatasetModificationError,
    DatasetNotFoundError,
)
from roughapp.business.ownership import ActorContext
from roughapp.business.policies import DatasetAccessPolicy, OwnerAccessPolicy


@dataclass
class FakeDataset:
    user_id: int | None = None
    guest_session_id: int | None = None


policy = DatasetAccessPolicy()
builtin = FakeDataset()
user_owned = FakeDataset(user_id=7)
guest_owned = FakeDataset(guest_session_id=3)

owner = ActorContext(user_id=7)
other_user = ActorContext(user_id=8)
guest = ActorContext(guest_session_id=3)
other_guest = ActorContext(guest_session_id=4)
anonymous = ActorContext()


class TestCanView:
    def test_builtin_is_public(self):
        for actor in (owner, other_user, guest, anonymous):
            assert policy.can_view(actor, builtin)

    def test_user_dataset_only_for_its_owner(self):
        assert policy.can_view(owner, user_owned)
        assert not policy.can_view(other_user, user_owned)
        assert not policy.can_view(guest, user_owned)
        assert not policy.can_view(anonymous, user_owned)

    def test_guest_dataset_only_for_its_session(self):
        assert policy.can_view(guest, guest_owned)
        assert not policy.can_view(other_guest, guest_owned)
        assert not policy.can_view(owner, guest_owned)
        assert not policy.can_view(anonymous, guest_owned)


class TestOwnerAccessPolicy:
    owner_policy = OwnerAccessPolicy()

    def test_private_resource_not_public(self):
        owned = FakeDataset(user_id=7)
        assert self.owner_policy.can_view(owner, owned)
        assert not self.owner_policy.can_view(other_user, owned)
        assert not self.owner_policy.can_view(anonymous, owned)

    def test_no_owner_means_inaccessible(self):
        assert not self.owner_policy.can_view(owner, FakeDataset())


class TestModify:
    def test_builtin_not_modifiable_even_by_users(self):
        assert not policy.can_modify(owner, builtin)
        with pytest.raises(BuiltinDatasetModificationError):
            policy.require_modify(owner, builtin)

    def test_owner_can_modify(self):
        assert policy.can_modify(owner, user_owned)
        policy.require_modify(owner, user_owned)  # no raise
        assert policy.can_delete(guest, guest_owned)

    def test_stranger_gets_not_found_not_forbidden(self):
        # privacy: existence must not be confirmed to non-owners
        with pytest.raises(DatasetNotFoundError):
            policy.require_modify(other_user, user_owned)


class TestActorContext:
    def test_flags(self):
        assert owner.is_authenticated and not owner.is_guest
        assert guest.is_guest and not guest.is_authenticated
        assert anonymous.is_anonymous
