"""Guard: the tie-breaking value sets used for API-level validation must stay
in sync with what the resolution strategies actually implement.

The API rejects unknown tie policies using MCD_TIE_POLICIES / QR_TIE_POLICIES;
those tuples must match the Literal annotations of the strategy signatures, or
a valid policy could be refused (or an unimplemented one accepted).
"""

from typing import Literal, get_args, get_type_hints

from roughapp.db.models import MCD_TIE_POLICIES, QR_TIE_POLICIES
from roughapp.business.algorithms.resolution import most_common_decision, rough_approximation


def _literal_values(fn, param: str) -> tuple:
    annotation = get_type_hints(fn)[param]
    assert getattr(annotation, "__origin__", None) is Literal or get_args(annotation), (
        f"{param} is not a Literal annotation"
    )
    return get_args(annotation)


def test_mcd_tie_policies_match_signature():
    assert set(MCD_TIE_POLICIES) == set(_literal_values(most_common_decision.run, "tie_policy"))


def test_qr_tie_policies_match_signature():
    assert set(QR_TIE_POLICIES) == set(_literal_values(rough_approximation.run, "tie_policy"))
