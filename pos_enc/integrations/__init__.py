"""Optional integrations for the framework-neutral :mod:`rope_contract`.

The baseline package remains importable without the middleware checkout.  The
adapter is therefore kept behind this namespace and imports ``rope_contract``
only when the integration is used.
"""

from .rope_contract_provider import (
    RayRoPEGeometry,
    RayRoPEProvider,
    RayRoPESession,
    make_prepare_request,
    make_transform_request,
)

__all__ = [
    "RayRoPEGeometry",
    "RayRoPEProvider",
    "RayRoPESession",
    "make_prepare_request",
    "make_transform_request",
]
