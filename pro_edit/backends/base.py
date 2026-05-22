from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]
    profile: dict[str, Any]


class ProEditBackend(Protocol):
    backend_name: str

    def supports(self, profile: dict[str, Any]) -> bool:
        raise NotImplementedError

    def prepare(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_bridge(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


def validate_manual_backend(backend_name: str, backend: ProEditBackend, profile: dict[str, Any]) -> None:
    if backend.supports(profile):
        return
    raise RuntimeError(
        f"[LLS] Pro edit backend '{backend_name}' is incompatible with profile "
        f"'{profile.get('profile_id')}' (family '{profile.get('family')}', role '{profile.get('role')}')."
    )
