from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]


class ProEditBackend(Protocol):
    backend_name: str

    def supports(self, capabilities: dict[str, Any]) -> bool:
        raise NotImplementedError

    def prepare(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_bridge(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


def validate_manual_backend(backend_name: str, backend: ProEditBackend, capabilities: dict[str, Any]) -> None:
    if backend.supports(capabilities):
        return
    family = capabilities.get("model_family") or "UNKNOWN"
    raise RuntimeError(
        f"[LLS] Pro edit backend '{backend_name}' is incompatible with model family '{family}'."
    )
