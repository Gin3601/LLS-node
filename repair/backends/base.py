from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..repair_utils import make_noise_mask


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]
    profile: dict[str, Any]
    execution_path: str


class RepairBackend(Protocol):
    backend_name: str

    def supports(self, profile: dict[str, Any]) -> bool:
        raise NotImplementedError

    def prepare(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


def validate_backend(backend_name: str, backend: RepairBackend, profile: dict[str, Any]) -> None:
    if backend.supports(profile):
        return
    raise RuntimeError(
        f"[LLS] Repair backend '{backend_name}' is incompatible with profile "
        f"'{profile.get('profile_id')}' (family '{profile.get('family')}', role '{profile.get('role')}')."
    )


def build_fallback_latent(vae, work_image, work_mask, *, repair_kernel: str, latent_source: str):
    latent_samples = vae.encode(work_image)
    latent = {
        "samples": latent_samples,
        "source": latent_source,
    }
    if str(repair_kernel or "") == "latent_mask":
        latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)
    return latent
