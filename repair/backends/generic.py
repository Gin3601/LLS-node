from __future__ import annotations

from .base import build_fallback_latent
from .registry import register_backend


class GenericRepairBackend:
    backend_name = "generic"

    def supports(self, profile):
        del profile
        return True

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, repair_kernel, routing, **_kwargs):
        latent = build_fallback_latent(
            vae,
            work_image,
            work_mask,
            repair_kernel=repair_kernel,
            latent_source=f"repair_prepare_{workspace['repair_scope']}",
        )
        return {
            "latent": latent,
            "positive": positive,
            "negative": negative,
            "backend_hints": {
                "backend_name": self.backend_name,
                "routing_reason": routing.routing_reason,
                "execution_path": routing.execution_path,
                "model_patch": "",
            },
        }


register_backend(GenericRepairBackend())
