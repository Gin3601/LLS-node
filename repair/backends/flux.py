from __future__ import annotations

from .. import runtime as repair_runtime
from .base import build_fallback_latent
from .registry import register_backend


class FluxRepairBackend:
    backend_name = "flux"

    def supports(self, profile):
        return str(profile.get("family") or "").strip().upper().startswith("FLUX")

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, repair_kernel, routing, **_kwargs):
        if routing.execution_path != "native_repair":
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

        if positive is None or negative is None:
            raise RuntimeError("[LLS] Native FLUX repair requires positive and negative conditioning.")

        positive_out, negative_out, latent = repair_runtime.encode_inpaint_conditioning(
            positive,
            negative,
            vae,
            work_image,
            work_mask,
            noise_mask=True,
        )
        latent["source"] = f"repair_prepare_{workspace['repair_scope']}"
        return {
            "latent": latent,
            "positive": positive_out,
            "negative": negative_out,
            "backend_hints": {
                "backend_name": self.backend_name,
                "routing_reason": routing.routing_reason,
                "execution_path": routing.execution_path,
                "model_patch": "differential_diffusion",
                "model_patch_strength": 1.0,
            },
        }


register_backend(FluxRepairBackend())
