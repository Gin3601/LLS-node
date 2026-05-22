from .registry import register_backend
from ..pro_edit_utils import build_fallback_latent_payload


class GenericProEditBackend:
    backend_name = "generic"

    def supports(self, profile):
        family = str(profile.get("family") or "").strip().upper()
        return not family.startswith("SDXL") and not family.startswith("FLUX")

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, routing, **_kwargs):
        latent = build_fallback_latent_payload(
            vae,
            work_image,
            work_mask,
            latent_source=f"pro_edit_fallback_{workspace['edit_scope']}",
        )
        return {
            "latent": latent,
            "positive": positive,
            "negative": negative,
            "backend_hints": {
                "backend_name": self.backend_name,
                "routing_reason": routing.routing_reason,
            },
        }

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(GenericProEditBackend())
