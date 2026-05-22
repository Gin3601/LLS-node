from .registry import register_backend
from ..pro_edit_utils import build_fallback_latent_payload, build_native_conditioning_payload, set_conditioning_values


class SDXLProEditBackend:
    backend_name = "sdxl"

    def supports(self, profile):
        return str(profile.get("family") or "").strip().upper().startswith("SDXL")

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, routing, **_kwargs):
        if routing.execution_path != "native_edit":
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

        latent, concat_latent_image, concat_mask = build_native_conditioning_payload(
            vae,
            work_image,
            work_mask,
            latent_source=f"pro_edit_prepare_{workspace['edit_scope']}",
            masked_fill_value=0.5,
        )
        values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask,
            "edit_backend": self.backend_name,
        }
        return {
            "latent": latent,
            "positive": set_conditioning_values(positive, values),
            "negative": set_conditioning_values(negative, values),
            "backend_hints": {
                "backend_name": self.backend_name,
                "routing_reason": routing.routing_reason,
            },
        }

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(SDXLProEditBackend())
