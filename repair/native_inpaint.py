from __future__ import annotations

from ..utils.model_info import info_to_json, resolve_edit_capabilities
from . import runtime as repair_runtime


_PATCH_MODE_CHOICES = ["auto", "disabled", "differential_diffusion"]


class LLSNativeInpaintConditioning:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "encode"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "inpaint_info")
    DESCRIPTION = (
        "Thin wrapper around ComfyUI InpaintModelConditioning. "
        "Keeps the standard KSampler path and can auto-apply DifferentialDiffusion for FLUX."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "patch_mode": (_PATCH_MODE_CHOICES, {"default": "auto"}),
                "patch_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_info": ("STRING",),
            },
        }

    def encode(
        self,
        model,
        positive,
        negative,
        vae,
        image,
        mask,
        patch_mode,
        patch_strength,
        noise_mask,
        model_info=None,
    ):
        capabilities = resolve_edit_capabilities(model=model, model_info=model_info)

        apply_patch = False
        if patch_mode == "differential_diffusion":
            apply_patch = True
        elif patch_mode == "auto" and str(capabilities["model_family"]).startswith("FLUX"):
            apply_patch = True

        patched_model = model
        applied_patch = ""
        if apply_patch:
            patched_model = repair_runtime.apply_differential_diffusion(
                model,
                strength=float(patch_strength),
            )
            applied_patch = "differential_diffusion"

        positive_out, negative_out, latent = repair_runtime.encode_inpaint_conditioning(
            positive,
            negative,
            vae,
            image,
            mask,
            noise_mask=bool(noise_mask),
        )

        inpaint_info = info_to_json(
            {
                "model_family": capabilities["model_family"],
                "model_role": capabilities["model_role"],
                "profile_id": capabilities["profile_id"],
                "backend_type": capabilities["backend_type"],
                "sampler_strategy": capabilities["sampler_strategy"],
                "loader_strategy": capabilities["loader_strategy"],
                "patch_mode": patch_mode,
                "applied_patch": applied_patch,
                "patch_strength": float(patch_strength),
                "noise_mask": bool(noise_mask),
            }
        )
        return (
            patched_model,
            positive_out,
            negative_out,
            latent,
            inpaint_info,
        )


NODE_CLASS_MAPPINGS = {
    "LLSNativeInpaintConditioning": LLSNativeInpaintConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSNativeInpaintConditioning": "LLS Native Inpaint Conditioning",
}
