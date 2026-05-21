from __future__ import annotations

import random

from .backends.registry import resolve_backend
from .pro_edit_utils import BACKEND_MODE_CHOICES, normalize_edit_info, set_conditioning_values
from ..sampling.nodes import (
    _QUALITY_PRESETS,
    _common_ksampler,
    _get_samplers,
    _get_schedulers,
    _normalize_flux_guidance,
)
from ..utils.model_info import (
    FAMILY_DEFAULT_PRESET,
    MODEL_FAMILY_CHOICES,
    get_family_defaults,
    get_sampling_preset,
    info_to_json,
)


class LLSProKSamplerBridge:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sample_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "quality_preset": (_QUALITY_PRESETS, {"default": FAMILY_DEFAULT_PRESET}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_samplers(), {"default": "euler"}),
                "scheduler": (_get_schedulers(), {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (["manual", "auto_from_edit"], {"default": "manual"}),
                "flux_guidance": ("STRING,FLOAT,INT", {"default": 3.5, "widgetType": "FLOAT"}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "edit_info": ("LLS_EDIT_INFO",),
                "model_info": ("STRING",),
            },
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        backend_mode,
        quality_preset,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        denoise_mode,
        flux_guidance,
        model_family,
        edit_info=None,
        model_info=None,
    ):
        effective_model_info = model_info
        normalized_model_family = str(model_family or "").strip()
        if isinstance(model_info, dict):
            effective_model_info = dict(model_info)
            if normalized_model_family not in {"", "Auto", "auto"}:
                effective_model_info.setdefault("family", normalized_model_family)
                effective_model_info.setdefault("model_family", normalized_model_family)
        elif model_info is None and normalized_model_family not in {"", "Auto", "auto"}:
            effective_model_info = {"family": normalized_model_family}

        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=effective_model_info,
            edit_info=edit_info,
        )
        defaults = get_family_defaults(routing.capabilities["model_family"])

        if quality_preset == FAMILY_DEFAULT_PRESET:
            steps = int(defaults["default_steps"])
            cfg = float(defaults["default_cfg"])
            sampler_name = str(defaults["default_sampler"])
            scheduler = str(defaults["default_scheduler"])
            denoise = float(defaults["default_denoise"])
        else:
            preset = get_sampling_preset({"family": routing.capabilities["model_family"]}, quality_preset)
            if preset is not None:
                steps = int(preset["steps"])
                cfg = float(preset["cfg"])
                sampler_name = str(preset["sampler_name"])
                scheduler = str(preset["scheduler"])
                denoise = float(preset["denoise"])

        normalized_edit_info = normalize_edit_info(edit_info)
        actual_denoise = float(denoise)
        if denoise_mode == "auto_from_edit":
            actual_denoise = float(normalized_edit_info.get("recommended_denoise", actual_denoise))

        guidance_value = None
        if routing.backend_name == "flux":
            guidance_value = _normalize_flux_guidance(flux_guidance, defaults.get("default_guidance"))
            positive = set_conditioning_values(positive, {"guidance": guidance_value})
            negative = set_conditioning_values(negative, {"guidance": guidance_value})

        actual_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if int(seed) == -1 else int(seed)
        result_latent = _common_ksampler(
            model=model,
            seed=actual_seed,
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=actual_denoise,
        )
        sample_info = info_to_json(
            {
                "backend_name": routing.backend_name,
                "routing_reason": routing.routing_reason,
                "family": routing.capabilities["model_family"],
                "model_role": routing.capabilities["model_role"],
                "seed": actual_seed,
                "steps": int(steps),
                "cfg": float(cfg),
                "guidance": guidance_value,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": actual_denoise,
                "denoise_mode": denoise_mode,
                "quality_preset": quality_preset,
            }
        )
        return result_latent, sample_info


NODE_CLASS_MAPPINGS = {"LLSProKSamplerBridge": LLSProKSamplerBridge}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProKSamplerBridge": "LLS Pro KSampler Bridge"}
