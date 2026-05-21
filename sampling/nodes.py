"""
LLS / Sampling
==============
功能域：图像采样与去噪（对应功能分类总览第 3 节）

CATEGORY = "LLS/Sampling"
"""
from __future__ import annotations

import random

from ..repair.repair_utils import normalize_model_info, normalize_repair_info, resolve_adapter_mode
from ..utils.model_info import (
    FAMILY_DEFAULT_PRESET,
    MODEL_FAMILY_CHOICES,
    get_family_defaults,
    get_sampling_preset,
    infer_task_mode_from_latent,
    info_to_json,
    is_flux_family,
    resolve_model_family,
)

# ---------- 防御性导入 ----------

try:
    import comfy.sample as comfy_sample
except Exception as exc:
    comfy_sample = None
    _COMFY_SAMPLE_ERR = exc
else:
    _COMFY_SAMPLE_ERR = None

try:
    import comfy.samplers as comfy_samplers
except Exception as exc:
    comfy_samplers = None
    _COMFY_SAMPLERS_ERR = exc
else:
    _COMFY_SAMPLERS_ERR = None

try:
    import comfy.utils as comfy_utils
except Exception as exc:
    comfy_utils = None
    _COMFY_UTILS_ERR = exc
else:
    _COMFY_UTILS_ERR = None

try:
    import latent_preview
except Exception as exc:
    latent_preview = None
    _LATENT_PREVIEW_ERR = exc
else:
    _LATENT_PREVIEW_ERR = None

try:
    import torch
except Exception:
    torch = None

try:
    import node_helpers
except Exception:
    node_helpers = None


# ---------- 预设定义 ----------

_QUALITY_PRESETS = [FAMILY_DEFAULT_PRESET, "Manual", "Fast", "Balanced", "High Quality"]

_DEFAULT_SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                     "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                     "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc"]
_DEFAULT_SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
_PRIMITIVE_NUMBER_INPUT = "STRING,FLOAT,INT"
_DENOISE_MODE_CHOICES = ["manual", "auto_from_repair"]
_ADAPTER_MODE_CHOICES = ["auto", "sd_classic", "flux", "sd3", "qwen", "zimage"]


def _get_samplers() -> list[str]:
    if comfy_samplers is not None:
        try:
            return list(comfy_samplers.KSampler.SAMPLERS)
        except Exception:
            pass
    return _DEFAULT_SAMPLERS


def _get_schedulers() -> list[str]:
    if comfy_samplers is not None:
        try:
            return list(comfy_samplers.KSampler.SCHEDULERS)
        except Exception:
            pass
    return _DEFAULT_SCHEDULERS


def _normalize_flux_guidance(value, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise RuntimeError("[LLS] flux_guidance must be a number, not BOOLEAN.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        value = stripped
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[LLS] Invalid flux_guidance value: {value!r}") from exc
    if normalized < 0.0 or normalized > 100.0:
        raise RuntimeError(f"[LLS] flux_guidance out of supported range [0.0, 100.0]: {normalized}")
    return normalized


def _common_ksampler(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step=None,
    last_step=None,
    force_full_denoise: bool = False,
):
    if comfy_sample is None:
        raise RuntimeError(
            "[LLS] comfy.sample is not available. "
            "Make sure this node runs inside a ComfyUI environment."
        ) from _COMFY_SAMPLE_ERR
    if comfy_utils is None:
        raise RuntimeError(
            "[LLS] comfy.utils is not available. "
            "Make sure this node runs inside a ComfyUI environment."
        ) from _COMFY_UTILS_ERR
    if latent_preview is None:
        raise RuntimeError(
            "[LLS] latent_preview is not available. "
            "Make sure this node runs inside a ComfyUI environment."
        ) from _LATENT_PREVIEW_ERR
    if not isinstance(latent, dict) or "samples" not in latent:
        raise RuntimeError(
            "[LLS] latent_image must be a LATENT dict containing the 'samples' tensor."
        )

    latent_image = latent["samples"]
    fix_empty_latent_channels = getattr(comfy_sample, "fix_empty_latent_channels", None)
    if callable(fix_empty_latent_channels):
        try:
            latent_image = fix_empty_latent_channels(
                model,
                latent_image,
                latent.get("downscale_ratio_spacial", None),
            )
        except TypeError:
            latent_image = fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        if torch is None:
            raise RuntimeError("[LLS] PyTorch is required when disable_noise is enabled.")
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_index = latent.get("batch_index")
        noise = comfy_sample.prepare_noise(latent_image, seed, batch_index)

    noise_mask = latent.get("noise_mask")
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not getattr(comfy_utils, "PROGRESS_BAR_ENABLED", True)
    sampled = comfy_sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out["samples"] = sampled
    return out


# ---------- 节点类 ----------

class LLSSimpleKSampler:
    """
    简化版 KSampler。
    内部复用 ComfyUI 原生采样能力，不重写采样算法。
    支持 quality_preset 一键切换采样参数，并输出 sample_info 串。
    """

    CATEGORY = "LLS/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sample_info")
    DESCRIPTION = (
        "Basic KSampler node. Reuses ComfyUI's native sampling pipeline. "
        "quality_preset overrides steps/cfg/denoise when not 'Manual'. "
        "seed=-1 generates a random seed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "quality_preset": (_QUALITY_PRESETS, {"default": FAMILY_DEFAULT_PRESET}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_samplers(), {"default": "euler_ancestral"}),
                "scheduler": (_get_schedulers(), {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (_DENOISE_MODE_CHOICES, {"default": "manual"}),
                "adapter_mode": (_ADAPTER_MODE_CHOICES, {"default": "auto"}),
                "flux_guidance": (
                    _PRIMITIVE_NUMBER_INPUT,
                    {"default": 3.5, "widgetType": "FLOAT", "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1},
                ),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "repair_info": ("LLS_REPAIR_INFO",),
                "guidance_stack": ("LLS_GUIDANCE_STACK",),
                "model_info": ("STRING",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, flux_guidance=None, input_types=None):
        received_type = None
        if isinstance(input_types, dict):
            received_type = input_types.get("flux_guidance")
        if received_type is not None and received_type not in {"STRING", "FLOAT", "INT"}:
            return f"flux_guidance only accepts STRING/FLOAT/INT inputs, got {received_type}."
        try:
            _normalize_flux_guidance(flux_guidance, fallback=3.5)
        except RuntimeError as exc:
            return str(exc)
        return True

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        quality_preset: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        denoise_mode: str = "manual",
        adapter_mode: str = "auto",
        flux_guidance=3.5,
        model_family: str = "Auto",
        repair_info=None,
        guidance_stack=None,
        model_info=None,
    ):
        if comfy_sample is None:
            raise RuntimeError(
                "[LLS] comfy.sample is not available. "
                "Make sure this node runs inside a ComfyUI environment."
            ) from _COMFY_SAMPLE_ERR
        if comfy_samplers is None:
            raise RuntimeError(
                "[LLS] comfy.samplers is not available. "
                "Make sure this node runs inside a ComfyUI environment."
            ) from _COMFY_SAMPLERS_ERR

        repair_meta = normalize_repair_info(repair_info) if repair_info is not None else None
        model_meta = normalize_model_info(model_info)

        family = resolve_model_family(model_family, model=model)
        if repair_meta is not None and repair_meta.get("model_family") not in {"UNKNOWN", "", None}:
            family = str(repair_meta["model_family"])
        elif model_meta.get("model_family") not in {"UNKNOWN", "", None}:
            family = str(model_meta["model_family"])

        effective_adapter = resolve_adapter_mode(adapter_mode, family)
        if effective_adapter == "sd3":
            raise RuntimeError("[LLS] SD3 repair-aware sampling is not implemented yet.")
        if effective_adapter == "qwen":
            raise RuntimeError("[LLS] QWEN repair-aware sampling is not implemented yet.")
        if effective_adapter == "zimage":
            raise RuntimeError("[LLS] ZIMAGE repair-aware sampling is not implemented yet.")

        defaults = get_family_defaults(family)
        default_flux_guidance = defaults.get("default_guidance")
        effective_task_mode = infer_task_mode_from_latent(latent_image)

        if quality_preset == FAMILY_DEFAULT_PRESET:
            steps = int(defaults["default_steps"])
            cfg = float(defaults["default_cfg"])
            sampler_name = str(defaults["default_sampler"])
            scheduler = str(defaults["default_scheduler"])
            denoise = float(defaults["default_denoise"])
            if default_flux_guidance is not None:
                flux_guidance = float(default_flux_guidance)
        else:
            preset = get_sampling_preset(defaults, quality_preset)
            if preset is not None:
                steps = int(preset["steps"])
                cfg = float(preset["cfg"])
                sampler_name = str(preset["sampler_name"])
                scheduler = str(preset["scheduler"])
                denoise = float(preset["denoise"])
                if preset.get("guidance") is not None:
                    flux_guidance = float(preset["guidance"])

        flux_guidance = _normalize_flux_guidance(flux_guidance, fallback=default_flux_guidance)
        actual_denoise = float(denoise)
        if repair_meta is not None and denoise_mode == "auto_from_repair":
            actual_denoise = float(repair_meta.get("recommended_denoise", denoise))

        if is_flux_family(family):
            guidance_value = flux_guidance
        else:
            guidance_value = None

        if is_flux_family(family) and node_helpers is not None and guidance_value is not None:
            try:
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance_value})
                negative = node_helpers.conditioning_set_values(negative, {"guidance": guidance_value})
            except Exception:
                pass

        # seed = -1 时随机生成
        actual_seed = seed
        if seed == -1:
            actual_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

        # 复用 ComfyUI 原生 common_ksampler
        try:
            result_latent = _common_ksampler(
                model=model,
                seed=actual_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent_image,
                denoise=actual_denoise,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] KSampler failed: {exc}"
            ) from exc

        sample_info = info_to_json(
            {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "guidance": guidance_value,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": actual_denoise,
                "quality_preset": quality_preset,
                "family": family,
                "task_mode": effective_task_mode,
                "repair_mode": repair_meta is not None,
                "repair_scope": repair_meta.get("repair_scope") if repair_meta else None,
                "repair_kernel": repair_meta.get("repair_kernel") if repair_meta else None,
                "guidance_used": bool(guidance_stack),
            }
        )
        return (result_latent, sample_info)


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleKSampler": LLSSimpleKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleKSampler": "LLS Simple KSampler",
}
