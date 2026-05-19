"""
LLS / Sampling
==============
功能域：图像采样与去噪（对应功能分类总览第 3 节）

CATEGORY = "LLS/Sampling"
"""
from __future__ import annotations

import random

from ..utils.model_info import (
    FAMILY_DEFAULT_PRESET,
    LLS_MODEL_INFO_TYPE,
    get_sampling_preset,
    info_to_json,
    is_flux_family,
    parse_model_info,
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
    out.pop("downscale_ratio_spacial", None)
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
        "quality_preset overrides steps and cfg when not 'Manual'. "
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
                "flux_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "model_info": (LLS_MODEL_INFO_TYPE,),
            },
        }

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
        flux_guidance: float,
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

        info = parse_model_info(model_info)
        family = info["family"]

        preset = get_sampling_preset(info, quality_preset)
        if preset is not None:
            steps = int(preset["steps"])
            cfg = float(preset["cfg"])
            sampler_name = str(preset["sampler_name"])
            scheduler = str(preset["scheduler"])
            denoise = float(preset["denoise"])
            if preset.get("guidance") is not None:
                flux_guidance = float(preset["guidance"])

        if is_flux_family(family) and node_helpers is not None:
            try:
                positive = node_helpers.conditioning_set_values(positive, {"guidance": flux_guidance})
                negative = node_helpers.conditioning_set_values(negative, {"guidance": flux_guidance})
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
                denoise=denoise,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] KSampler failed: {exc}"
            ) from exc

        guidance_value = flux_guidance if is_flux_family(family) else None
        sample_info = info_to_json(
            {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "guidance": guidance_value,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "quality_preset": quality_preset,
                "family": family,
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
