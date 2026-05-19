from __future__ import annotations

import json
from typing import Any, TypedDict

from .model_info import FAMILY_DEFAULT_PRESET, get_family_defaults, get_sampling_preset, is_flux_family


LLS_TASK_CONTEXT_TYPE = "LLS_TASK_CONTEXT"

TASK_MODE_CHOICES = ["txt2img", "img2img", "inpaint", "outpaint", "upscale"]
TASK_CONTROLLER_MODEL_FAMILY_CHOICES = [
    "auto",
    "SD1.5",
    "SDXL",
    "FLUX",
    "SD3",
    "Qwen-Image",
    "Z-Image",
    "Hunyuan",
]
WORKFLOW_PRESET_CHOICES = ["simple", "standard", "advanced"]
QUALITY_PRESET_CHOICES = ["fast", "balanced", "quality"]

_RECOMMENDATION_KEYS = {
    "recommended_width",
    "recommended_height",
    "recommended_steps",
    "recommended_cfg",
    "recommended_denoise",
    "recommended_guidance",
    "recommended_sampler",
    "recommended_scheduler",
    "latent_channels",
    "downscale_ratio",
}
_TRIGGER_RECOMMENDATION_REFRESH_KEYS = {"model_family", "resolved_model_family", "task_mode", "quality_preset"}
_QUALITY_TO_SAMPLING_PRESET = {
    "fast": "Fast",
    "balanced": "Balanced",
    "quality": "High Quality",
}
_UNKNOWN_FAMILY_DEFAULTS = {
    "recommended_width": 1024,
    "recommended_height": 1024,
    "recommended_steps": 24,
    "recommended_cfg": 4.0,
    "recommended_denoise": 1.0,
    "recommended_guidance": None,
    "recommended_sampler": "euler",
    "recommended_scheduler": "normal",
    "latent_channels": 4,
    "downscale_ratio": 8,
    "text_encoder_type": "unknown",
    "vae_type": "vae",
}


class TaskContext(TypedDict, total=False):
    task_mode: str
    model_family: str
    resolved_model_family: str
    workflow_preset: str
    quality_preset: str
    enable_upscale: bool
    enable_controlnet: bool
    enable_reference: bool
    use_external_vae: bool
    use_external_text_encoder: bool
    recommended_width: int
    recommended_height: int
    recommended_steps: int
    recommended_cfg: float
    recommended_denoise: float
    recommended_guidance: float | None
    recommended_sampler: str
    recommended_scheduler: str
    checkpoint_name: str
    vae_source: str
    text_encoder_source: str
    latent_channels: int
    downscale_ratio: int
    text_encoder_type: str
    vae_type: str
    supports_img2img: bool
    supports_inpaint: bool
    supports_controlnet: bool
    supports_clip_skip: bool
    supports_flux_guidance: bool
    latent_source: str
    final_width: int
    final_height: int
    batch_size: int
    sampler_name: str
    scheduler: str
    steps: int
    cfg: float
    seed: int
    denoise: float
    guidance: float | None
    decoded: bool
    upscaled: bool
    source: str


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_task_mode(value: Any) -> str:
    normalized = str(value or "txt2img").strip().lower()
    return normalized if normalized in TASK_MODE_CHOICES else "txt2img"


def _normalize_workflow_preset(value: Any) -> str:
    normalized = str(value or "simple").strip().lower()
    return normalized if normalized in WORKFLOW_PRESET_CHOICES else "simple"


def _normalize_quality_preset(value: Any) -> str:
    normalized = str(value or "balanced").strip().lower()
    if normalized in QUALITY_PRESET_CHOICES:
        return normalized
    return "balanced"


def _normalize_requested_family(value: Any) -> str:
    normalized = str(value or "auto").strip()
    lowered = normalized.lower()
    if lowered in {"", "auto"}:
        return "auto"
    if normalized in {"SD15", "SD1.5"}:
        return "SD1.5"
    if normalized in {"SDXL", "SDXL_TURBO"}:
        return "SDXL"
    if normalized in {"FLUX", "FLUX_SCHNELL", "FLUX_DEV"}:
        return "FLUX"
    return normalized


def _normalize_resolved_family(value: Any, requested_family: str) -> str:
    normalized = str(value or requested_family or "auto").strip()
    if normalized in {"", "auto"}:
        return requested_family or "auto"
    if normalized in {"SD15", "SD1.5"}:
        return "SD1.5"
    if normalized in {"SDXL", "SDXL_TURBO"}:
        return normalized
    if normalized in {"FLUX", "FLUX_SCHNELL", "FLUX_DEV"}:
        return normalized
    return normalized


def _compat_family(resolved_family: str, requested_family: str) -> str:
    if resolved_family and resolved_family not in {"auto", "FLUX"}:
        return resolved_family
    if requested_family == "FLUX":
        return "FLUX_DEV"
    if requested_family in {"auto", ""}:
        return "SD1.5"
    return requested_family


def _profile_family(family: str) -> str:
    if family in {"auto", ""}:
        return "SD1.5"
    if family in {"SD3", "Qwen-Image", "Z-Image", "Hunyuan"}:
        return "SDXL"
    if family == "FLUX":
        return "FLUX_DEV"
    return family


def _denoise_for_task_mode(task_mode: str) -> float:
    return {
        "txt2img": 1.0,
        "img2img": 0.5,
        "inpaint": 0.6,
        "outpaint": 0.6,
        "upscale": 0.35,
    }.get(task_mode, 1.0)


def _coerce_number(value: Any, caster, default: Any):
    if value in (None, ""):
        return default
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def _context_defaults_for_family(requested_family: str, resolved_family: str, task_mode: str, quality_preset: str) -> dict[str, Any]:
    compat_family = _compat_family(resolved_family, requested_family)
    profile_family = _profile_family(compat_family)

    if profile_family in {"SD1.5", "SDXL", "SDXL_TURBO", "FLUX_SCHNELL", "FLUX_DEV"}:
        family_defaults = get_family_defaults(profile_family)
        preset_name = _QUALITY_TO_SAMPLING_PRESET.get(quality_preset, FAMILY_DEFAULT_PRESET)
        preset = get_sampling_preset({"family": profile_family}, preset_name) or {
            "steps": family_defaults["default_steps"],
            "cfg": family_defaults["default_cfg"],
            "guidance": family_defaults["default_guidance"],
            "sampler_name": family_defaults["default_sampler"],
            "scheduler": family_defaults["default_scheduler"],
            "denoise": family_defaults["default_denoise"],
        }
        defaults = {
            "recommended_width": int(family_defaults["default_width"]),
            "recommended_height": int(family_defaults["default_height"]),
            "recommended_steps": int(preset["steps"]),
            "recommended_cfg": float(preset["cfg"]),
            "recommended_denoise": float(_denoise_for_task_mode(task_mode)),
            "recommended_guidance": None if preset.get("guidance") in ("", None) else float(preset["guidance"]),
            "recommended_sampler": str(preset["sampler_name"]),
            "recommended_scheduler": str(preset["scheduler"]),
            "latent_channels": int(family_defaults["latent_channels"]),
            "downscale_ratio": int(family_defaults["downscale_ratio"]),
            "text_encoder_type": str(family_defaults["text_encoder_type"]),
            "vae_type": "ae" if is_flux_family(profile_family) else "vae",
        }
        return defaults

    defaults = dict(_UNKNOWN_FAMILY_DEFAULTS)
    defaults["recommended_denoise"] = float(_denoise_for_task_mode(task_mode))
    return defaults


def _context_capabilities(compat_family: str, latent_channels: Any) -> dict[str, bool]:
    is_flux = compat_family in {"FLUX", "FLUX_SCHNELL", "FLUX_DEV"}
    has_latent = latent_channels not in (None, "", 0)
    return {
        "supports_img2img": has_latent,
        "supports_inpaint": has_latent,
        "supports_controlnet": compat_family in {"SD1.5", "SDXL", "SDXL_TURBO"},
        "supports_clip_skip": not is_flux,
        "supports_flux_guidance": is_flux,
    }


def _parse_jsonish_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {}
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _finalize_task_context(raw_context: dict[str, Any]) -> dict[str, Any]:
    context = dict(raw_context)
    requested_family = _normalize_requested_family(
        context.get("model_family") or context.get("requested_model_family") or context.get("family")
    )
    resolved_family = _normalize_resolved_family(context.get("resolved_model_family") or context.get("family"), requested_family)
    if resolved_family == "auto":
        resolved_family = _compat_family(resolved_family, requested_family)
    task_mode = _normalize_task_mode(context.get("task_mode"))
    workflow_preset = _normalize_workflow_preset(context.get("workflow_preset"))
    quality_preset = _normalize_quality_preset(context.get("quality_preset"))

    defaults = _context_defaults_for_family(requested_family, resolved_family, task_mode, quality_preset)
    compat_family = _compat_family(resolved_family, requested_family)
    capabilities = _context_capabilities(compat_family, context.get("latent_channels", defaults.get("latent_channels")))

    context["task_mode"] = task_mode
    context["model_family"] = requested_family
    context["resolved_model_family"] = resolved_family
    context["workflow_preset"] = workflow_preset
    context["quality_preset"] = quality_preset
    context["enable_upscale"] = _normalize_bool(context.get("enable_upscale"), True)
    context["enable_controlnet"] = _normalize_bool(context.get("enable_controlnet"), False)
    context["enable_reference"] = _normalize_bool(context.get("enable_reference"), False)
    context["use_external_vae"] = _normalize_bool(context.get("use_external_vae"), False)
    context["use_external_text_encoder"] = _normalize_bool(context.get("use_external_text_encoder"), False)

    for key, default_value in defaults.items():
        if key in {"recommended_cfg", "recommended_denoise", "recommended_guidance"}:
            context[key] = None if default_value is None else _coerce_number(context.get(key), float, float(default_value))
            continue
        if key in {"recommended_steps", "recommended_width", "recommended_height", "latent_channels", "downscale_ratio"}:
            context[key] = _coerce_number(context.get(key), int, int(default_value))
            continue
        context[key] = context.get(key, default_value)

    if context["recommended_guidance"] is None and defaults.get("recommended_guidance") is not None:
        context["recommended_guidance"] = float(defaults["recommended_guidance"])

    context["checkpoint_name"] = str(context.get("checkpoint_name") or context.get("model_name") or "")
    context["vae_source"] = str(context.get("vae_source") or ("external" if context["use_external_vae"] else "auto"))
    context["text_encoder_source"] = str(
        context.get("text_encoder_source") or ("external" if context["use_external_text_encoder"] else "auto")
    )
    context["text_encoder_type"] = str(context.get("text_encoder_type") or defaults["text_encoder_type"])
    context["vae_type"] = str(context.get("vae_type") or defaults["vae_type"])

    for key, default_value in capabilities.items():
        context[key] = _normalize_bool(context.get(key), default_value)

    context["family"] = compat_family
    context["default_width"] = int(context["recommended_width"])
    context["default_height"] = int(context["recommended_height"])
    context["default_steps"] = int(context["recommended_steps"])
    context["default_cfg"] = float(context["recommended_cfg"])
    context["default_denoise"] = float(context["recommended_denoise"])
    context["default_guidance"] = context["recommended_guidance"]
    context["default_sampler"] = str(context["recommended_sampler"])
    context["default_scheduler"] = str(context["recommended_scheduler"])
    context["guidance"] = (
        None
        if context.get("guidance") in (None, "")
        else _coerce_number(context.get("guidance"), float, context["recommended_guidance"])
    )
    if context["guidance"] is None:
        context["guidance"] = context["recommended_guidance"]
    context["is_flux"] = _normalize_bool(context.get("is_flux"), is_flux_family(compat_family))
    context["is_turbo"] = _normalize_bool(context.get("is_turbo"), compat_family == "SDXL_TURBO")

    if "final_width" in context:
        context["final_width"] = _coerce_number(context.get("final_width"), int, context["recommended_width"])
    if "final_height" in context:
        context["final_height"] = _coerce_number(context.get("final_height"), int, context["recommended_height"])
    if "batch_size" in context:
        context["batch_size"] = _coerce_number(context.get("batch_size"), int, 1)
    if "steps" in context:
        context["steps"] = _coerce_number(context.get("steps"), int, context["recommended_steps"])
    if "cfg" in context:
        context["cfg"] = _coerce_number(context.get("cfg"), float, context["recommended_cfg"])
    if "seed" in context:
        context["seed"] = _coerce_number(context.get("seed"), int, -1)
    if "denoise" in context:
        context["denoise"] = _coerce_number(context.get("denoise"), float, context["recommended_denoise"])
    return context


def parse_task_context(task_context: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = _parse_jsonish_dict(task_context)
    if isinstance(task_context, dict):
        raw = dict(task_context)

    if "resolved_model_family" not in raw and raw.get("family"):
        raw["resolved_model_family"] = raw.get("family")
    if "model_family" not in raw and raw.get("family"):
        raw["model_family"] = raw.get("family")

    if "recommended_width" not in raw and raw.get("default_width") is not None:
        raw["recommended_width"] = raw.get("default_width")
    if "recommended_height" not in raw and raw.get("default_height") is not None:
        raw["recommended_height"] = raw.get("default_height")
    if "recommended_steps" not in raw and raw.get("default_steps") is not None:
        raw["recommended_steps"] = raw.get("default_steps")
    if "recommended_cfg" not in raw and raw.get("default_cfg") is not None:
        raw["recommended_cfg"] = raw.get("default_cfg")
    if "recommended_denoise" not in raw and raw.get("default_denoise") is not None:
        raw["recommended_denoise"] = raw.get("default_denoise")
    if "recommended_guidance" not in raw and raw.get("default_guidance") is not None:
        raw["recommended_guidance"] = raw.get("default_guidance")
    if "recommended_sampler" not in raw and raw.get("default_sampler") is not None:
        raw["recommended_sampler"] = raw.get("default_sampler")
    if "recommended_scheduler" not in raw and raw.get("default_scheduler") is not None:
        raw["recommended_scheduler"] = raw.get("default_scheduler")

    return _finalize_task_context(raw)


def create_task_context(
    task_mode: str = "txt2img",
    model_family: str = "auto",
    workflow_preset: str = "simple",
    quality_preset: str = "balanced",
    enable_upscale: bool = False,
    enable_controlnet: bool = False,
    enable_reference: bool = False,
    use_external_vae: bool = False,
    use_external_text_encoder: bool = False,
    **updates: Any,
) -> dict[str, Any]:
    context = {
        "task_mode": task_mode,
        "model_family": model_family,
        "workflow_preset": workflow_preset,
        "quality_preset": quality_preset,
        "enable_upscale": enable_upscale,
        "enable_controlnet": enable_controlnet,
        "enable_reference": enable_reference,
        "use_external_vae": use_external_vae,
        "use_external_text_encoder": use_external_text_encoder,
        "source": "LLS Task Controller",
    }
    context.update({key: value for key, value in updates.items() if value is not None})
    return _finalize_task_context(context)


def infer_default_params(task_context: dict[str, Any] | str | None) -> dict[str, Any]:
    context = parse_task_context(task_context)
    return {
        key: context.get(key)
        for key in (
            "recommended_width",
            "recommended_height",
            "recommended_steps",
            "recommended_cfg",
            "recommended_denoise",
            "recommended_guidance",
            "recommended_sampler",
            "recommended_scheduler",
            "latent_channels",
            "downscale_ratio",
        )
    }


def update_task_context(task_context: dict[str, Any] | str | None = None, **updates: Any) -> dict[str, Any]:
    context = parse_task_context(task_context)
    refresh_recommendations = any(key in _TRIGGER_RECOMMENDATION_REFRESH_KEYS for key in updates)
    if refresh_recommendations:
        for key in _RECOMMENDATION_KEYS:
            if key not in updates:
                context.pop(key, None)
    for key, value in updates.items():
        if value is not None:
            context[key] = value
    return _finalize_task_context(context)


def merge_task_context(*contexts: dict[str, Any] | str | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for context in contexts:
        if context in (None, ""):
            continue
        merged.update(_parse_jsonish_dict(context) if not isinstance(context, dict) else dict(context))
    return _finalize_task_context(merged)


def task_context_to_json(task_context: dict[str, Any] | str | None) -> str:
    return json.dumps(parse_task_context(task_context), ensure_ascii=True, sort_keys=True)
