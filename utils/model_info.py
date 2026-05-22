from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


LLS_TEXT_ENCODER_TYPE = "LLS_TEXT_ENCODER"
AUTO_FAMILY = "Auto"
FAMILY_DEFAULT_PRESET = "Family Default"
SIZE_PRESET_AUTO = "Family Default"

LEGACY_FAMILY_ALIASES: dict[str, str] = {
    "SD1.5": "SD1.5",
    "SD15": "SD1.5",
    "SDXL": "SDXL",
    "SDXL_TURBO": "SDXL_TURBO",
    "FLUX": "FLUX_DEV",
    "FLUX_SCHNELL": "FLUX_SCHNELL",
    "FLUX_DEV": "FLUX_DEV",
}

MODEL_FAMILY_CHOICES: list[str] = [
    AUTO_FAMILY,
    "SD1.5",
    "SD15",
    "SDXL",
    "SDXL_TURBO",
    "FLUX_SCHNELL",
    "FLUX_DEV",
]

FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "SD1.5": {
        "text_encoder_type": "clip",
        "required_text_encoders": ["clip"],
        "required_vae": "optional",
        "latent_channels": 4,
        "downscale_ratio": 8,
        "default_width": 512,
        "default_height": 512,
        "default_steps": 20,
        "default_cfg": 7.0,
        "default_guidance": None,
        "default_sampler": "euler",
        "default_scheduler": "normal",
        "default_denoise": 1.0,
        "guidance_embed": False,
        "is_turbo": False,
        "is_flux": False,
    },
    "SDXL": {
        "text_encoder_type": "sdxl_dual_clip",
        "required_text_encoders": ["clip_l", "clip_g"],
        "required_vae": "optional",
        "latent_channels": 4,
        "downscale_ratio": 8,
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 25,
        "default_cfg": 7.0,
        "default_guidance": None,
        "default_sampler": "dpmpp_2m",
        "default_scheduler": "karras",
        "default_denoise": 1.0,
        "guidance_embed": False,
        "is_turbo": False,
        "is_flux": False,
    },
    "SDXL_TURBO": {
        "text_encoder_type": "sdxl_dual_clip",
        "required_text_encoders": ["clip_l", "clip_g"],
        "required_vae": "optional",
        "latent_channels": 4,
        "downscale_ratio": 8,
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 4,
        "default_cfg": 1.0,
        "default_guidance": None,
        "default_sampler": "euler",
        "default_scheduler": "normal",
        "default_denoise": 1.0,
        "guidance_embed": False,
        "is_turbo": True,
        "is_flux": False,
    },
    "FLUX_SCHNELL": {
        "text_encoder_type": "flux_clip_l_t5xxl",
        "required_text_encoders": ["clip_l", "t5xxl"],
        "required_vae": "required",
        "latent_channels": 128,
        "downscale_ratio": 16,
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 4,
        "default_cfg": 1.0,
        "default_guidance": 3.5,
        "default_sampler": "euler",
        "default_scheduler": "simple",
        "default_denoise": 1.0,
        "guidance_embed": False,
        "is_turbo": False,
        "is_flux": True,
    },
    "FLUX_DEV": {
        "text_encoder_type": "flux_clip_l_t5xxl",
        "required_text_encoders": ["clip_l", "t5xxl"],
        "required_vae": "required",
        "latent_channels": 128,
        "downscale_ratio": 16,
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 20,
        "default_cfg": 1.0,
        "default_guidance": 3.5,
        "default_sampler": "euler",
        "default_scheduler": "simple",
        "default_denoise": 1.0,
        "guidance_embed": True,
        "is_turbo": False,
        "is_flux": True,
    },
}

SAMPLING_PRESETS: dict[str, dict[str, dict[str, float | int | None]]] = {
    "SD1.5": {
        FAMILY_DEFAULT_PRESET: {
            "steps": 20,
            "cfg": 7.0,
            "guidance": None,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
        "Fast": {"steps": 12, "cfg": 6.5, "guidance": None, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
        "Balanced": {"steps": 20, "cfg": 7.0, "guidance": None, "sampler_name": "euler_ancestral", "scheduler": "karras", "denoise": 1.0},
        "High Quality": {"steps": 30, "cfg": 7.5, "guidance": None, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0},
    },
    "SDXL": {
        FAMILY_DEFAULT_PRESET: {
            "steps": 25,
            "cfg": 7.0,
            "guidance": None,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        "Fast": {"steps": 16, "cfg": 6.0, "guidance": None, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
        "Balanced": {"steps": 25, "cfg": 7.0, "guidance": None, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0},
        "High Quality": {"steps": 35, "cfg": 7.0, "guidance": None, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0},
    },
    "SDXL_TURBO": {
        FAMILY_DEFAULT_PRESET: {
            "steps": 4,
            "cfg": 1.0,
            "guidance": None,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
        "Fast": {"steps": 4, "cfg": 1.0, "guidance": None, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
        "Balanced": {"steps": 4, "cfg": 1.0, "guidance": None, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
        "High Quality": {"steps": 6, "cfg": 1.2, "guidance": None, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
    },
    "FLUX_SCHNELL": {
        FAMILY_DEFAULT_PRESET: {
            "steps": 4,
            "cfg": 1.0,
            "guidance": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "Fast": {"steps": 4, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
        "Balanced": {"steps": 4, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
        "High Quality": {"steps": 6, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
    },
    "FLUX_DEV": {
        FAMILY_DEFAULT_PRESET: {
            "steps": 20,
            "cfg": 1.0,
            "guidance": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "Fast": {"steps": 12, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
        "Balanced": {"steps": 20, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
        "High Quality": {"steps": 28, "cfg": 1.0, "guidance": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
    },
}

_LEGACY_MODEL_INFO_PATTERN = re.compile(r"(\w+)=([^|]+)")
_ROLE_KEYWORDS = (
    ("inpaint", "inpaint"),
    ("kontext", "edit"),
    ("img2img", "edit"),
    ("imageedit", "edit"),
    ("image-edit", "edit"),
    ("edit", "edit"),
    ("fill", "fill"),
    ("refiner", "refiner"),
)


def canonicalize_family(family: str | None) -> str:
    if family is None:
        return "SD1.5"
    normalized = str(family).strip()
    if normalized == AUTO_FAMILY:
        return "SD1.5"
    return LEGACY_FAMILY_ALIASES.get(normalized, normalized or "SD1.5")


def infer_family_from_name(model_name: str | None, fallback: str = "SD1.5") -> str:
    if not model_name:
        return canonicalize_family(fallback)

    name = str(model_name).lower()
    if "flux" in name:
        if "schnell" in name:
            return "FLUX_SCHNELL"
        return "FLUX_DEV"
    if "turbo" in name and "xl" in name:
        return "SDXL_TURBO"
    if any(token in name for token in ("sdxl", "_xl", "-xl", "xl_")):
        return "SDXL"
    return canonicalize_family(fallback)


def is_flux_family(family: str | None) -> bool:
    return canonicalize_family(family) in {"FLUX_SCHNELL", "FLUX_DEV"}


def is_sdxl_family(family: str | None) -> bool:
    return canonicalize_family(family) in {"SDXL", "SDXL_TURBO"}


def family_requires_guidance(family: str | None) -> bool:
    resolved = canonicalize_family(family)
    return bool(FAMILY_DEFAULTS.get(resolved, FAMILY_DEFAULTS["SD1.5"])["guidance_embed"])


def get_family_defaults(family: str | None) -> dict[str, Any]:
    resolved = canonicalize_family(family)
    defaults = deepcopy(FAMILY_DEFAULTS.get(resolved, FAMILY_DEFAULTS["SD1.5"]))
    defaults["family"] = resolved
    defaults["default_size_preset"] = f"{defaults['default_width']}x{defaults['default_height']}"
    return defaults


def info_to_json(info: dict[str, Any] | None) -> str:
    return json.dumps(info or {}, ensure_ascii=True, sort_keys=True)


def parse_jsonish_info(value: dict[str, Any] | str | None) -> dict[str, Any]:
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
        return {
            key: _coerce_scalar(val.strip())
            for key, val in _LEGACY_MODEL_INFO_PATTERN.findall(stripped)
        }
    return loaded if isinstance(loaded, dict) else {}


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _coerce_bool(value: Any) -> bool:
    coerced = _coerce_scalar(value)
    if isinstance(coerced, str):
        return bool(coerced)
    return bool(coerced)


def _get_model_name_alias_value(info: dict[str, Any]) -> Any:
    return (
        info.get("checkpoint_name")
        or info.get("model_name")
        or info.get("ckpt_name")
        or info.get("ckpt")
    )


def infer_model_role_from_name(model_name: str | None, family: str | None = None) -> str:
    del family
    name = str(model_name or "").lower()
    for needle, role in _ROLE_KEYWORDS:
        if needle in name:
            return role
    return "base"


def infer_edit_capabilities(model_name: str | None, family: str | None = None) -> dict[str, Any]:
    resolved_family = canonicalize_family(family or infer_family_from_name(model_name, "SD1.5"))
    role = infer_model_role_from_name(model_name, resolved_family)

    supports_inpaint_native = False
    supports_image_edit_native = False
    preferred_edit_backend = None

    if is_sdxl_family(resolved_family):
        preferred_edit_backend = "sdxl" if role in {"inpaint", "edit", "fill"} else None
        supports_inpaint_native = role in {"inpaint", "edit", "fill"}
        supports_image_edit_native = role in {"edit", "fill"}
    elif is_flux_family(resolved_family):
        preferred_edit_backend = "flux" if role in {"inpaint", "edit", "fill"} else None
        supports_inpaint_native = role == "inpaint"
        supports_image_edit_native = role in {"inpaint", "edit", "fill"}

    return {
        "model_role": role,
        "supports_inpaint_native": supports_inpaint_native,
        "supports_image_edit_native": supports_image_edit_native,
        "preferred_edit_backend": preferred_edit_backend,
    }


def _load_profile_resolver():
    try:
        from ..model_profiles.registry import resolve_model_profile
    except ImportError:
        from model_profiles.registry import resolve_model_profile
    return resolve_model_profile


def parse_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    raw_model_name = _get_model_name_alias_value(raw)
    family = canonicalize_family(
        raw.get("family")
        or raw.get("model_family")
        or infer_family_from_name(
            raw_model_name,
            "SD1.5",
        )
    )
    defaults = get_family_defaults(family)
    resolve_model_profile = _load_profile_resolver()
    profile = resolve_model_profile(
        model=None,
        model_info=raw,
        checkpoint_name=raw_model_name,
        family=family,
    )

    info: dict[str, Any] = {**defaults}
    info.update(raw)
    info["family"] = profile["family"]
    info["model_family"] = profile["family"]
    info["model_role"] = profile["role"]
    info["profile_id"] = profile["profile_id"]
    info["backend_type"] = profile["backend_type"]
    info["sampler_strategy"] = profile["sampler_strategy"]
    info["loader_strategy"] = profile["loader_strategy"]
    info["supports_inpaint_native"] = _coerce_bool(profile["supports_inpaint_native"])
    info["supports_image_edit_native"] = _coerce_bool(profile["supports_image_edit_native"])
    info["preferred_edit_backend"] = profile["preferred_edit_backend"]
    info["text_encoder_type"] = raw.get("text_encoder_type", defaults["text_encoder_type"])
    required_text_encoders = raw.get("required_text_encoders", defaults["required_text_encoders"])
    if isinstance(required_text_encoders, str):
        required_text_encoders = [required_text_encoders]
    info["required_text_encoders"] = list(required_text_encoders)
    info["required_vae"] = raw.get("required_vae", defaults["required_vae"])
    info["has_embedded_vae"] = bool(raw.get("has_embedded_vae", False))
    info["is_turbo"] = bool(raw.get("is_turbo", defaults["is_turbo"]))
    info["is_flux"] = bool(raw.get("is_flux", defaults["is_flux"]))
    info["guidance_embed"] = bool(raw.get("guidance_embed", defaults["guidance_embed"]))

    for key in (
        "default_width",
        "default_height",
        "latent_channels",
        "downscale_ratio",
        "default_steps",
    ):
        info[key] = int(raw.get(key, defaults[key]))
    for key in ("default_cfg", "default_denoise"):
        info[key] = float(raw.get(key, defaults[key]))

    default_guidance = raw.get("default_guidance", defaults["default_guidance"])
    info["default_guidance"] = None if default_guidance in ("", None) else float(default_guidance)
    info["guidance"] = info["default_guidance"] if raw.get("guidance") in ("", None) else float(raw["guidance"])

    info["default_sampler"] = raw.get("default_sampler", defaults["default_sampler"])
    info["default_scheduler"] = raw.get("default_scheduler", defaults["default_scheduler"])
    info["default_size_preset"] = raw.get(
        "default_size_preset",
        f"{info['default_width']}x{info['default_height']}",
    )

    info.setdefault("checkpoint_name", raw_model_name or "")
    info.setdefault("model_name", info["checkpoint_name"])
    info.setdefault("vae_source", raw.get("vae_source", "auto"))
    info.setdefault("text_encoder_source", raw.get("text_encoder_source", "auto"))
    info.setdefault("load_mode", raw.get("load_mode", "simple"))

    return info


def build_model_info(**kwargs: Any) -> dict[str, Any]:
    info = parse_model_info(kwargs)
    for key, value in kwargs.items():
        if value is not None:
            info[key] = value
    info["family"] = canonicalize_family(info.get("family"))
    info["is_flux"] = bool(info.get("is_flux", is_flux_family(info["family"])))
    info["is_turbo"] = bool(info.get("is_turbo", info["family"] == "SDXL_TURBO"))
    return parse_model_info(info)


def get_latent_spec(model_info: dict[str, Any] | str | None) -> dict[str, int]:
    info = parse_model_info(model_info)
    return {
        "latent_channels": int(info["latent_channels"]),
        "downscale_ratio": int(info["downscale_ratio"]),
    }


def get_sampling_preset(model_info: dict[str, Any] | str | None, quality_preset: str) -> dict[str, Any] | None:
    if quality_preset == "Manual":
        return None
    info = parse_model_info(model_info)
    family = info["family"]
    family_presets = SAMPLING_PRESETS.get(family, SAMPLING_PRESETS["SD1.5"])
    preset_name = quality_preset if quality_preset in family_presets else FAMILY_DEFAULT_PRESET
    preset = dict(family_presets[preset_name])
    return preset


# ---------- 隐式推导工具函数 ----------

def infer_family_from_clip(clip) -> str:
    """从 CLIP 对象的 tokenizer 结构自动推断模型家族。"""
    lls_family = getattr(clip, "_lls_family", None)
    if lls_family and lls_family not in ("auto", "Auto", ""):
        return canonicalize_family(lls_family)
    try:
        tokens = clip.tokenize("")
    except Exception:
        return "SD1.5"
    if "t5xxl" in tokens:
        return "FLUX_DEV"
    if "g" in tokens:
        return "SDXL"
    return "SD1.5"


def infer_family_from_model(model) -> str:
    """从 MODEL 对象自动推断模型家族。"""
    lls_family = getattr(model, "_lls_family", None)
    if lls_family and lls_family not in ("auto", "Auto", ""):
        return canonicalize_family(lls_family)

    try:
        model_type = getattr(model, "model_type", None)
        if model_type is not None and "FLUX" in str(model_type).upper():
            return "FLUX_DEV"
    except Exception:
        pass

    try:
        latent_format = model.get_model_object("latent_format")
        channels = getattr(latent_format, "latent_channels", None)
        if channels is not None and int(channels) >= 16:
            return "FLUX_DEV"
    except Exception:
        pass

    return "SD1.5"


def infer_task_mode_from_latent(latent: dict) -> str:
    """从 LATENT dict 的 source 字段自动推断任务模式。"""
    if isinstance(latent, dict):
        source = latent.get("source", "empty_latent")
        if source == "image_encode":
            return "img2img"
    return "txt2img"


def tag_lls_object(obj, **kwargs: Any):
    """为原生 ComfyUI 对象附加轻量级 `_lls_*` 元信息。"""
    if obj is None:
        return obj
    for key, value in kwargs.items():
        if value is None:
            continue
        attr_name = key if key.startswith("_lls_") else f"_lls_{key}"
        try:
            setattr(obj, attr_name, value)
        except Exception:
            continue
    return obj


def get_lls_attr(obj, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    attr_name = name if name.startswith("_lls_") else f"_lls_{name}"
    return getattr(obj, attr_name, default)


def resolve_edit_capabilities(model=None, model_info: dict[str, Any] | str | None = None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    info = parse_model_info(model_info)
    model_name = str(
        _get_model_name_alias_value(raw)
        or get_lls_attr(model, "checkpoint_name", None)
        or get_lls_attr(model, "model_name", "")
        or info.get("checkpoint_name")
        or info.get("model_name")
        or ""
    )
    family = canonicalize_family(
        raw.get("family")
        or raw.get("model_family")
        or get_lls_attr(model, "family", None)
        or info.get("family")
        or info.get("model_family")
        or infer_family_from_model(model)
    )
    resolve_model_profile = _load_profile_resolver()
    profile = resolve_model_profile(
        model=model,
        model_info=model_info,
        checkpoint_name=model_name,
        family=family,
    )
    return {
        "model_family": profile["family"],
        "model_name": model_name,
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": _coerce_bool(profile["supports_inpaint_native"]),
        "supports_image_edit_native": _coerce_bool(profile["supports_image_edit_native"]),
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }


def resolve_model_family(
    model_family: str | None = None,
    *,
    model=None,
    clip=None,
    fallback: str = "SD1.5",
) -> str:
    normalized = str(model_family or "").strip()
    if normalized and normalized not in {"Auto", "auto"}:
        return canonicalize_family(normalized)

    if model is not None:
        inferred_from_model = infer_family_from_model(model)
        if inferred_from_model:
            return canonicalize_family(inferred_from_model)

    if clip is not None:
        inferred_from_clip = infer_family_from_clip(clip)
        if inferred_from_clip:
            return canonicalize_family(inferred_from_clip)

    return canonicalize_family(fallback)


def resolve_model_name(model=None, clip=None, fallback: str = "") -> str:
    return str(
        get_lls_attr(model, "model_name")
        or get_lls_attr(clip, "model_name")
        or fallback
        or ""
    )


def resolve_vae_name(vae=None, fallback: str = "") -> str:
    return str(get_lls_attr(vae, "vae_name") or fallback or "")


def resolve_text_encoder_names(clip=None) -> dict[str, str]:
    name = get_lls_attr(clip, "text_encoder_name") or ""
    name_1 = get_lls_attr(clip, "text_encoder_name_1") or ""
    name_2 = get_lls_attr(clip, "text_encoder_name_2") or ""

    if not name and name_1 and name_2:
        name = ", ".join(part for part in (name_1, name_2) if part)
    elif not name:
        name = name_1 or name_2 or ""

    return {
        "text_encoder_name": str(name),
        "text_encoder_name_1": str(name_1),
        "text_encoder_name_2": str(name_2),
    }


def build_info_payload(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}
