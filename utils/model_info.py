from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


LLS_TEXT_ENCODER_TYPE = "LLS_TEXT_ENCODER"
LLS_MODEL_INFO_TYPE = "LLS_MODEL_INFO"

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


def parse_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    family = canonicalize_family(
        raw.get("family")
        or infer_family_from_name(
            raw.get("checkpoint_name") or raw.get("model_name") or raw.get("ckpt_name") or raw.get("ckpt"),
            "SD1.5",
        )
    )
    defaults = get_family_defaults(family)

    info: dict[str, Any] = {**defaults}
    info.update(raw)
    info["family"] = family
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

    info.setdefault("checkpoint_name", raw.get("checkpoint_name") or raw.get("model_name") or "")
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
