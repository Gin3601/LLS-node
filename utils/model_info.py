from __future__ import annotations

import json
import re
from typing import Any


AUTO_FAMILY = "Auto"

LEGACY_FAMILY_ALIASES: dict[str, str] = {
    "SD1.5": "SD15",
    "SD15": "SD15",
    "SDXL": "SDXL",
    "SDXL_TURBO": "SDXL_TURBO",
    "FLUX": "FLUX_DEV",
    "FLUX_SCHNELL": "FLUX_SCHNELL",
    "FLUX_DEV": "FLUX_DEV",
}

MODEL_FAMILY_CHOICES: list[str] = [
    AUTO_FAMILY,
    "SD15",
    "SD1.5",
    "SDXL",
    "SDXL_TURBO",
    "FLUX_SCHNELL",
    "FLUX_DEV",
]

TEXT_ENCODER_TYPES: dict[str, str] = {
    "SD15": "clip",
    "SDXL": "sdxl_dual_clip",
    "SDXL_TURBO": "sdxl_dual_clip",
    "FLUX_SCHNELL": "flux_clip_l_t5xxl",
    "FLUX_DEV": "flux_clip_l_t5xxl",
}

REQUIRED_TEXT_ENCODERS: dict[str, list[str]] = {
    "SD15": ["clip"],
    "SDXL": ["clip_l", "clip_g"],
    "SDXL_TURBO": ["clip_l", "clip_g"],
    "FLUX_SCHNELL": ["clip_l", "t5xxl"],
    "FLUX_DEV": ["clip_l", "t5xxl"],
}

REQUIRED_VAE: dict[str, str] = {
    "SD15": "optional",
    "SDXL": "optional",
    "SDXL_TURBO": "optional",
    "FLUX_SCHNELL": "required",
    "FLUX_DEV": "required",
}

LATENT_SPECS: dict[str, dict[str, int]] = {
    "SD15": {"latent_channels": 4, "downscale_ratio": 8},
    "SDXL": {"latent_channels": 4, "downscale_ratio": 8},
    "SDXL_TURBO": {"latent_channels": 4, "downscale_ratio": 8},
    "FLUX_SCHNELL": {"latent_channels": 128, "downscale_ratio": 16},
    "FLUX_DEV": {"latent_channels": 128, "downscale_ratio": 16},
}

BASE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "SD15": (512, 512),
    "SDXL": (1024, 1024),
    "SDXL_TURBO": (1024, 1024),
    "FLUX_SCHNELL": (1024, 1024),
    "FLUX_DEV": (1024, 1024),
}

GUIDANCE_EMBED: dict[str, bool] = {
    "SD15": False,
    "SDXL": False,
    "SDXL_TURBO": False,
    "FLUX_SCHNELL": False,
    "FLUX_DEV": True,
}

DEFAULT_GUIDANCE: dict[str, float | None] = {
    "SD15": None,
    "SDXL": None,
    "SDXL_TURBO": None,
    "FLUX_SCHNELL": None,
    "FLUX_DEV": 3.5,
}

SAMPLING_PRESETS: dict[str, dict[str, dict[str, float | int]]] = {
    "SD15": {
        "Fast": {"steps": 12, "cfg": 6.5},
        "Balanced": {"steps": 20, "cfg": 7.0},
        "High Quality": {"steps": 30, "cfg": 7.5},
    },
    "SDXL": {
        "Fast": {"steps": 16, "cfg": 5.5},
        "Balanced": {"steps": 24, "cfg": 6.0},
        "High Quality": {"steps": 36, "cfg": 6.5},
    },
    "SDXL_TURBO": {
        "Fast": {"steps": 4, "cfg": 1.0},
        "Balanced": {"steps": 6, "cfg": 1.5},
        "High Quality": {"steps": 8, "cfg": 2.0},
    },
    "FLUX_SCHNELL": {
        "Fast": {"steps": 4, "cfg": 1.0},
        "Balanced": {"steps": 6, "cfg": 1.0},
        "High Quality": {"steps": 8, "cfg": 1.0},
    },
    "FLUX_DEV": {
        "Fast": {"steps": 18, "cfg": 1.0},
        "Balanced": {"steps": 28, "cfg": 1.0},
        "High Quality": {"steps": 36, "cfg": 1.0},
    },
}

_LEGACY_MODEL_INFO_PATTERN = re.compile(r"(\w+)=([^|]+)")


def canonicalize_family(family: str | None) -> str:
    if family is None:
        return "SD15"
    normalized = str(family).strip()
    if normalized == AUTO_FAMILY:
        return "SD15"
    return LEGACY_FAMILY_ALIASES.get(normalized, normalized or "SD15")


def infer_family_from_name(model_name: str | None, fallback: str = "SD15") -> str:
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
    return bool(GUIDANCE_EMBED.get(canonicalize_family(family), False))


def get_latent_spec(model_info: dict[str, Any] | str | None) -> dict[str, int]:
    info = parse_model_info(model_info)
    family = info["family"]
    return {
        "latent_channels": int(info.get("latent_channels", LATENT_SPECS[family]["latent_channels"])),
        "downscale_ratio": int(info.get("downscale_ratio", LATENT_SPECS[family]["downscale_ratio"])),
    }


def get_sampling_preset(model_info: dict[str, Any] | str | None, quality_preset: str) -> dict[str, float | int] | None:
    if quality_preset == "Manual":
        return None
    info = parse_model_info(model_info)
    family = info["family"]
    family_presets = SAMPLING_PRESETS.get(family, SAMPLING_PRESETS["SD15"])
    return family_presets.get(quality_preset)


def parse_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw: dict[str, Any] = {}

    if isinstance(model_info, dict):
        raw = dict(model_info)
    elif isinstance(model_info, str):
        stripped = model_info.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                raw = {
                    key: value.strip()
                    for key, value in _LEGACY_MODEL_INFO_PATTERN.findall(stripped)
                }
            else:
                if isinstance(parsed, dict):
                    raw = dict(parsed)

    family = canonicalize_family(raw.get("family") or infer_family_from_name(raw.get("model_name") or raw.get("ckpt_name") or raw.get("ckpt"), "SD15"))
    width, height = BASE_RESOLUTIONS.get(family, BASE_RESOLUTIONS["SD15"])
    latent_spec = LATENT_SPECS.get(family, LATENT_SPECS["SD15"])
    required_text_encoders = raw.get("required_text_encoders", REQUIRED_TEXT_ENCODERS.get(family, ["clip"]))
    if isinstance(required_text_encoders, str):
        required_text_encoders = [required_text_encoders]

    info: dict[str, Any] = {
        "family": family,
        "text_encoder_type": raw.get("text_encoder_type", TEXT_ENCODER_TYPES.get(family, "clip")),
        "has_embedded_vae": bool(raw.get("has_embedded_vae", False)),
        "required_text_encoders": list(required_text_encoders),
        "required_vae": raw.get("required_vae", REQUIRED_VAE.get(family, "optional")),
        "is_turbo": bool(raw.get("is_turbo", family == "SDXL_TURBO")),
        "guidance_embed": bool(raw.get("guidance_embed", GUIDANCE_EMBED.get(family, False))),
        "guidance": raw.get("guidance", DEFAULT_GUIDANCE.get(family)),
        "latent_channels": int(raw.get("latent_channels", latent_spec["latent_channels"])),
        "downscale_ratio": int(raw.get("downscale_ratio", latent_spec["downscale_ratio"])),
        "base_width": int(raw.get("base_width", width)),
        "base_height": int(raw.get("base_height", height)),
    }

    for key in (
        "model_name",
        "model_source",
        "text_encoder_source",
        "vae_source",
        "load_mode",
        "has_embedded_text_encoder",
    ):
        if key in raw:
            info[key] = raw[key]

    return info


def build_model_info(**kwargs: Any) -> str:
    info = parse_model_info(kwargs)
    for key, value in kwargs.items():
        if value is not None:
            info[key] = value
    info["family"] = canonicalize_family(info.get("family"))
    info.setdefault("text_encoder_type", TEXT_ENCODER_TYPES.get(info["family"], "clip"))
    return json.dumps(info, ensure_ascii=True, sort_keys=True)
