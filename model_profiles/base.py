from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _normalize_text(value: Any, default: str = "", *, lower: bool = False) -> str:
    if value is None:
        return default
    normalized = str(value).strip()
    if not normalized:
        return default
    return normalized.lower() if lower else normalized


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    family: str
    role: str
    backend_type: str
    sampler_strategy: str
    loader_strategy: str
    supports_inpaint_native: bool
    supports_image_edit_native: bool
    preferred_edit_backend: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_profile_dict(value: dict[str, Any]) -> dict[str, Any]:
    preferred_backend = value.get("preferred_edit_backend")
    return {
        "profile_id": _normalize_text(value.get("profile_id"), lower=True),
        "family": _normalize_text(value.get("family") or value.get("model_family"), "SD1.5"),
        "role": _normalize_text(value.get("role") or value.get("model_role"), "base", lower=True),
        "backend_type": _normalize_text(value.get("backend_type"), "none", lower=True),
        "sampler_strategy": _normalize_text(value.get("sampler_strategy"), "standard_k", lower=True),
        "loader_strategy": _normalize_text(value.get("loader_strategy"), lower=True),
        "supports_inpaint_native": _coerce_bool(value.get("supports_inpaint_native", False)),
        "supports_image_edit_native": _coerce_bool(value.get("supports_image_edit_native", False)),
        "preferred_edit_backend": (
            None if preferred_backend in (None, "") else _normalize_text(preferred_backend, lower=True)
        ),
    }


def build_model_profile(**kwargs: Any) -> ModelProfile:
    normalized = normalize_profile_dict(kwargs)
    return ModelProfile(**normalized)
