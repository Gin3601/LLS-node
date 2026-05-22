from __future__ import annotations

from typing import Any

from .base import build_model_profile
from .rules import match_builtin_profile

try:
    from ..utils.model_info import canonicalize_family, get_lls_attr, parse_jsonish_info
except ImportError:
    from utils.model_info import canonicalize_family, get_lls_attr, parse_jsonish_info


def _pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _derive_role_from_legacy(
    family: str | None,
    role: Any,
    *,
    supports_inpaint_native: Any,
    supports_image_edit_native: Any,
    preferred_edit_backend: Any,
) -> str | None:
    normalized_role = str(role or "").strip().lower()
    if normalized_role and normalized_role != "base":
        return normalized_role

    resolved_family = canonicalize_family(family)
    preferred = str(preferred_edit_backend or "").strip().lower()
    inpaint_native = supports_inpaint_native not in (None, "", False, 0, "0", "false", "False")
    image_edit_native = supports_image_edit_native not in (None, "", False, 0, "0", "false", "False")

    if resolved_family.startswith("SDXL"):
        if image_edit_native:
            return "edit"
        if inpaint_native or preferred == "sdxl":
            return "inpaint"
    if resolved_family.startswith("FLUX"):
        if inpaint_native:
            return "inpaint"
        if image_edit_native or preferred == "flux":
            return "edit"
    return normalized_role or None


def _extract_profile_overrides(model=None, model_info=None, extra_info=None) -> dict[str, object]:
    raw = parse_jsonish_info(model_info)
    extra = parse_jsonish_info(extra_info)

    return {
        "profile_id": _pick(extra.get("profile_id"), raw.get("profile_id"), get_lls_attr(model, "profile_id", None)),
        "family": _pick(
            extra.get("family"),
            extra.get("model_family"),
            raw.get("family"),
            raw.get("model_family"),
            get_lls_attr(model, "family", None),
        ),
        "role": _pick(
            extra.get("role"),
            extra.get("model_role"),
            raw.get("role"),
            raw.get("model_role"),
            get_lls_attr(model, "model_role", None),
        ),
        "backend_type": _pick(extra.get("backend_type"), raw.get("backend_type"), get_lls_attr(model, "backend_type", None)),
        "sampler_strategy": _pick(
            extra.get("sampler_strategy"),
            raw.get("sampler_strategy"),
            get_lls_attr(model, "sampler_strategy", None),
        ),
        "loader_strategy": _pick(extra.get("loader_strategy"), raw.get("loader_strategy"), get_lls_attr(model, "loader_strategy", None)),
        "supports_inpaint_native": (
            extra["supports_inpaint_native"] if "supports_inpaint_native" in extra
            else raw["supports_inpaint_native"] if "supports_inpaint_native" in raw
            else get_lls_attr(model, "supports_inpaint_native", None)
        ),
        "supports_image_edit_native": (
            extra["supports_image_edit_native"] if "supports_image_edit_native" in extra
            else raw["supports_image_edit_native"] if "supports_image_edit_native" in raw
            else get_lls_attr(model, "supports_image_edit_native", None)
        ),
        "preferred_edit_backend": _pick(
            extra.get("preferred_edit_backend"),
            raw.get("preferred_edit_backend"),
            get_lls_attr(model, "preferred_edit_backend", None),
        ),
        "model_name": _pick(
            extra.get("checkpoint_name"),
            extra.get("model_name"),
            raw.get("checkpoint_name"),
            raw.get("model_name"),
            raw.get("ckpt_name"),
            raw.get("ckpt"),
            get_lls_attr(model, "checkpoint_name", None),
            get_lls_attr(model, "model_name", None),
        ),
    }


def resolve_model_profile(
    model=None,
    model_info=None,
    extra_info=None,
    checkpoint_name: str | None = None,
    family: str | None = None,
) -> dict[str, object]:
    overrides = _extract_profile_overrides(model=model, model_info=model_info, extra_info=extra_info)
    resolved_family = canonicalize_family(family or overrides.get("family") or "SD1.5")
    resolved_role = _derive_role_from_legacy(
        resolved_family,
        overrides.get("role"),
        supports_inpaint_native=overrides.get("supports_inpaint_native"),
        supports_image_edit_native=overrides.get("supports_image_edit_native"),
        preferred_edit_backend=overrides.get("preferred_edit_backend"),
    )
    model_name = str(checkpoint_name or overrides.get("model_name") or "").strip()

    merged = dict(match_builtin_profile(model_name, resolved_family, resolved_role))
    if overrides.get("profile_id") not in (None, ""):
        merged["profile_id"] = overrides["profile_id"]
    if overrides.get("family") not in (None, ""):
        merged["family"] = canonicalize_family(overrides["family"])
    if overrides.get("role") not in (None, ""):
        merged["role"] = str(overrides["role"]).strip().lower()
    elif resolved_role:
        merged["role"] = str(resolved_role).strip().lower()
    if overrides.get("backend_type") not in (None, ""):
        merged["backend_type"] = overrides["backend_type"]
    if overrides.get("sampler_strategy") not in (None, ""):
        merged["sampler_strategy"] = overrides["sampler_strategy"]
    if overrides.get("loader_strategy") not in (None, ""):
        merged["loader_strategy"] = overrides["loader_strategy"]
    if overrides.get("preferred_edit_backend") not in (None, ""):
        merged["preferred_edit_backend"] = overrides["preferred_edit_backend"]
    for key in ("supports_inpaint_native", "supports_image_edit_native"):
        if overrides.get(key) is not None:
            merged[key] = overrides[key]

    return build_model_profile(**merged).to_dict()
