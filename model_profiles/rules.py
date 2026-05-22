from __future__ import annotations

try:
    from ..utils.model_info import (
        canonicalize_family,
        infer_family_from_name,
        infer_model_role_from_name,
        is_flux_family,
        is_sdxl_family,
    )
except ImportError:
    from utils.model_info import (
        canonicalize_family,
        infer_family_from_name,
        infer_model_role_from_name,
        is_flux_family,
        is_sdxl_family,
    )


_ROLE_TO_PROFILE = {
    ("SDXL", "inpaint"): ("sdxl_inpaint", "sdxl_native", "standard_k", "sdxl_checkpoint", True, False, "sdxl"),
    ("SDXL", "edit"): ("sdxl_edit", "sdxl_native", "standard_k", "sdxl_checkpoint", True, True, "sdxl"),
    ("SDXL", "fill"): ("sdxl_edit", "sdxl_native", "standard_k", "sdxl_checkpoint", True, True, "sdxl"),
    ("FLUX_DEV", "inpaint"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", True, True, "flux"),
    ("FLUX_DEV", "edit"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_DEV", "fill"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_SCHNELL", "inpaint"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", True, True, "flux"),
    ("FLUX_SCHNELL", "edit"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_SCHNELL", "fill"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
}


def _base_profile_for_family(family: str) -> dict[str, object]:
    resolved_family = canonicalize_family(family)
    if is_sdxl_family(resolved_family):
        return {
            "profile_id": "sdxl_base",
            "family": resolved_family,
            "role": "base",
            "backend_type": "none",
            "sampler_strategy": "standard_k",
            "loader_strategy": "sdxl_checkpoint",
            "supports_inpaint_native": False,
            "supports_image_edit_native": False,
            "preferred_edit_backend": None,
        }
    if is_flux_family(resolved_family):
        return {
            "profile_id": "flux_base",
            "family": resolved_family,
            "role": "base",
            "backend_type": "none",
            "sampler_strategy": "flux_guided",
            "loader_strategy": "flux_split_or_bundle",
            "supports_inpaint_native": False,
            "supports_image_edit_native": False,
            "preferred_edit_backend": None,
        }
    return {
        "profile_id": "sd15_base",
        "family": "SD1.5",
        "role": "base",
        "backend_type": "none",
        "sampler_strategy": "standard_k",
        "loader_strategy": "sd15_checkpoint",
        "supports_inpaint_native": False,
        "supports_image_edit_native": False,
        "preferred_edit_backend": None,
    }


def match_builtin_profile(model_name: str | None, family: str | None, role: str | None = None) -> dict[str, object]:
    resolved_family = canonicalize_family(family or infer_family_from_name(model_name, "SD1.5"))
    resolved_role = str(role or infer_model_role_from_name(model_name, resolved_family) or "base").strip().lower()
    matched = _ROLE_TO_PROFILE.get((resolved_family, resolved_role))
    if matched is None:
        return _base_profile_for_family(resolved_family)
    profile_id, backend_type, sampler_strategy, loader_strategy, supports_inpaint_native, supports_image_edit_native, preferred_backend = matched
    return {
        "profile_id": profile_id,
        "family": resolved_family,
        "role": resolved_role,
        "backend_type": backend_type,
        "sampler_strategy": sampler_strategy,
        "loader_strategy": loader_strategy,
        "supports_inpaint_native": supports_inpaint_native,
        "supports_image_edit_native": supports_image_edit_native,
        "preferred_edit_backend": preferred_backend,
    }
