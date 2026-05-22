from __future__ import annotations

import importlib

from .base import RoutingResult, validate_manual_backend
from ...model_profiles.registry import resolve_model_profile
from ...utils.model_info import get_lls_attr, parse_jsonish_info


_BACKENDS = {}
_BUILTINS_LOADED = False


def _ensure_builtin_backends():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module(f"{__package__}.generic")
    importlib.import_module(f"{__package__}.sdxl")
    importlib.import_module(f"{__package__}.flux")
    _BUILTINS_LOADED = True


def register_backend(backend):
    _BACKENDS[str(backend.backend_name).strip().lower()] = backend
    return backend


def get_backend(name: str):
    _ensure_builtin_backends()
    backend = _BACKENDS.get(str(name or "").strip().lower())
    if backend is None:
        raise RuntimeError(f"[LLS] Unknown pro edit backend '{name}'.")
    return backend


def _resolve_profile_and_capabilities(model=None, model_info=None, edit_info=None):
    raw_model_info = parse_jsonish_info(model_info)
    raw_edit_info = parse_jsonish_info(edit_info)
    merged_extra_info = dict(raw_model_info)
    merged_extra_info.update(raw_edit_info)
    profile = resolve_model_profile(
        model=model,
        model_info=model_info,
        extra_info=merged_extra_info,
        checkpoint_name=str(
            merged_extra_info.get("checkpoint_name")
            or merged_extra_info.get("model_name")
            or get_lls_attr(model, "checkpoint_name", None)
            or get_lls_attr(model, "model_name", "")
            or ""
        ),
        family=(
            merged_extra_info.get("family")
            or merged_extra_info.get("model_family")
            or get_lls_attr(model, "family", None)
            or None
        ),
    )
    capabilities = {
        "model_family": profile["family"],
        "model_name": str(
            merged_extra_info.get("checkpoint_name")
            or merged_extra_info.get("model_name")
            or get_lls_attr(model, "checkpoint_name", None)
            or get_lls_attr(model, "model_name", "")
            or ""
        ),
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }
    return profile, capabilities


def _resolve_fallback_backend_name(family: str) -> str:
    normalized = str(family or "").strip().upper()
    if normalized.startswith("SDXL"):
        return "sdxl"
    if normalized.startswith("FLUX"):
        return "flux"
    return "generic"


def resolve_backend(backend_mode: str, *, model=None, model_info=None, edit_info=None):
    _ensure_builtin_backends()
    mode = str(backend_mode or "auto").strip().lower() or "auto"
    profile, capabilities = _resolve_profile_and_capabilities(
        model=model,
        model_info=model_info,
        edit_info=edit_info,
    )
    backend_type = str(profile.get("backend_type") or "none").strip().lower()
    native_backend_name = {
        "sdxl_native": "sdxl",
        "flux_edit": "flux",
    }.get(backend_type)
    if backend_type != "none" and native_backend_name is None:
        raise RuntimeError(f"[LLS] Unsupported pro edit backend_type '{backend_type}'.")

    if mode != "auto":
        backend = get_backend(mode)
        validate_manual_backend(mode, backend, profile)
        routing_reason = "manual.override" if native_backend_name is not None else "manual.family_fallback"
        execution_path = "native_edit" if native_backend_name is not None else "fallback_repair"
        return backend, RoutingResult(backend.backend_name, routing_reason, capabilities, profile, execution_path)

    if native_backend_name is not None:
        backend = get_backend(native_backend_name)
        validate_manual_backend(native_backend_name, backend, profile)
        return backend, RoutingResult(backend.backend_name, "profile.backend_type", capabilities, profile, "native_edit")

    fallback_backend_name = _resolve_fallback_backend_name(str(profile.get("family") or ""))
    backend = get_backend(fallback_backend_name)
    validate_manual_backend(fallback_backend_name, backend, profile)
    return backend, RoutingResult(backend.backend_name, "profile.family_fallback", capabilities, profile, "fallback_repair")
