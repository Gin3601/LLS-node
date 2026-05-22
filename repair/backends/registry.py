from __future__ import annotations

import importlib

from .base import RoutingResult, validate_backend
from ...utils.model_info import canonicalize_family, parse_jsonish_info, resolve_edit_capabilities


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
        raise RuntimeError(f"[LLS] Unknown repair backend '{name}'.")
    return backend


def _build_profile(capabilities: dict[str, object]) -> dict[str, object]:
    return {
        "family": capabilities["model_family"],
        "role": capabilities["model_role"],
        "profile_id": capabilities["profile_id"],
        "backend_type": capabilities["backend_type"],
        "sampler_strategy": capabilities["sampler_strategy"],
        "loader_strategy": capabilities["loader_strategy"],
        "supports_inpaint_native": capabilities["supports_inpaint_native"],
        "supports_image_edit_native": capabilities["supports_image_edit_native"],
        "preferred_edit_backend": capabilities["preferred_edit_backend"],
    }


def _resolve_profile_and_capabilities(model=None, model_info=None, repair_info=None):
    raw_model_info = parse_jsonish_info(model_info)
    raw_repair_info = parse_jsonish_info(repair_info)
    merged_info = dict(raw_model_info)
    merged_info.update(raw_repair_info)
    capabilities = resolve_edit_capabilities(model=model, model_info=merged_info)
    return _build_profile(capabilities), capabilities


def _resolve_fallback_backend_name(family: str) -> str:
    normalized = canonicalize_family(family)
    if normalized.startswith("SDXL"):
        return "sdxl"
    if normalized.startswith("FLUX"):
        return "flux"
    return "generic"


def resolve_backend(repair_kernel: str, *, model=None, model_info=None, repair_info=None):
    _ensure_builtin_backends()
    profile, capabilities = _resolve_profile_and_capabilities(
        model=model,
        model_info=model_info,
        repair_info=repair_info,
    )

    if str(repair_kernel or "") == "native_fill":
        native_backend_name = {
            "sdxl_native": "sdxl",
            "flux_edit": "flux",
        }.get(str(profile.get("backend_type") or "").strip().lower())
        if native_backend_name is not None:
            backend = get_backend(native_backend_name)
            validate_backend(native_backend_name, backend, profile)
            return backend, RoutingResult(
                backend.backend_name,
                "profile.backend_type",
                capabilities,
                profile,
                "native_repair",
            )

    fallback_backend_name = _resolve_fallback_backend_name(str(profile.get("family") or ""))
    backend = get_backend(fallback_backend_name)
    validate_backend(fallback_backend_name, backend, profile)
    return backend, RoutingResult(
        backend.backend_name,
        "profile.family_fallback",
        capabilities,
        profile,
        "fallback_repair",
    )
