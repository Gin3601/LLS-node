from __future__ import annotations

import importlib

from .base import RoutingResult, validate_manual_backend
from ...utils.model_info import parse_jsonish_info, resolve_edit_capabilities


_BACKENDS = {}
_BUILTINS_LOADED = False


def _ensure_builtin_backends():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
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


def _normalize_capabilities(model=None, model_info=None, edit_info=None):
    resolved = resolve_edit_capabilities(model=model, model_info=model_info)
    raw_edit_info = parse_jsonish_info(edit_info)

    if raw_edit_info.get("model_family"):
        resolved["model_family"] = str(raw_edit_info["model_family"])
    if raw_edit_info.get("model_name"):
        resolved["model_name"] = str(raw_edit_info["model_name"])
    if raw_edit_info.get("model_role"):
        resolved["model_role"] = str(raw_edit_info["model_role"])
    if "supports_inpaint_native" in raw_edit_info:
        resolved["supports_inpaint_native"] = bool(raw_edit_info["supports_inpaint_native"])
    if "supports_image_edit_native" in raw_edit_info:
        resolved["supports_image_edit_native"] = bool(raw_edit_info["supports_image_edit_native"])
    if raw_edit_info.get("preferred_edit_backend"):
        resolved["preferred_edit_backend"] = str(raw_edit_info["preferred_edit_backend"]).strip().lower()

    return resolved, raw_edit_info


def resolve_backend(backend_mode: str, *, model=None, model_info=None, edit_info=None):
    _ensure_builtin_backends()
    mode = str(backend_mode or "auto").strip().lower() or "auto"
    capabilities, raw_edit_info = _normalize_capabilities(
        model=model,
        model_info=model_info,
        edit_info=edit_info,
    )

    if mode != "auto":
        backend = get_backend(mode)
        validate_manual_backend(mode, backend, capabilities)
        return backend, RoutingResult(backend.backend_name, "manual.override", capabilities)

    backend_name = str(raw_edit_info.get("backend_name") or "").strip().lower()
    if backend_name:
        backend = get_backend(backend_name)
        validate_manual_backend(backend_name, backend, capabilities)
        return backend, RoutingResult(backend.backend_name, "edit_info.backend_name", capabilities)

    preferred = str(capabilities.get("preferred_edit_backend") or "").strip().lower()
    if preferred:
        backend = get_backend(preferred)
        validate_manual_backend(preferred, backend, capabilities)
        return backend, RoutingResult(backend.backend_name, "model.preferred_edit_backend", capabilities)

    for candidate_name in sorted(_BACKENDS):
        backend = get_backend(candidate_name)
        if backend.supports(capabilities):
            return backend, RoutingResult(backend.backend_name, "model.capability_match", capabilities)

    raise RuntimeError(
        "[LLS] No professional edit backend matched the current model capability. "
        "Use an edit-capable SDXL or FLUX model, or provide explicit capability metadata."
    )
