from __future__ import annotations

from .backend_base import UniversalBackendBase
from .backend_flux import FluxBackend
from .backend_sd15 import SD15Backend
from .backend_sdxl import SDXLBackend


_BACKEND_TYPES: dict[str, type[UniversalBackendBase]] = {
    "SD1.5": SD15Backend,
    "SDXL": SDXLBackend,
    "FLUX": FluxBackend,
}


def get_backend(model_family: str) -> UniversalBackendBase:
    try:
        backend_type = _BACKEND_TYPES[model_family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. Expected one of {tuple(_BACKEND_TYPES.keys())}."
        ) from exc
    return backend_type()
