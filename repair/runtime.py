from __future__ import annotations

import importlib
from typing import Any

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception as exc:  # pragma: no cover - runtime-only import
    comfy_core_nodes = None
    _CORE_NODES_ERR = exc
else:  # pragma: no cover - runtime-only import
    _CORE_NODES_ERR = None

try:
    nodes_differential_diffusion = importlib.import_module("comfy_extras.nodes_differential_diffusion")
except Exception as exc:  # pragma: no cover - runtime-only import
    nodes_differential_diffusion = None
    _DIFFUSION_ERR = exc
else:  # pragma: no cover - runtime-only import
    _DIFFUSION_ERR = None


def _require_class(module: Any, class_name: str, import_error: Exception | None = None):
    if module is None:
        raise RuntimeError(
            f"[LLS] Required ComfyUI runtime component '{class_name}' is unavailable."
        ) from import_error
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"[LLS] Required ComfyUI runtime component '{class_name}' is unavailable."
        ) from import_error
    return cls


def _unwrap_result(result: Any):
    if hasattr(result, "result"):
        return result.result
    return result


def _unwrap_first(result: Any):
    values = _unwrap_result(result)
    if isinstance(values, tuple):
        return values[0]
    if isinstance(values, list):
        return values[0]
    return values


def _unwrap_three(result: Any):
    values = _unwrap_result(result)
    if isinstance(values, tuple):
        return values[:3]
    if isinstance(values, list):
        return tuple(values[:3])
    raise RuntimeError("[LLS] Expected a three-value runtime result from ComfyUI.")


def encode_inpaint_conditioning(positive, negative, vae, pixels, mask, *, noise_mask: bool = True):
    node_cls = _require_class(comfy_core_nodes, "InpaintModelConditioning", _CORE_NODES_ERR)
    node = node_cls()
    if not hasattr(node, "encode"):
        raise RuntimeError(
            "[LLS] ComfyUI InpaintModelConditioning is missing the expected encode() method."
        )
    return _unwrap_three(
        node.encode(
            positive,
            negative,
            pixels,
            vae,
            mask,
            bool(noise_mask),
        )
    )


def apply_differential_diffusion(model, *, strength: float = 1.0):
    node_cls = _require_class(
        nodes_differential_diffusion,
        "DifferentialDiffusion",
        _DIFFUSION_ERR,
    )
    node = node_cls()
    if hasattr(node, "execute"):
        return _unwrap_first(node.execute(model, float(strength)))
    if hasattr(node, "apply"):
        return _unwrap_first(node.apply(model, float(strength)))
    if hasattr(node, "patch"):
        return _unwrap_first(node.patch(model, float(strength)))
    if hasattr(node_cls, "execute"):
        return _unwrap_first(node_cls.execute(model, float(strength)))
    raise RuntimeError(
        "[LLS] ComfyUI DifferentialDiffusion is missing a supported execute/apply method."
    )
