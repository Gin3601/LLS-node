"""
LLS / Upscale
=============
功能域：图像超分辨率放大
对应功能分类总览第 5 节：图像处理与后处理 → 超分（Upscale）

包含节点：
  - LLSUpscaleSwitcher  : 在 upscale_model 与 PyTorch 插值之间切换
"""
from __future__ import annotations

from typing import Any

from ..utils.model_info import info_to_json

# ---------- 从根模块导入共享工具函数 ----------

from ..nodes import (
    _torch_inference_mode,
    _require_torch,
    _require_comfy_modules,
    _clamp_image,
    _sorted_unique,
    _is_oom_exception,
    _make_progress_bar,
    _supports_argument,
)

# ---------- 可选依赖（防御性导入） ----------

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import folder_paths
except Exception as exc:
    folder_paths = None
    _FOLDER_PATHS_IMPORT_ERROR = exc
else:
    _FOLDER_PATHS_IMPORT_ERROR = None

try:
    import comfy.model_management as model_management
    import comfy.utils as comfy_utils
except Exception as exc:
    model_management = None
    comfy_utils = None
    _COMFY_IMPORT_ERROR = exc
else:
    _COMFY_IMPORT_ERROR = None

# ---------- 常量 ----------

UPSCALE_MODE_CHOICES = ["none", "interpolation", "upscale_model", "latent_upscale", "tile_upscale", "pytorch"]
INTERPOLATION_CHOICES = ["nearest", "bilinear", "bicubic", "area"]
NO_UPSCALE_MODEL_PLACEHOLDER = "(no upscale models found)"


# ---------- 内部工具函数（仅本模块专用） ----------

def _get_upscale_model_names() -> list[str]:
    if folder_paths is None:
        return [NO_UPSCALE_MODEL_PLACEHOLDER]
    try:
        names = list(folder_paths.get_filename_list("upscale_models"))
    except Exception:
        names = []
    return names if names else [NO_UPSCALE_MODEL_PLACEHOLDER]


def _has_real_upscale_models(model_names: list[str]) -> bool:
    return any(name and name != NO_UPSCALE_MODEL_PLACEHOLDER for name in model_names)


def _is_missing_upscale_model_selection(model_name: str | None) -> bool:
    return model_name in (None, "", NO_UPSCALE_MODEL_PLACEHOLDER)


def _resolve_upscale_model_path(model_name: str) -> str:
    _require_comfy_modules()
    if model_name == NO_UPSCALE_MODEL_PLACEHOLDER:
        raise RuntimeError(
            "No upscale models were found in ComfyUI/models/upscale_models/. "
            "Add a model there or switch mode to 'pytorch'."
        )
    try:
        get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_full_path_or_raise):
            return get_full_path_or_raise("upscale_models", model_name)
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        if model_path:
            return model_path
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Upscale model '{model_name}' was not found in ComfyUI/models/upscale_models/."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve upscale model '{model_name}': {exc}"
        ) from exc
    raise RuntimeError(
        f"Upscale model '{model_name}' was not found in ComfyUI/models/upscale_models/."
    )


def _load_upscale_model_descriptor(model_name: str):
    _require_comfy_modules()
    model_path = _resolve_upscale_model_path(model_name)
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ComfyUI's upscale model loader. "
            "Make sure the node runs inside a recent ComfyUI installation."
        ) from exc
    try:
        loader = ModelLoader()
        descriptor = None
        if hasattr(comfy_utils, "load_torch_file") and hasattr(loader, "load_from_state_dict"):
            state_dict = comfy_utils.load_torch_file(model_path, safe_load=True)
            if (
                isinstance(state_dict, dict)
                and "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict
                and hasattr(comfy_utils, "state_dict_prefix_replace")
            ):
                state_dict = comfy_utils.state_dict_prefix_replace(state_dict, {"module.": ""})
            descriptor = loader.load_from_state_dict(state_dict)
        elif hasattr(loader, "load_from_file"):
            descriptor = loader.load_from_file(model_path)
        else:
            raise RuntimeError("No supported spandrel loading API is available.")
    except Exception as exc:
        raise RuntimeError(f"Failed to load upscale model '{model_name}': {exc}") from exc
    if isinstance(ImageModelDescriptor, type) and not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(
            f"File '{model_name}' is not a valid image upscale model supported by ComfyUI."
        )
    if not callable(descriptor):
        raise RuntimeError(f"Loaded upscale model '{model_name}' is not callable.")
    eval_fn = getattr(descriptor, "eval", None)
    if callable(eval_fn):
        maybe_descriptor = eval_fn()
        if maybe_descriptor is not None:
            descriptor = maybe_descriptor
    return descriptor


def _get_upscale_amount(upscale_model: Any) -> float:
    scale = getattr(upscale_model, "scale", None)
    if scale is None:
        scale = getattr(getattr(upscale_model, "model", None), "scale", None)
    if scale is None:
        raise RuntimeError("Loaded upscale model does not expose a scale factor.")
    return float(scale)


def _estimate_model_memory(image_bchw: Any, upscale_model: Any) -> int:
    if model_management is None:
        return 0
    estimate = 0
    module = getattr(upscale_model, "model", None)
    try:
        if module is not None and hasattr(model_management, "module_size"):
            estimate += int(model_management.module_size(module))
        if hasattr(image_bchw, "element_size") and hasattr(image_bchw, "nelement"):
            element_size = int(image_bchw.element_size())
            estimate += int(
                (512 * 512 * 3) * element_size * max(_get_upscale_amount(upscale_model), 1.0) * 384.0
            )
            estimate += int(image_bchw.nelement() * element_size)
    except Exception:
        return 0
    return estimate


def _get_tiled_scale_steps(image_bchw: Any, tile: int, overlap: int) -> int:
    if comfy_utils is None or not hasattr(comfy_utils, "get_tiled_scale_steps"):
        return 1
    return max(
        1,
        int(
            comfy_utils.get_tiled_scale_steps(
                image_bchw.shape[3],
                image_bchw.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
        ),
    )


def _next_tile_size(current_tile: int) -> int:
    halved = current_tile // 2
    if halved < 128:
        return 0
    return max(128, (halved // 64) * 64)


# ---------- 节点类 ----------

class LLSUpscaleSwitcher:
    """
    在 upscale_model 与 PyTorch 插值之间切换的超分节点。

    - upscale_model 模式：调用 spandrel 加载 ESRGAN 等模型，支持分块推理防 OOM。
    - pytorch 模式：使用 torch.nn.functional.interpolate 做普通插值放大。
    """

    CATEGORY = "LLS/Upscale"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "upscale_info")
    DESCRIPTION = "Switch between ComfyUI upscale models and PyTorch interpolation."

    @classmethod
    def INPUT_TYPES(cls):
        model_names = _sorted_unique(_get_upscale_model_names())
        default_mode = "upscale_model" if _has_real_upscale_models(model_names) else "interpolation"
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (UPSCALE_MODE_CHOICES, {"default": default_mode}),
                "scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "interpolation": (INTERPOLATION_CHOICES, {"default": "bilinear"}),
                "model_name": (model_names,),
                "tile": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
            },
        }

    def upscale(self, image, mode: str, scale: float, interpolation: str, model_name: str, tile: int, overlap: int):
        requested_mode = mode
        warning = None
        if mode not in UPSCALE_MODE_CHOICES:
            raise RuntimeError(f"Unsupported mode '{mode}'. Expected one of {UPSCALE_MODE_CHOICES}.")
        if mode == "none":
            return (
                image,
                info_to_json(
                    {
                        "mode": "none",
                        "requested_mode": requested_mode,
                        "scale": 1.0,
                        "upscaled": False,
                    }
                ),
            )
        if mode in {"interpolation", "pytorch"}:
            result = self._upscale_with_pytorch(image=image, scale=scale, interpolation=interpolation)[0]
            return (
                result,
                info_to_json(
                    {
                        "mode": "interpolation",
                        "requested_mode": requested_mode,
                        "scale": scale,
                        "interpolation": interpolation,
                        "upscaled": True,
                    }
                ),
            )
        if mode == "latent_upscale":
            warning = "latent_upscale_not_available_on_IMAGE_input_fallback_to_interpolation"
            result = self._upscale_with_pytorch(image=image, scale=scale, interpolation=interpolation)[0]
            return (
                result,
                info_to_json(
                    {
                        "mode": "interpolation",
                        "requested_mode": requested_mode,
                        "scale": scale,
                        "interpolation": interpolation,
                        "warning": warning,
                        "upscaled": True,
                    }
                ),
            )
        if _is_missing_upscale_model_selection(model_name):
            warning = "no_upscale_model_found_fallback_to_interpolation"
            print(
                "[LLS] WARNING: No valid upscale model is selected. "
                "Falling back to interpolation mode."
            )
            result = self._upscale_with_pytorch(image=image, scale=scale, interpolation=interpolation)[0]
            return (
                result,
                info_to_json(
                    {
                        "mode": "interpolation",
                        "requested_mode": requested_mode,
                        "scale": scale,
                        "interpolation": interpolation,
                        "warning": warning,
                        "upscaled": True,
                    }
                ),
            )
        result = self._upscale_with_model(image=image, model_name=model_name, tile=tile, overlap=overlap)[0]
        actual_mode = "tile_upscale" if requested_mode == "tile_upscale" else "upscale_model"
        return (
            result,
            info_to_json(
                {
                    "mode": actual_mode,
                    "requested_mode": requested_mode,
                    "scale": scale,
                    "model_name": model_name,
                    "tile": tile,
                    "overlap": overlap,
                    "warning": warning,
                    "upscaled": True,
                }
            ),
        )

    def _upscale_with_pytorch(self, image, scale: float, interpolation: str):
        _require_torch()
        if F is None:
            raise RuntimeError("torch.nn.functional is not available.")
        if interpolation not in INTERPOLATION_CHOICES:
            raise RuntimeError(f"Unsupported interpolation '{interpolation}'. Expected one of {INTERPOLATION_CHOICES}.")
        if scale <= 0:
            raise RuntimeError("Scale must be greater than 0.")
        try:
            with _torch_inference_mode():
                image_bchw = image.permute(0, 3, 1, 2).contiguous()
                interpolate_kwargs: dict[str, Any] = {"scale_factor": float(scale), "mode": interpolation}
                if interpolation in {"bilinear", "bicubic"}:
                    interpolate_kwargs["align_corners"] = False
                result = F.interpolate(image_bchw, **interpolate_kwargs)
                result = result.permute(0, 2, 3, 1).contiguous()
                if model_management is not None and hasattr(model_management, "intermediate_dtype"):
                    try:
                        result = result.to(dtype=model_management.intermediate_dtype())
                    except Exception:
                        pass
                return (_clamp_image(result),)
        except Exception as exc:
            if _is_oom_exception(exc):
                raise RuntimeError(
                    "Out of memory while running PyTorch interpolation. "
                    "Reduce the scale value, or switch to 'upscale_model' mode and lower tile."
                ) from exc
            raise RuntimeError(f"PyTorch interpolation upscale failed: {exc}") from exc

    def _upscale_with_model(self, image, model_name: str, tile: int, overlap: int):
        _require_torch()
        _require_comfy_modules()
        try:
            upscale_model = _load_upscale_model_descriptor(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare upscale model '{model_name}': {exc}") from exc

        with _torch_inference_mode():
            image_bchw = image.permute(0, 3, 1, 2).contiguous()
            device = model_management.get_torch_device()
            output_device = None
            if hasattr(model_management, "intermediate_device"):
                try:
                    output_device = model_management.intermediate_device()
                except Exception:
                    output_device = None
            image_bchw = image_bchw.to(device)
            if hasattr(upscale_model, "to"):
                upscale_model = upscale_model.to(device)
            try:
                memory_required = _estimate_model_memory(image_bchw, upscale_model)
                if memory_required > 0 and hasattr(model_management, "free_memory"):
                    try:
                        model_management.free_memory(memory_required, device)
                    except Exception:
                        pass
                upscale_amount = _get_upscale_amount(upscale_model)
                current_tile = int(tile)
                last_oom: Exception | None = None

                while current_tile >= 128:
                    steps = image_bchw.shape[0] * _get_tiled_scale_steps(image_bchw, current_tile, overlap)
                    progress_bar = _make_progress_bar(steps)
                    tiled_kwargs = {
                        "tile_x": current_tile,
                        "tile_y": current_tile,
                        "overlap": int(overlap),
                        "upscale_amount": upscale_amount,
                        "pbar": progress_bar,
                    }
                    tiled_scale = getattr(comfy_utils, "tiled_scale", None)
                    if not callable(tiled_scale):
                        raise RuntimeError("ComfyUI tiled_scale helper is not available.")
                    if _supports_argument(tiled_scale, "output_device") and output_device is not None:
                        tiled_kwargs["output_device"] = output_device
                    try:
                        result = tiled_scale(
                            image_bchw,
                            lambda tile_image: upscale_model(tile_image.float()),
                            **tiled_kwargs,
                        )
                        result = result.permute(0, 2, 3, 1).contiguous()
                        if model_management is not None and hasattr(model_management, "intermediate_dtype"):
                            try:
                                result = result.to(dtype=model_management.intermediate_dtype())
                            except Exception:
                                pass
                        return (_clamp_image(result),)
                    except Exception as exc:
                        if _is_oom_exception(exc):
                            last_oom = exc
                            next_tile = _next_tile_size(current_tile)
                            if next_tile <= 0 or next_tile == current_tile:
                                break
                            current_tile = next_tile
                            continue
                        raise RuntimeError(
                            f"Upscale model '{model_name}' failed during tiled inference: {exc}"
                        ) from exc

                raise RuntimeError(
                    f"Out of memory while running upscale model '{model_name}'. "
                    f"Try reducing tile below {tile} or lowering overlap."
                ) from last_oom
            finally:
                if hasattr(upscale_model, "to"):
                    try:
                        upscale_model.to("cpu")
                    except Exception:
                        pass


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS = {
    "LLSUpscaleSwitcher": LLSUpscaleSwitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSUpscaleSwitcher": "LLS Upscale Switcher",
}
