"""
LLS / Latent
============
功能域：Latent 空间操作（对应功能分类总览第 4 节）

CATEGORY = "LLS/Latent"
"""
from __future__ import annotations

try:
    import torch
except Exception as exc:
    torch = None
    _TORCH_ERR = exc
else:
    _TORCH_ERR = None

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

from ..utils.model_info import (
    LLS_MODEL_INFO_TYPE,
    SIZE_PRESET_AUTO,
    get_family_defaults,
    get_latent_spec,
    info_to_json,
    parse_model_info,
)


_SIZE_PRESETS = [
    SIZE_PRESET_AUTO,
    "Custom",
    "512x512",
    "768x768",
    "512x768",
    "768x512",
    "1024x1024",
    "832x1216",
    "1216x832",
    "896x1152",
    "1152x896",
    "1024x576",
    "576x1024",
]


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


class LLSSimpleEmptyLatent:
    """
    生成空白 Latent，作为 txt2img 推理起点。

    - `Family Default` 时按照 model_info.family 自动取默认尺寸。
    - `Custom` 时才使用手填 width / height。
    """

    CATEGORY = "LLS/Latent"
    FUNCTION = "create_empty_latent"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "latent_info")
    DESCRIPTION = (
        "Create an empty latent image as the starting point for txt2img. "
        "Uses family defaults when size_preset is 'Family Default'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_preset": (_SIZE_PRESETS, {"default": SIZE_PRESET_AUTO}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "model_info": (LLS_MODEL_INFO_TYPE,),
            },
        }

    def create_empty_latent(
        self,
        size_preset: str,
        width: int,
        height: int,
        batch_size: int,
        model_info=None,
    ):
        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. Make sure this node runs inside a ComfyUI environment."
            ) from _TORCH_ERR

        info = parse_model_info(model_info)
        family_defaults = get_family_defaults(info["family"])
        latent_spec = get_latent_spec(info)
        latent_channels = latent_spec["latent_channels"]
        downscale_ratio = latent_spec["downscale_ratio"]

        if size_preset == SIZE_PRESET_AUTO:
            width = int(family_defaults["default_width"])
            height = int(family_defaults["default_height"])
        elif size_preset != "Custom":
            try:
                width, height = [int(part) for part in size_preset.split("x", 1)]
            except Exception as exc:
                raise RuntimeError(f"[LLS] Invalid size_preset value '{size_preset}': {exc}") from exc

        width = _round_to_multiple(int(width), downscale_ratio)
        height = _round_to_multiple(int(height), downscale_ratio)

        device = "cpu"
        dtype = torch.float32
        if model_management is not None:
            try:
                device = model_management.intermediate_device()
            except Exception:
                device = "cpu"
            try:
                dtype = model_management.intermediate_dtype()
            except Exception:
                dtype = torch.float32

        latent = torch.zeros(
            [batch_size, latent_channels, height // downscale_ratio, width // downscale_ratio],
            device=device,
            dtype=dtype,
        )

        latent_info = info_to_json(
            {
                "family": info["family"],
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "latent_channels": latent_channels,
                "downscale_ratio": downscale_ratio,
                "size_preset": size_preset,
            }
        )

        return (
            {"samples": latent, "downscale_ratio_spacial": downscale_ratio},
            width,
            height,
            latent_info,
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleEmptyLatent": LLSSimpleEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleEmptyLatent": "LLS Simple Empty Latent",
}
