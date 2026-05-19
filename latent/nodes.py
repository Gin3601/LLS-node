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
    SIZE_PRESET_AUTO,
    get_latent_spec,
    parse_model_info,
)
from ..utils.task_context import LLS_TASK_CONTEXT_TYPE, parse_task_context, update_task_context


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

    - `Family Default` 时按照 task_context 推荐尺寸自动取默认值。
    - `Custom` 时才使用手填 width / height。
    """

    CATEGORY = "LLS/Latent"
    FUNCTION = "create_empty_latent"
    RETURN_TYPES = ("LATENT", "INT", "INT", LLS_TASK_CONTEXT_TYPE)
    RETURN_NAMES = ("latent", "width", "height", "task_context")
    DESCRIPTION = (
        "Create an empty latent image as the starting point for txt2img. "
        "Uses task_context recommendations when size_preset is 'Family Default'."
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
                "task_context": (LLS_TASK_CONTEXT_TYPE,),
            },
        }

    def create_empty_latent(
        self,
        size_preset: str,
        width: int,
        height: int,
        batch_size: int,
        task_context=None,
    ):
        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. Make sure this node runs inside a ComfyUI environment."
            ) from _TORCH_ERR

        context = parse_task_context(task_context)
        info = parse_model_info(context)
        latent_spec = get_latent_spec(info)
        latent_channels = latent_spec["latent_channels"]
        downscale_ratio = latent_spec["downscale_ratio"]

        if size_preset == SIZE_PRESET_AUTO:
            width = int(context.get("recommended_width", info["default_width"]))
            height = int(context.get("recommended_height", info["default_height"]))
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

        next_context = update_task_context(
            context,
            resolved_model_family=context.get("resolved_model_family") or info["family"],
            task_mode="txt2img" if context.get("task_mode") in (None, "", "img2img") else context.get("task_mode"),
            latent_source="empty_latent",
            final_width=width,
            final_height=height,
            batch_size=batch_size,
            latent_channels=latent_channels,
            downscale_ratio=downscale_ratio,
            size_preset=size_preset,
            source="LLS Simple Empty Latent",
        )

        return (
            {"samples": latent, "downscale_ratio_spacial": downscale_ratio, "source": "empty_latent"},
            width,
            height,
            next_context,
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleEmptyLatent": LLSSimpleEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleEmptyLatent": "LLS Simple Empty Latent",
}
