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

from ..utils.model_info import get_latent_spec, parse_model_info


_SIZE_PRESETS = [
    "Custom",
    "512x512",
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
    生成空白 Latent，作为 txt2img 推理的起点。

    默认保持 SD 风格 4 通道 / 8x 下采样；
    若提供 model_info，则可自动切换到 FLUX 的 128 通道 / 16x 下采样。
    """

    CATEGORY = "LLS/Latent"
    FUNCTION = "create_empty_latent"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "latent_info")
    DESCRIPTION = (
        "Create an empty latent image as the starting point for txt2img. "
        "When model_info is connected, latent channels and downscale ratio adapt to the model family."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_preset": (_SIZE_PRESETS, {"default": "512x512"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "model_info": ("STRING", {"default": ""}),
            },
        }

    def create_empty_latent(
        self,
        size_preset: str,
        width: int,
        height: int,
        batch_size: int,
        model_info: str | None = None,
    ):
        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. Make sure this node runs inside a ComfyUI environment."
            ) from _TORCH_ERR

        info = parse_model_info(model_info)
        latent_spec = get_latent_spec(info)
        latent_channels = latent_spec["latent_channels"]
        downscale_ratio = latent_spec["downscale_ratio"]
        notes = []

        if size_preset != "Custom":
            try:
                width, height = [int(part) for part in size_preset.split("x", 1)]
            except Exception:
                notes.append(f"preset parse failed, using custom {width}x{height}")

        corrected_w = _round_to_multiple(width, downscale_ratio)
        corrected_h = _round_to_multiple(height, downscale_ratio)
        if corrected_w != width or corrected_h != height:
            notes.append(f"size corrected {width}x{height} -> {corrected_w}x{corrected_h}")
        width, height = corrected_w, corrected_h

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

        note_str = " | ".join(notes) if notes else "ok"
        latent_info = (
            f"family={info['family']} | size={width}x{height} | batch={batch_size} | "
            f"channels={latent_channels} | downscale={downscale_ratio} | preset={size_preset} | {note_str}"
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
