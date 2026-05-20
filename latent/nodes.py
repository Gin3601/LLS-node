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
    MODEL_FAMILY_CHOICES,
    SIZE_PRESET_AUTO,
    get_family_defaults,
    get_latent_spec,
    info_to_json,
    parse_jsonish_info,
    resolve_model_family,
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

_RESIZE_MODES = ["keep_aspect", "crop_center", "stretch", "none"]


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _resolve_size_preset(size_preset: str, width: int, height: int, defaults: dict) -> tuple[int, int]:
    if size_preset == SIZE_PRESET_AUTO:
        return int(defaults["default_width"]), int(defaults["default_height"])
    if size_preset == "Custom":
        return int(width), int(height)
    try:
        resolved_width, resolved_height = [int(part) for part in size_preset.split("x", 1)]
    except Exception as exc:
        raise RuntimeError(f"[LLS] Invalid size_preset value '{size_preset}': {exc}") from exc
    return resolved_width, resolved_height


class LLSSimpleEmptyLatent:
    """
    统一的 Latent 入口节点。

    - 未连接 `image` 时，生成空白 Latent，作为 txt2img 起点。
    - 连接 `image` 时，复用 VAE Encode 流程生成 img2img Latent。
    - `Family Default` 时按照模型家族默认尺寸自动取值。
    - `Custom` 时才使用手填 width / height。
    """

    CATEGORY = "LLS/Latent"
    FUNCTION = "create_empty_latent"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "latent_info")
    DESCRIPTION = (
        "Create an empty latent for txt2img, or encode an input image into latent space for img2img. "
        "Uses inferred family defaults when size_preset is 'Family Default'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_preset": (_SIZE_PRESETS, {"default": SIZE_PRESET_AUTO}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "resize_mode": (_RESIZE_MODES, {"default": "keep_aspect"}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    def create_empty_latent(
        self,
        size_preset: str,
        width: int,
        height: int,
        batch_size: int,
        model_family: str = "Auto",
        resize_mode: str = "keep_aspect",
        model=None,
        image=None,
        vae=None,
    ):
        family = resolve_model_family(model_family, model=model)
        defaults = get_family_defaults(family)
        requested_width, requested_height = _resolve_size_preset(size_preset, width, height, defaults)

        if image is not None:
            if vae is None:
                raise RuntimeError(
                    "[LLS] Missing VAE. Connect the Loader VAE output or choose an external VAE in the loader."
                )

            from ..image.nodes import LLSSimpleVAEEncode

            size_source = "model_recommended" if size_preset == SIZE_PRESET_AUTO else "custom"
            latent_payload, final_width, final_height, latent_info = LLSSimpleVAEEncode().encode(
                image=image,
                vae=vae,
                resize_mode=resize_mode,
                size_source=size_source,
                width=requested_width,
                height=requested_height,
                model_family=family,
                model=model,
            )
            latent_meta = parse_jsonish_info(latent_info)
            latent_meta["size_preset"] = size_preset
            return (latent_payload, final_width, final_height, info_to_json(latent_meta))

        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. Make sure this node runs inside a ComfyUI environment."
            ) from _TORCH_ERR

        latent_spec = get_latent_spec(defaults)
        latent_channels = latent_spec["latent_channels"]
        downscale_ratio = latent_spec["downscale_ratio"]
        width = requested_width
        height = requested_height

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
                "model_family": family,
                "task_mode": "txt2img",
                "latent_source": "empty_latent",
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "latent_channels": latent_channels,
                "downscale_ratio": downscale_ratio,
                "size_preset": size_preset,
            }
        )
        return (
            {"samples": latent, "downscale_ratio_spacial": downscale_ratio, "source": "empty_latent"},
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
