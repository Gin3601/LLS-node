"""
LLS / Latent
============
功能域：Latent 空间操作（对应功能分类总览第 4 节）

CATEGORY = "LLS/Latent"
"""
from __future__ import annotations

# ---------- 防御性导入 ----------

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


# ---------- 工具函数 ----------

_SIZE_PRESETS = [
    "Custom",
    # SD1.5 推荐尺寸
    "512x512",
    "512x768",
    "768x512",
    # SDXL 推荐尺寸
    "1024x1024",
    "832x1216",
    "1216x832",
    "896x1152",
    "1152x896",
    "1024x576",
    "576x1024",
]


def _round_to_8(value: int) -> int:
    """将整数修正为最近的 8 的倍数。"""
    return max(8, round(value / 8) * 8)


# ---------- 节点类 ----------

class LLSSimpleEmptyLatent:
    """
    生成空白 Latent，作为 txt2img 推理的起点。
    支持常用分辨率预设，自动修正宿高为 8 的倍数。
    """

    CATEGORY = "LLS/Latent"
    FUNCTION = "create_empty_latent"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "latent_info")
    DESCRIPTION = (
        "Create an empty latent image as the starting point for txt2img. "
        "Select a preset or enter custom width/height. "
        "Width and height are automatically rounded to multiples of 8. "
        "SD1.5 recommended: 512x512; SDXL recommended: 1024x1024 or 896x1152."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_preset": (_SIZE_PRESETS, {"default": "512x512"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    def create_empty_latent(self, size_preset: str, width: int, height: int, batch_size: int):
        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. "
                "Make sure this node runs inside a ComfyUI environment."
            ) from _TORCH_ERR

        notes = []

        # 如果选择了预设，覆盖 width / height
        if size_preset != "Custom":
            try:
                w_str, h_str = size_preset.split("x")
                width = int(w_str)
                height = int(h_str)
            except Exception:
                notes.append(f"preset parse failed, using custom {width}x{height}")

        # 修正为 8 的倍数
        corrected_w = _round_to_8(width)
        corrected_h = _round_to_8(height)
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

        # 生成空白 Latent（与 ComfyUI EmptyLatentImage 格式一致）
        # Latent 尺寸是图像尺寸的 1/8
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=device,
            dtype=dtype,
        )

        note_str = " | ".join(notes) if notes else "ok"
        latent_info = (
            f"size={width}x{height} | batch={batch_size} "
            f"| preset={size_preset} | {note_str}"
        )

        return ({"samples": latent, "downscale_ratio_spacial": 8}, width, height, latent_info)


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleEmptyLatent": LLSSimpleEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleEmptyLatent": "LLS Simple Empty Latent",
}
