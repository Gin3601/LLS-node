"""
LLS / Image
===========
功能域：图像处理与后处理（对应功能分类总览第 5 节）

CATEGORY = "LLS/Image"
"""
from __future__ import annotations

from ..utils.model_info import get_latent_spec, parse_model_info


# ---------- 节点类 ----------

class LLSSimpleVAEDecode:
    """
    简化版 VAE Decode 节点。
    内部复用 ComfyUI 原生 VAEDecode 能力，将 Latent 解码为像素图像。
    第一版只做普通解码，不实现 Tiled VAE。
    """

    CATEGORY = "LLS/Image"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "decode_info")
    DESCRIPTION = (
        "Decode a LATENT tensor into a pixel IMAGE using the VAE. "
        "First version uses standard (non-tiled) decoding."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "model_info": ("STRING", {"default": ""}),
            },
        }

    def decode(self, samples, vae, model_info: str | None = None):
        if vae is None:
            raise RuntimeError(
                "[LLS] VAE is None. Make sure a checkpoint is loaded and VAE is connected."
            )
        if samples is None:
            raise RuntimeError(
                "[LLS] LATENT samples is None. Connect the output of KSampler to this node."
            )

        latent_tensor = samples.get("samples")
        if latent_tensor is None:
            raise RuntimeError(
                "[LLS] LATENT dict does not contain 'samples' key. "
                "Make sure the upstream node outputs a valid LATENT."
            )
        if getattr(latent_tensor, "is_nested", False):
            latent_tensor = latent_tensor.unbind()[0]

        # 复用 ComfyUI VAE 原生 decode 方法
        try:
            image = vae.decode(latent_tensor)
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] VAE decode failed: {exc}"
            ) from exc

        if len(getattr(image, "shape", ())) == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])

        info = parse_model_info(model_info)
        latent_spec = get_latent_spec(info)
        downscale_ratio = int(samples.get("downscale_ratio_spacial", latent_spec["downscale_ratio"]))

        h, w = latent_tensor.shape[2] * downscale_ratio, latent_tensor.shape[3] * downscale_ratio
        batch = latent_tensor.shape[0]
        decode_info = (
            f"family={info['family']} | decoded={w}x{h} | batch={batch} "
            f"| downscale={downscale_ratio} | latent_shape={list(latent_tensor.shape)}"
        )

        return (image, decode_info)


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleVAEDecode": LLSSimpleVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleVAEDecode": "LLS Simple VAE Decode",
}
