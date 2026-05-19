"""
LLS / Image
===========
功能域：图像处理与后处理（对应功能分类总览第 5 节）

CATEGORY = "LLS/Image"
"""
from __future__ import annotations

from datetime import datetime
import importlib

from ..utils.model_info import (
    LLS_MODEL_INFO_TYPE,
    get_latent_spec,
    info_to_json,
    parse_jsonish_info,
    parse_model_info,
)


try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None


class LLSSimpleVAEDecode:
    """
    简化版 VAE Decode 节点。
    内部复用 ComfyUI 原生 VAE.decode。
    """

    CATEGORY = "LLS/Image"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "decode_info")
    DESCRIPTION = "Decode a LATENT tensor into a pixel IMAGE using the VAE."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "model_info": (LLS_MODEL_INFO_TYPE,),
            },
        }

    def decode(self, samples, vae, model_info=None):
        info = parse_model_info(model_info)
        family = info["family"]

        if vae is None:
            if info.get("is_flux"):
                raise RuntimeError(
                    "[LLS] Missing FLUX AE/VAE. Place ae.safetensors in ComfyUI/models/vae/ "
                    "or set Loader.vae_source to 'external'."
                )
            raise RuntimeError(
                "[LLS] Missing VAE. Connect the Loader VAE output or choose an external VAE in the loader."
            )
        if samples is None:
            raise RuntimeError(
                "[LLS] LATENT samples is None. Connect the output of KSampler to this node."
            )

        latent_tensor = samples.get("samples")
        if latent_tensor is None:
            raise RuntimeError(
                "[LLS] LATENT dict does not contain 'samples'. Make sure the upstream node outputs a valid LATENT."
            )
        if getattr(latent_tensor, "is_nested", False):
            latent_tensor = latent_tensor.unbind()[0]

        try:
            image = vae.decode(latent_tensor)
        except Exception as exc:
            raise RuntimeError(f"[LLS] VAE decode failed: {exc}") from exc

        if len(getattr(image, "shape", ())) == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])

        latent_spec = get_latent_spec(info)
        downscale_ratio = int(samples.get("downscale_ratio_spacial", latent_spec["downscale_ratio"]))
        height = latent_tensor.shape[2] * downscale_ratio
        width = latent_tensor.shape[3] * downscale_ratio
        decode_info = info_to_json(
            {
                "vae_name": info.get("vae_name", "unknown"),
                "family": family,
                "decode_mode": "standard",
                "width": width,
                "height": height,
                "batch_size": latent_tensor.shape[0],
            }
        )
        return (image, decode_info)


class LLSSaveImage:
    """
    复用 ComfyUI 原生 SaveImage，并把 LLS 生成链路信息合并进 PNG metadata。
    """

    CATEGORY = "LLS/Image"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    DESCRIPTION = "Save images with LLS generation metadata."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "LLS"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_info": (LLS_MODEL_INFO_TYPE,),
                "prompt_info": ("STRING", {"forceInput": True}),
                "latent_info": ("STRING", {"forceInput": True}),
                "sample_info": ("STRING", {"forceInput": True}),
                "decode_info": ("STRING", {"forceInput": True}),
                "upscale_info": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def _build_metadata(
        self,
        image,
        model_info=None,
        prompt_info: str | None = None,
        latent_info: str | None = None,
        sample_info: str | None = None,
        decode_info: str | None = None,
        upscale_info: str | None = None,
    ) -> dict:
        model = parse_model_info(model_info)
        prompt = parse_jsonish_info(prompt_info)
        latent = parse_jsonish_info(latent_info)
        sample = parse_jsonish_info(sample_info)
        decode = parse_jsonish_info(decode_info)
        upscale = parse_jsonish_info(upscale_info)

        text_encoder_name = model.get("text_encoder_name")
        if model.get("text_encoder_name_2"):
            joined = [name for name in (model.get("text_encoder_name_1"), model.get("text_encoder_name_2")) if name]
            text_encoder_name = ", ".join(joined)

        metadata = {
            "positive_prompt": prompt.get("positive_prompt", ""),
            "negative_prompt": prompt.get("negative_prompt", ""),
            "seed": sample.get("seed"),
            "model_family": model.get("family"),
            "checkpoint_name": model.get("checkpoint_name"),
            "vae_name": decode.get("vae_name") or model.get("vae_name"),
            "text_encoder_name": text_encoder_name,
            "steps": sample.get("steps"),
            "cfg": sample.get("cfg"),
            "guidance": sample.get("guidance"),
            "sampler_name": sample.get("sampler_name"),
            "scheduler": sample.get("scheduler"),
            "denoise": sample.get("denoise"),
            "width": decode.get("width") or latent.get("width") or image.shape[2],
            "height": decode.get("height") or latent.get("height") or image.shape[1],
            "batch_size": latent.get("batch_size") or decode.get("batch_size") or image.shape[0],
            "upscale_mode": upscale.get("mode"),
            "scale": upscale.get("scale"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        return metadata

    def save(
        self,
        image,
        filename_prefix: str = "LLS",
        save_metadata: bool = True,
        model_info=None,
        prompt_info: str | None = None,
        latent_info: str | None = None,
        sample_info: str | None = None,
        decode_info: str | None = None,
        upscale_info: str | None = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if comfy_core_nodes is None or not hasattr(comfy_core_nodes, "SaveImage"):
            raise RuntimeError("[LLS] ComfyUI core SaveImage node is not available.")

        saver = comfy_core_nodes.SaveImage()
        merged_extra_pnginfo = dict(extra_pnginfo or {})
        if save_metadata:
            merged_extra_pnginfo["lls_metadata"] = self._build_metadata(
                image=image,
                model_info=model_info,
                prompt_info=prompt_info,
                latent_info=latent_info,
                sample_info=sample_info,
                decode_info=decode_info,
                upscale_info=upscale_info,
            )
        return saver.save_images(
            image,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=merged_extra_pnginfo,
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleVAEDecode": LLSSimpleVAEDecode,
    "LLSSaveImage": LLSSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleVAEDecode": "LLS Simple VAE Decode",
    "LLSSaveImage": "LLS Save Image",
}
