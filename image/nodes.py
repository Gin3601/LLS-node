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
    MODEL_FAMILY_CHOICES,
    canonicalize_family,
    get_family_defaults,
    get_latent_spec,
    info_to_json,
    parse_jsonish_info,
    resolve_model_family,
    resolve_model_name,
    resolve_text_encoder_names,
    resolve_vae_name,
)


try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None

try:
    import comfy.utils as comfy_utils
except Exception as exc:
    comfy_utils = None
    _COMFY_UTILS_ERR = exc
else:
    _COMFY_UTILS_ERR = None


_RESIZE_MODES = ["keep_aspect", "crop_center", "stretch", "none"]
_SIZE_SOURCES = ["input_image", "custom", "model_recommended"]


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def _get_image_dimensions(image) -> tuple[int, int, int]:
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) < 4:
        raise RuntimeError(
            "[LLS] IMAGE input must be a tensor shaped like [batch, height, width, channels]."
        )
    return int(shape[0]), int(shape[2]), int(shape[1])


def _resolve_requested_size(
    size_source: str,
    width: int,
    height: int,
    source_width: int,
    source_height: int,
    model_info: dict,
) -> tuple[int, int]:
    if size_source == "input_image":
        return source_width, source_height
    if size_source == "custom":
        return int(width), int(height)
    if size_source == "model_recommended":
        return int(model_info["default_width"]), int(model_info["default_height"])
    raise RuntimeError(f"[LLS] Unsupported size_source '{size_source}'.")


def _fit_inside_box(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    multiple: int,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("[LLS] IMAGE input has invalid dimensions.")

    scale = min(target_width / source_width, target_height / source_height)
    fitted_width = max(multiple, int(round(source_width * scale)))
    fitted_height = max(multiple, int(round(source_height * scale)))
    fitted_width = min(target_width, _round_to_multiple(fitted_width, multiple))
    fitted_height = min(target_height, _round_to_multiple(fitted_height, multiple))
    return fitted_width, fitted_height


def _resize_image_to(image, target_width: int, target_height: int, resize_mode: str):
    _batch_size, source_width, source_height = _get_image_dimensions(image)
    if source_width == target_width and source_height == target_height:
        return image

    if comfy_utils is None:
        raise RuntimeError(
            "[LLS] comfy.utils is not available, so IMAGE resizing cannot run outside a ComfyUI environment."
        ) from _COMFY_UTILS_ERR

    crop_mode = "center" if resize_mode == "crop_center" else "disabled"
    try:
        channel_first = image.movedim(-1, 1)
        resized = comfy_utils.common_upscale(
            channel_first,
            target_width,
            target_height,
            "bilinear",
            crop_mode,
        )
        return resized.movedim(1, -1)
    except Exception as exc:
        raise RuntimeError(f"[LLS] Failed to resize IMAGE for VAE encode: {exc}") from exc


def _infer_family_from_latent_or_vae(samples, vae=None) -> str:
    tagged_family = getattr(vae, "_lls_family", None)
    if tagged_family and tagged_family not in {"Auto", "auto", ""}:
        return canonicalize_family(tagged_family)

    if isinstance(samples, dict):
        try:
            ratio = int(samples.get("downscale_ratio_spacial", 0))
            if ratio >= 16:
                return "FLUX_DEV"
        except Exception:
            pass
        latent_tensor = samples.get("samples")
        latent_shape = tuple(getattr(latent_tensor, "shape", ()))
        if len(latent_shape) >= 2:
            try:
                if int(latent_shape[1]) >= 16:
                    return "FLUX_DEV"
            except Exception:
                pass

    return "SD1.5"


class LLSSimpleVAEEncode:
    """
    把 IMAGE 编码为 LATENT，供 img2img 直接接入现有 KSampler。
    编码本身复用 ComfyUI 原生 vae.encode，只在外层补尺寸处理与元信息。
    """

    CATEGORY = "LLS/Image"
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "latent_info")
    DESCRIPTION = "Encode an IMAGE into a LATENT tensor for img2img workflows."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "resize_mode": (_RESIZE_MODES, {"default": "keep_aspect"}),
                "size_source": (_SIZE_SOURCES, {"default": "input_image"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    def encode(
        self,
        image,
        vae,
        resize_mode: str,
        size_source: str,
        width: int,
        height: int,
        model_family: str = "Auto",
        model=None,
        clip=None,
    ):
        if image is None:
            raise RuntimeError("[LLS] IMAGE input is required for LLS Simple VAE Encode.")
        if vae is None:
            raise RuntimeError(
                "[LLS] Missing VAE. Connect the Loader VAE output or choose an external VAE in the loader."
            )

        family = resolve_model_family(model_family, model=model, clip=clip)
        defaults = get_family_defaults(family)
        latent_spec = get_latent_spec(defaults)
        alignment = max(8, int(latent_spec["downscale_ratio"]))
        batch_size, source_width, source_height = _get_image_dimensions(image)

        requested_width, requested_height = _resolve_requested_size(
            size_source=size_source,
            width=width,
            height=height,
            source_width=source_width,
            source_height=source_height,
            model_info={
                "default_width": defaults["default_width"],
                "default_height": defaults["default_height"],
            },
        )
        requested_width = _round_to_multiple(requested_width, alignment)
        requested_height = _round_to_multiple(requested_height, alignment)

        if resize_mode == "none":
            final_width = _round_to_multiple(source_width, alignment)
            final_height = _round_to_multiple(source_height, alignment)
        elif resize_mode == "keep_aspect":
            final_width, final_height = _fit_inside_box(
                source_width=source_width,
                source_height=source_height,
                target_width=requested_width,
                target_height=requested_height,
                multiple=alignment,
            )
        elif resize_mode in {"crop_center", "stretch"}:
            final_width = requested_width
            final_height = requested_height
        else:
            raise RuntimeError(f"[LLS] Unsupported resize_mode '{resize_mode}'.")

        processed_image = _resize_image_to(
            image=image,
            target_width=final_width,
            target_height=final_height,
            resize_mode=resize_mode,
        )

        try:
            latent = vae.encode(processed_image)
        except Exception as exc:
            raise RuntimeError(f"[LLS] VAE encode failed: {exc}") from exc

        latent_channels = int(getattr(latent, "shape", [0, latent_spec["latent_channels"]])[1])
        latent_payload = {
            "samples": latent,
            "downscale_ratio_spacial": int(latent_spec["downscale_ratio"]),
            "source": "image_encode",
        }
        latent_info = info_to_json(
            {
                "model_family": family,
                "task_mode": "img2img",
                "latent_source": "image_encode",
                "input_image_width": source_width,
                "input_image_height": source_height,
                "width": final_width,
                "height": final_height,
                "batch_size": batch_size,
                "resize_mode": resize_mode,
                "size_source": size_source,
                "vae_name": resolve_vae_name(vae, fallback="unknown"),
                "latent_channels": latent_channels,
                "downscale_ratio": int(latent_spec["downscale_ratio"]),
            }
        )
        return (latent_payload, final_width, final_height, latent_info)


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
        }

    def decode(self, samples, vae):
        family = _infer_family_from_latent_or_vae(samples, vae)

        if vae is None:
            if family.startswith("FLUX"):
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

        defaults = get_family_defaults(family)
        downscale_ratio = int(samples.get("downscale_ratio_spacial", defaults["downscale_ratio"]))
        height = latent_tensor.shape[2] * downscale_ratio
        width = latent_tensor.shape[3] * downscale_ratio
        decode_info = info_to_json(
            {
                "model_family": family,
                "width": width,
                "height": height,
                "batch_size": latent_tensor.shape[0],
                "vae_name": resolve_vae_name(vae, fallback="unknown"),
                "decode_stage": "vae_decode",
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
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
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
        model=None,
        clip=None,
        vae=None,
        prompt_info: str | None = None,
        latent_info: str | None = None,
        sample_info: str | None = None,
        decode_info: str | None = None,
        upscale_info: str | None = None,
    ) -> dict:
        prompt = parse_jsonish_info(prompt_info)
        latent = parse_jsonish_info(latent_info)
        sample = parse_jsonish_info(sample_info)
        decode = parse_jsonish_info(decode_info)
        upscale = parse_jsonish_info(upscale_info)

        family = prompt.get("model_family") or latent.get("model_family") or decode.get("model_family")
        if not family:
            family = resolve_model_family("Auto", model=model, clip=clip)

        text_encoder_name = resolve_text_encoder_names(clip).get("text_encoder_name", "")
        checkpoint_name = resolve_model_name(
            model=model,
            clip=clip,
            fallback=prompt.get("checkpoint_name") or latent.get("checkpoint_name") or decode.get("checkpoint_name") or "",
        )
        vae_name = resolve_vae_name(
            vae,
            fallback=decode.get("vae_name") or latent.get("vae_name") or "",
        )

        metadata = {
            "positive_prompt": prompt.get("positive_prompt", ""),
            "negative_prompt": prompt.get("negative_prompt", ""),
            "seed": sample.get("seed"),
            "model_family": family,
            "checkpoint_name": checkpoint_name,
            "vae_name": vae_name,
            "text_encoder_name": text_encoder_name,
            "steps": sample.get("steps"),
            "cfg": sample.get("cfg"),
            "guidance": sample.get("guidance"),
            "sampler_name": sample.get("sampler_name"),
            "scheduler": sample.get("scheduler"),
            "denoise": sample.get("denoise"),
            "width": decode.get("width") or latent.get("width") or image.shape[2],
            "height": decode.get("height") or latent.get("height") or image.shape[1],
            "batch_size": decode.get("batch_size") or latent.get("batch_size") or image.shape[0],
            "upscale_mode": upscale.get("mode") or upscale.get("upscale_mode"),
            "scale": upscale.get("scale") or upscale.get("upscale_scale"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        return metadata

    def save(
        self,
        image,
        filename_prefix: str = "LLS",
        save_metadata: bool = True,
        model=None,
        clip=None,
        vae=None,
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
                model=model,
                clip=clip,
                vae=vae,
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
    "LLSSimpleVAEEncode": LLSSimpleVAEEncode,
    "LLSSimpleVAEDecode": LLSSimpleVAEDecode,
    "LLSSaveImage": LLSSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleVAEEncode": "LLS Simple VAE Encode",
    "LLSSimpleVAEDecode": "LLS Simple VAE Decode",
    "LLSSaveImage": "LLS Save Image",
}
