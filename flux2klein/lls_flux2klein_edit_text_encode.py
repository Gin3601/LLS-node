from __future__ import annotations

import math

from ..repair.repair_utils import (
    clamp_mask_values,
    get_image_size,
    invert_mask_values,
    normalize_mask_values,
    resize_image_to,
    resize_mask_to,
)

try:
    import node_helpers
except Exception:  # pragma: no cover - optional ComfyUI runtime dependency
    node_helpers = None

try:
    import torch
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised in ComfyUI runtime/tests when torch exists
    _TORCH_ERR = None

try:
    import comfy.utils as comfy_utils
except Exception as exc:  # pragma: no cover - optional runtime dependency
    comfy_utils = None
    _COMFY_UTILS_ERR = exc
else:  # pragma: no cover - exercised in ComfyUI runtime/tests when comfy exists
    _COMFY_UTILS_ERR = None

try:
    import comfy_extras.nodes_flux as native_flux_nodes
except Exception:  # pragma: no cover - optional ComfyUI runtime dependency
    native_flux_nodes = None

try:
    import comfy_extras.nodes_edit_model as native_edit_model_nodes
except Exception:  # pragma: no cover - optional ComfyUI runtime dependency
    native_edit_model_nodes = None


RESIZE_MODE_CHOICES = ["longest_edge", "keep_original"]
MASK_MODE_CHOICES = ["none", "use_mask", "invert_mask"]
CUSTOM_OUTPUT_TYPE = "LLS_FLUX2KLEIN_OUTPUT"
_VISION_TOTAL_PIXELS = int(384 * 384)
_VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


class _ConstantMask:
    """Fallback mask object for non-Comfy test/runtime contexts without torch."""

    def __init__(self, shape, value: float = 1.0, label: str = "mask"):
        self.shape = tuple(shape)
        self.value = float(value)
        self.label = label
        width = int(self.shape[2])
        height = int(self.shape[1])
        if self.value <= 0.0:
            self.mask_bbox = None
            self.mask_area_ratio = 0.0
        else:
            self.mask_bbox = (0, 0, width, height)
            self.mask_area_ratio = max(0.0, min(1.0, self.value))

    def resized(self, width, height):
        return _ConstantMask((self.shape[0], int(height), int(width)), value=self.value, label=f"{self.label}:resized")

    def normalized(self):
        return _ConstantMask(self.shape, value=max(0.0, min(1.0, self.value)), label=f"{self.label}:normalized")

    def inverted(self, image_size):
        width, height = image_size
        return _ConstantMask(
            (self.shape[0], int(height), int(width)),
            value=max(0.0, min(1.0, 1.0 - self.value)),
            label=f"{self.label}:inverted",
        )


class _ShapeOnlyLatent:
    """Fallback latent object for tests when torch tensors are unavailable."""

    def __init__(self, shape):
        self.shape = tuple(shape)


def _extract_latent_samples(latent):
    if isinstance(latent, dict) and "samples" in latent:
        return latent["samples"]
    return latent


def _get_image_batch_size(image) -> int:
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError("[LLS] image1 must have shape [batch, height, width, channels].")
    batch = int(shape[0])
    if batch <= 0:
        raise RuntimeError("[LLS] image1 batch size must be positive.")
    return batch


def _extract_node_output_value(value):
    if hasattr(value, "result"):
        value = value.result
    elif hasattr(value, "args"):
        value = value.args

    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _scale_size_by_longest_edge(width: int, height: int, target_longest_edge: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    longest = max(width, height)
    if width <= 0 or height <= 0 or longest <= 0:
        raise RuntimeError("[LLS] IMAGE width and height must be positive.")

    scale = float(target_longest_edge) / float(longest)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return target_width, target_height


def _prepare_image(image, resize_mode: str, ref_longest_edge: int):
    if image is None:
        return None
    if resize_mode == "keep_original":
        return image
    if resize_mode != "longest_edge":
        raise RuntimeError(f"[LLS] Unsupported resize_mode '{resize_mode}'.")
    width, height = get_image_size(image)
    target_width, target_height = _scale_size_by_longest_edge(width, height, int(ref_longest_edge))
    return resize_image_to(image, target_width, target_height)


def _make_full_mask_like_image(image):
    batch = _get_image_batch_size(image)
    width, height = get_image_size(image)

    if torch is not None:
        device = getattr(image, "device", "cpu")
        try:
            return torch.ones((batch, height, width), dtype=torch.float32, device=device)
        except Exception:
            return torch.ones((batch, height, width), dtype=torch.float32)

    return _ConstantMask((batch, height, width), value=1.0, label="full-mask")


def _broadcast_mask_batch_if_needed(mask, image):
    if torch is None or not isinstance(mask, torch.Tensor) or not isinstance(image, torch.Tensor):
        return mask

    image_batch = int(image.shape[0])
    mask_batch = int(mask.shape[0])
    if image_batch == mask_batch:
        return mask
    if mask_batch == 1 and image_batch > 1:
        return mask.repeat(image_batch, 1, 1)
    raise RuntimeError(
        f"[LLS] MASK batch size {mask_batch} does not match image1 batch size {image_batch}."
    )


def _prepare_mask(mask, image1, mask_mode: str):
    # mask only applies to the main edit image (image1), never to image2/image3.
    if mask_mode == "none" or mask is None:
        return _make_full_mask_like_image(image1)

    width, height = get_image_size(image1)
    processed = resize_mask_to(mask, width, height)
    processed = normalize_mask_values(processed)
    if mask_mode == "invert_mask":
        processed = invert_mask_values(processed, (width, height))
    elif mask_mode != "use_mask":
        raise RuntimeError(f"[LLS] Unsupported mask_mode '{mask_mode}'.")
    processed = clamp_mask_values(processed)
    return _broadcast_mask_batch_if_needed(processed, image1)


def _resize_image_with_mode(image, width: int, height: int, *, upscale_method: str, crop: str):
    if image is None:
        return None

    current_width, current_height = get_image_size(image)
    if (current_width, current_height) == (int(width), int(height)):
        return image

    if torch is not None and isinstance(image, torch.Tensor):
        if comfy_utils is None:
            raise RuntimeError(
                "[LLS] comfy.utils is required for Flux2Klein vision/image resizing in tensor mode."
            ) from _COMFY_UTILS_ERR
        channel_first = image.movedim(-1, 1)
        resized = comfy_utils.common_upscale(channel_first, int(width), int(height), upscale_method, crop)
        return resized.movedim(1, -1)

    return resize_image_to(image, int(width), int(height))


def _scale_image_to_total_pixels(image, total_pixels: int, *, upscale_method: str = "area"):
    width, height = get_image_size(image)
    current_total = max(1, int(width) * int(height))
    scale_by = math.sqrt(float(total_pixels) / float(current_total))
    target_width = max(1, int(round(width * scale_by)))
    target_height = max(1, int(round(height * scale_by)))
    return _resize_image_with_mode(image, target_width, target_height, upscale_method=upscale_method, crop="center")


def _count_present_reference_images(image1, image2=None, image3=None) -> int:
    return int(image1 is not None) + int(image2 is not None) + int(image3 is not None)


def _build_flux2klein_prompt(prompt: str, image_count: int) -> str:
    prefix = " ".join(f"image{i}: {_VISION_TOKEN}" for i in range(1, max(0, int(image_count)) + 1))
    prompt_text = str(prompt or "")
    if prefix and prompt_text:
        return f"{prefix} {prompt_text}"
    if prefix:
        return prefix
    return prompt_text


def _append_reference_latents(conditioning, ref_latents):
    if not ref_latents:
        return conditioning

    if node_helpers is not None:
        try:
            return node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        except Exception:
            pass

    updated = []
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            cond = entry[0]
            meta = dict(entry[1])
        else:
            cond = entry
            meta = {}
        current_refs = list(meta.get("reference_latents", []))
        current_refs.extend(ref_latents)
        meta["reference_latents"] = current_refs
        updated.append([cond, meta])
    return updated


def _encode_tokens_to_conditioning(clip, tokens):
    scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
    if callable(scheduled):
        try:
            return scheduled(tokens)
        except Exception as exc:
            raise RuntimeError(
                "[LLS] Flux2Klein vision token encoding failed while calling encode_from_tokens_scheduled(). "
                f"Flux2Klein 视觉 token 编码失败: {exc}"
            ) from exc

    plain_encode = getattr(clip, "encode_from_tokens", None)
    if not callable(plain_encode):
        raise RuntimeError(
            "[LLS] No usable vision-aware text encoding backend for LLS Flux2Klein Edit Text Encode. "
            "Current clip object does not expose encode_from_tokens_scheduled()/encode_from_tokens()."
        )

    try:
        encoded = plain_encode(tokens, return_pooled=True, return_dict=True)
    except TypeError:
        encoded = plain_encode(tokens, return_pooled=True)
    except Exception as exc:
        raise RuntimeError(
            "[LLS] Flux2Klein vision token encoding failed while calling encode_from_tokens(). "
            f"Flux2Klein 视觉 token 编码失败: {exc}"
        ) from exc

    if isinstance(encoded, dict) and "cond" in encoded:
        meta = dict(encoded)
        cond = meta.pop("cond")
        return [[cond, meta]]
    if isinstance(encoded, tuple) and len(encoded) >= 2:
        cond, pooled = encoded[:2]
        return [[cond, {"pooled_output": pooled}]]
    return [[encoded, {}]]


def _encode_multivision_conditioning(clip, prompt: str, images):
    tokenize = getattr(clip, "tokenize", None)
    if not callable(tokenize):
        raise RuntimeError(
            "[LLS] No usable vision-aware text encoding backend for LLS Flux2Klein Edit Text Encode. "
            "Current clip object does not expose tokenize()."
        )

    full_prompt = _build_flux2klein_prompt(prompt, len(images))
    try:
        tokens = tokenize(full_prompt, images=images)
    except TypeError as exc:
        raise RuntimeError(
            "[LLS] The connected clip does not support vision-aware tokenize(images=...). "
            "请连接 Flux2 / Klein 兼容的多模态 clip。"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "[LLS] Flux2Klein vision token preparation failed while calling tokenize(images=...). "
            f"Flux2Klein 视觉 token 准备失败: {exc}"
        ) from exc

    return _encode_tokens_to_conditioning(clip, tokens), full_prompt


def _encode_reference_latent(vae, image, target_width: int, target_height: int, source_name: str):
    prepared = _resize_image_with_mode(
        image,
        target_width,
        target_height,
        upscale_method="lanczos",
        crop="center",
    )
    try:
        samples = vae.encode(prepared)
    except Exception as exc:
        raise RuntimeError(
            f"[LLS] VAE encode failed for {source_name} / {source_name} 的 VAE 编码失败: {exc}"
        ) from exc
    return {"samples": samples}, prepared


def _encode_image_latent(vae, image, source_name: str):
    try:
        samples = vae.encode(image)
    except Exception as exc:
        raise RuntimeError(
            f"[LLS] VAE encode failed for {source_name} / {source_name} 的 VAE 编码失败: {exc}"
        ) from exc
    return {"samples": samples}


def _get_latent_channels(latent_samples) -> int:
    shape = tuple(getattr(latent_samples, "shape", ()))
    if len(shape) < 2:
        raise RuntimeError("[LLS] Invalid latent tensor shape. 无法识别 latent 通道数。")
    return int(shape[1])


def _build_empty_like_latent(samples):
    shape = tuple(getattr(samples, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError("[LLS] Invalid latent tensor shape for Empty Flux2 latent fallback.")

    if torch is not None and isinstance(samples, torch.Tensor):
        return {"samples": torch.zeros(shape, dtype=samples.dtype, device=samples.device)}

    try:
        return {"samples": samples.__class__(shape)}
    except Exception:
        return {"samples": _ShapeOnlyLatent(shape)}


def _build_empty_flux2_latent(main_image, main_latent):
    width, height = get_image_size(main_image)
    batch_size = _get_image_batch_size(main_image)

    execute = getattr(getattr(native_flux_nodes, "EmptyFlux2LatentImage", None), "execute", None)
    if callable(execute):
        try:
            return _extract_node_output_value(execute(width=width, height=height, batch_size=batch_size))
        except TypeError:
            return _extract_node_output_value(execute(width, height, batch_size))
        except Exception:
            pass

    return _build_empty_like_latent(_extract_latent_samples(main_latent))


def _attach_reference_latents(conditioning, ref_latents):
    if not ref_latents:
        return conditioning

    execute = getattr(getattr(native_edit_model_nodes, "ReferenceLatent", None), "execute", None)
    if callable(execute):
        updated = conditioning
        for ref_latent in ref_latents:
            try:
                updated = _extract_node_output_value(execute(updated, latent=ref_latent))
            except TypeError:
                updated = _extract_node_output_value(execute(updated, ref_latent))
        return updated

    return _append_reference_latents(conditioning, [_extract_latent_samples(latent) for latent in ref_latents])


def _build_noise_mask(mask, latent_samples):
    target_width = int(latent_samples.shape[-1])
    target_height = int(latent_samples.shape[-2])
    noise_mask = resize_mask_to(mask, target_width, target_height)
    noise_mask = normalize_mask_values(noise_mask)
    noise_mask = clamp_mask_values(noise_mask)
    samples_batch = int(latent_samples.shape[0])

    if torch is not None and isinstance(noise_mask, torch.Tensor):
        mask_batch = int(noise_mask.shape[0])
        if mask_batch == samples_batch:
            return noise_mask
        if mask_batch == 1 and samples_batch > 1:
            return noise_mask.repeat(samples_batch, 1, 1)
        raise RuntimeError(
            f"[LLS] noise_mask batch size {mask_batch} does not match latent batch size {samples_batch}."
        )

    return noise_mask


class LLSFlux2KleinEditTextEncode:
    CATEGORY = "LLS/Flux2Klein"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT", CUSTOM_OUTPUT_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    SEARCH_ALIASES = ["LLS", "Flux2Klein", "Edit Text Encode", "Flux Edit"]
    DESCRIPTION = (
        "PainterFluxImageEdit-style Flux2Klein edit wrapper for LLS. Uses image1 as the main edit image, "
        "image2/image3 as reference images, never concatenates them, and prepares conditioning/latent/mask outputs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "resize_mode": (RESIZE_MODE_CHOICES, {"default": "longest_edge"}),
                "mask_mode": (MASK_MODE_CHOICES, {"default": "none"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def encode(
        self,
        clip,
        vae,
        image1,
        prompt,
        ref_longest_edge,
        resize_mode,
        mask_mode,
        image2=None,
        image3=None,
        mask=None,
    ):
        if clip is None:
            raise RuntimeError("[LLS] Missing CLIP. 请连接可用的 clip 文本编码器。")
        if vae is None:
            raise RuntimeError("[LLS] Missing VAE. 请连接可用的 vae。")
        if image1 is None:
            raise RuntimeError("[LLS] Missing image1. image1 是主编辑图 / main edit image。")

        main_image = _prepare_image(image1, str(resize_mode), int(ref_longest_edge))
        reference_image_2 = _prepare_image(image2, str(resize_mode), int(ref_longest_edge))
        reference_image_3 = _prepare_image(image3, str(resize_mode), int(ref_longest_edge))
        output_mask = _prepare_mask(mask, main_image, str(mask_mode))

        images_for_edit = [img for img in (main_image, reference_image_2, reference_image_3) if img is not None]
        vision_images = [_scale_image_to_total_pixels(img, _VISION_TOTAL_PIXELS, upscale_method="area") for img in images_for_edit]
        conditioning, full_prompt = _encode_multivision_conditioning(clip, str(prompt), vision_images)

        target_width, target_height = get_image_size(main_image)
        main_latent = _encode_image_latent(vae, main_image, source_name="image1")
        main_latent_channels = _get_latent_channels(_extract_latent_samples(main_latent))
        use_official_flux2_latent = main_latent_channels >= 128

        ref_latents = [main_latent]
        if reference_image_2 is not None:
            if use_official_flux2_latent:
                ref_latents.append(_encode_image_latent(vae, reference_image_2, source_name="image2"))
            else:
                ref_latent, _prepared = _encode_reference_latent(
                    vae,
                    reference_image_2,
                    target_width,
                    target_height,
                    source_name="image2",
                )
                ref_latents.append(ref_latent)
        if reference_image_3 is not None:
            if use_official_flux2_latent:
                ref_latents.append(_encode_image_latent(vae, reference_image_3, source_name="image3"))
            else:
                ref_latent, _prepared = _encode_reference_latent(
                    vae,
                    reference_image_3,
                    target_width,
                    target_height,
                    source_name="image3",
                )
                ref_latents.append(ref_latent)

        conditioning = _attach_reference_latents(conditioning, ref_latents)

        if use_official_flux2_latent:
            latent = _build_empty_flux2_latent(main_image, main_latent)
            latent_mode = "empty_flux2_latent"
            conditioning_backend = "flux2klein_multivision_clip+reference_latent"
        else:
            latent = {
                "samples": _extract_latent_samples(main_latent),
            }
            latent_mode = "image1_reference_latent"
            conditioning_backend = "flux2klein_multivision_clip"

        if mask is not None and str(mask_mode) != "none":
            latent["noise_mask"] = _build_noise_mask(output_mask, latent["samples"])

        reference_names = []
        if reference_image_2 is not None:
            reference_names.append("image2")
        if reference_image_3 is not None:
            reference_names.append("image3")

        custom_output = {
            "node_name": "LLS Flux2Klein Edit Text Encode",
            "prompt": str(prompt),
            "full_prompt": full_prompt,
            "ref_longest_edge": int(ref_longest_edge),
            "resize_mode": str(resize_mode),
            "mask_mode": str(mask_mode),
            "has_mask": mask is not None,
            "main_image": "image1",
            "reference_images": reference_names,
            "num_reference_images": len(reference_names),
            "conditioning_backend": conditioning_backend,
            "native_error": None,
            "native_wrapper": "official Flux2 ReferenceLatent" if use_official_flux2_latent else "PainterFluxImageEdit-compatible",
            "latent_mode": latent_mode,
            "latent_has_noise_mask": "noise_mask" in latent,
            "vision_image_count": len(vision_images),
            "reference_latent_count": len(ref_latents),
            "main_latent_channels": main_latent_channels,
            "target_size": {"width": int(target_width), "height": int(target_height)},
            "note": "image1 is main edit image; image2 and image3 are reference images; images are not concatenated",
        }

        return (
            conditioning,
            latent,
            custom_output,
            main_image,
            output_mask,
        )


NODE_CLASS_MAPPINGS = {
    "LLSFlux2KleinEditTextEncode": LLSFlux2KleinEditTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSFlux2KleinEditTextEncode": "LLS Flux2Klein Edit Text Encode",
}
