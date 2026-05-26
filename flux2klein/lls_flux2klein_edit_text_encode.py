from __future__ import annotations

import importlib

from ..conditioning.nodes import LLSUniversalPromptEncode
from ..repair.repair_utils import (
    clamp_mask_values,
    get_image_size,
    invert_mask_values,
    normalize_mask_values,
    resize_image_to,
    resize_mask_to,
)

try:
    import torch
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised in ComfyUI runtime/tests when torch exists
    _TORCH_ERR = None

try:
    nodes_qwen = importlib.import_module("comfy_extras.nodes_qwen")
except Exception as exc:  # pragma: no cover - runtime-only import
    nodes_qwen = None
    _QWEN_ERR = exc
else:  # pragma: no cover - runtime-only import
    _QWEN_ERR = None

try:
    nodes_flux = importlib.import_module("comfy_extras.nodes_flux")
except Exception as exc:  # pragma: no cover - runtime-only import
    nodes_flux = None
    _FLUX_ERR = exc
else:  # pragma: no cover - runtime-only import
    _FLUX_ERR = None


RESIZE_MODE_CHOICES = ["longest_edge", "keep_original"]
MASK_MODE_CHOICES = ["none", "use_mask", "invert_mask"]
CUSTOM_OUTPUT_TYPE = "LLS_FLUX2KLEIN_OUTPUT"
DEFAULT_REFERENCE_LATENTS_METHOD = "index_timestep_zero"


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


def _unwrap_first(result):
    if hasattr(result, "result"):
        values = result.result
        if isinstance(values, tuple):
            return values[0]
        if isinstance(values, list):
            return values[0]
        return values
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, list):
        return result[0]
    return result


def _get_image_batch_size(image) -> int:
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError("[LLS] image1 must have shape [batch, height, width, channels].")
    batch = int(shape[0])
    if batch <= 0:
        raise RuntimeError("[LLS] image1 batch size must be positive.")
    return batch


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


def _has_native_qwen_flux_runtime() -> bool:
    return (
        nodes_qwen is not None
        and getattr(nodes_qwen, "TextEncodeQwenImageEditPlus", None) is not None
    )


def _encode_with_native_qwen_flux(clip, vae, prompt: str, image1, image2=None, image3=None):
    encoder_cls = getattr(nodes_qwen, "TextEncodeQwenImageEditPlus", None)
    if encoder_cls is None:
        raise RuntimeError("[LLS] Native Qwen edit encoder is unavailable.") from _QWEN_ERR

    if hasattr(encoder_cls, "execute"):
        conditioning = _unwrap_first(
            encoder_cls.execute(
                clip,
                prompt,
                vae=vae,
                image1=image1,
                image2=image2,
                image3=image3,
            )
        )
    else:
        encoder = encoder_cls()
        conditioning = _unwrap_first(
            encoder.execute(
                clip,
                prompt,
                vae=vae,
                image1=image1,
                image2=image2,
                image3=image3,
            )
        )

    # When reference images are present, append the standard Flux Kontext method tag
    # so downstream KSampler paths can preserve multi-reference intent.
    if (image2 is not None or image3 is not None) and nodes_flux is not None:
        method_cls = getattr(nodes_flux, "FluxKontextMultiReferenceLatentMethod", None)
        if method_cls is not None:
            if hasattr(method_cls, "execute"):
                conditioning = _unwrap_first(
                    method_cls.execute(conditioning, DEFAULT_REFERENCE_LATENTS_METHOD)
                )
            else:
                method_node = method_cls()
                conditioning = _unwrap_first(
                    method_node.execute(conditioning, DEFAULT_REFERENCE_LATENTS_METHOD)
                )

    return conditioning


def _encode_with_lls_fallback(clip, prompt: str):
    try:
        conditioning, _negative, _prompt_info = LLSUniversalPromptEncode().encode(
            text_encoder=clip,
            positive_prompt=prompt,
            negative_prompt="",
            clip_skip=-1,
            model_info=None,
        )
        return conditioning
    except Exception:
        tokenize = getattr(clip, "tokenize", None)
        if not callable(tokenize):
            raise RuntimeError(
                "[LLS] No usable text encoding backend for LLS Flux2Klein Edit Text Encode. "
                "Current clip object does not expose tokenize()/encode methods."
            )

        tokens = tokenize(prompt)
        scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
        if callable(scheduled):
            try:
                return scheduled(tokens)
            except Exception as exc:
                raise RuntimeError(
                    "[LLS] No usable text encoding backend for LLS Flux2Klein Edit Text Encode. "
                    f"Scheduled CLIP encoding failed: {exc}"
                ) from exc

        plain_encode = getattr(clip, "encode_from_tokens", None)
        if not callable(plain_encode):
            raise RuntimeError(
                "[LLS] No usable text encoding backend for LLS Flux2Klein Edit Text Encode. "
                "Connect a compatible CLIP/Qwen encoder or run inside a ComfyUI build with the official edit encoders."
            )

        try:
            encoded = plain_encode(tokens, return_pooled=True, return_dict=True)
        except TypeError:
            encoded = plain_encode(tokens, return_pooled=True)
        except Exception as exc:
            raise RuntimeError(
                "[LLS] No usable text encoding backend for LLS Flux2Klein Edit Text Encode. "
                f"Plain CLIP encoding failed: {exc}"
            ) from exc

        if isinstance(encoded, dict) and "cond" in encoded:
            meta = dict(encoded)
            cond = meta.pop("cond")
            return [[cond, meta]]
        if isinstance(encoded, tuple) and len(encoded) >= 2:
            cond, pooled = encoded[:2]
            return [[cond, {"pooled_output": pooled}]]
        return [[encoded, {}]]


def _encode_conditioning(clip, vae, prompt: str, image1, image2=None, image3=None):
    native_error = None
    if _has_native_qwen_flux_runtime():
        try:
            return _encode_with_native_qwen_flux(
                clip,
                vae,
                prompt,
                image1=image1,
                image2=image2,
                image3=image3,
            ), "native_qwen_flux", None
        except Exception as exc:
            native_error = str(exc)

    try:
        return _encode_with_lls_fallback(clip, prompt), "lls_fallback", native_error
    except Exception as exc:
        message = str(exc)
        if native_error:
            message = f"{message} Native encoder fallback reason: {native_error}"
        raise RuntimeError(message) from exc


class LLSFlux2KleinEditTextEncode:
    CATEGORY = "LLS/Flux2Klein"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT", CUSTOM_OUTPUT_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    DESCRIPTION = (
        "Flux2Klein-style edit text encode node for LLS. Uses image1 as the main edit image, "
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

        conditioning, conditioning_backend, native_error = _encode_conditioning(
            clip,
            vae,
            str(prompt),
            main_image,
            image2=reference_image_2,
            image3=reference_image_3,
        )

        # latent only encodes image1, because image2/image3 are reference images rather than target latents.
        try:
            latent_tensor = vae.encode(main_image)
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] VAE encode failed for image1 / 主编辑图编码失败: {exc}"
            ) from exc

        reference_names = []
        if reference_image_2 is not None:
            reference_names.append("image2")
        if reference_image_3 is not None:
            reference_names.append("image3")

        custom_output = {
            "node_name": "LLS Flux2Klein Edit Text Encode",
            "prompt": str(prompt),
            "ref_longest_edge": int(ref_longest_edge),
            "resize_mode": str(resize_mode),
            "mask_mode": str(mask_mode),
            "has_mask": mask is not None,
            "main_image": "image1",
            "reference_images": reference_names,
            "num_reference_images": len(reference_names),
            "conditioning_backend": conditioning_backend,
            "native_error": native_error,
            "note": "image1 is main edit image; image2 and image3 are reference images; images are not concatenated",
        }

        return (
            conditioning,
            {"samples": latent_tensor},
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
