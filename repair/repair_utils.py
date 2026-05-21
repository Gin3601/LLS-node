from __future__ import annotations

from typing import Any

from ..utils.model_info import canonicalize_family, parse_jsonish_info

try:
    import torch
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    _TORCH_ERR = exc
    torch_nn_functional = None
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    import torch.nn.functional as torch_nn_functional

    _TORCH_ERR = None

try:
    import comfy.utils as comfy_utils
except Exception as exc:  # pragma: no cover - optional runtime dependency
    comfy_utils = None
    _COMFY_UTILS_ERR = exc
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    _COMFY_UTILS_ERR = None


REPAIR_INFO_TYPE = "LLS_REPAIR_INFO"
GUIDANCE_STACK_TYPE = "LLS_GUIDANCE_STACK"
SUPPORTED_SAMPLER_FAMILIES = {"SD1.5", "SDXL", "SDXL_TURBO", "FLUX_DEV", "FLUX_SCHNELL"}

_REPAIR_SCOPE_VALUES = {"auto", "region", "crop", "canvas"}
_REPAIR_KERNEL_VALUES = {"auto", "latent_mask", "vae_inpaint", "native_fill"}
_TASK_HINT_LATENT_MASK = {"repair", "appearance", "dehaze", "deshadow", "recolor"}
_TASK_DENOISE_VALUES = {
    "repair": 0.45,
    "appearance": 0.50,
    "dehaze": 0.55,
    "deshadow": 0.55,
    "recolor": 0.55,
    "structure": 0.60,
    "content": 0.65,
    "replace": 0.72,
    "remove": 0.88,
    "fill": 0.90,
    "auto": 0.55,
}
_RESIZE_MODE_ALIASES = {
    "fit": "keep_aspect",
    "keep_aspect": "keep_aspect",
    "pad": "force_square",
    "force_square": "force_square",
    "stretch": "ranged_size",
    "ranged_size": "ranged_size",
}


def normalize_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    raw_family = raw.get("model_family") or raw.get("family")
    return {
        "model_family": _normalize_model_family(raw_family),
        "model_role": str(raw.get("model_role") or raw.get("role") or "unknown"),
        "supports_inpaint_native": _coerce_bool(raw.get("supports_inpaint_native"), False),
    }


def normalize_repair_info(repair_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(repair_info)
    info = dict(raw)
    model_info = normalize_model_info(raw.get("model_info"))

    info["repair_scope"] = str(info.get("repair_scope") or "region")
    info["repair_kernel"] = str(info.get("repair_kernel") or "vae_inpaint")
    info["task_hint"] = str(info.get("task_hint") or "auto")
    info["original_size"] = _normalize_size_pair(info.get("original_size"), default=(0, 0))
    info["work_size"] = _normalize_size_pair(info.get("work_size"), default=(0, 0))
    info["crop_box"] = _normalize_box(info.get("crop_box"))
    info["original_box_in_canvas"] = _normalize_box(info.get("original_box_in_canvas"))
    info["crop_scale"] = _normalize_optional_float(info.get("crop_scale"))
    info["canvas_expand"] = _normalize_expand(info.get("canvas_expand"))
    info["mask_grow"] = _coerce_int(info.get("mask_grow"), 8)
    info["mask_blur"] = _coerce_float(info.get("mask_blur"), 8.0)
    info["mask_threshold"] = _coerce_float(info.get("mask_threshold"), 0.5)
    info["invert_mask"] = _coerce_bool(info.get("invert_mask"), False)
    info["recommended_denoise"] = _coerce_float(info.get("recommended_denoise"), 0.55)
    info["model_family"] = _normalize_model_family(info.get("model_family") or model_info["model_family"])
    info["model_role"] = str(info.get("model_role") or model_info["model_role"])
    info["supports_inpaint_native"] = _coerce_bool(
        info.get("supports_inpaint_native", model_info["supports_inpaint_native"]),
        model_info["supports_inpaint_native"],
    )
    info["repair_payload_version"] = str(info.get("repair_payload_version") or "1.0")
    info["warnings"] = _normalize_warning_list(info.get("warnings"))
    return info


def resolve_repair_scope(
    requested_scope: str,
    *,
    mask_area_ratio: float,
    mask_bbox: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    canvas_expand: tuple[int, int, int, int],
) -> str:
    scope = str(requested_scope or "auto")
    if scope != "auto":
        if scope not in _REPAIR_SCOPE_VALUES:
            raise RuntimeError(f"[LLS] Unsupported repair_scope '{requested_scope}'.")
        return scope

    if any(_coerce_int(value, 0) > 0 for value in canvas_expand):
        return "canvas"

    if mask_bbox is None:
        raise RuntimeError("[LLS] mask is empty after preprocessing.")

    image_width, image_height = image_size
    x1, y1, x2, y2 = mask_bbox
    bbox_width_ratio = max(0, x2 - x1) / float(max(1, image_width))
    bbox_height_ratio = max(0, y2 - y1) / float(max(1, image_height))
    if mask_area_ratio <= 0.18:
        return "crop"
    if mask_area_ratio <= 0.35 and bbox_width_ratio <= 0.8 and bbox_height_ratio <= 0.8:
        return "crop"
    return "region"


def resolve_repair_kernel(
    requested_kernel: str,
    *,
    scope: str,
    task_hint: str,
    mask_area_ratio: float,
    model_info: dict[str, Any] | str | None,
) -> tuple[str, list[str]]:
    normalized_model = normalize_model_info(model_info)
    kernel = str(requested_kernel or "auto")
    warnings: list[str] = []

    if kernel != "auto":
        if kernel not in _REPAIR_KERNEL_VALUES:
            raise RuntimeError(f"[LLS] Unsupported repair_kernel '{requested_kernel}'.")
        if kernel == "native_fill" and not normalized_model["supports_inpaint_native"]:
            warnings.append("native_fill requested but unsupported; falling back to vae_inpaint")
            return "vae_inpaint", warnings
        return kernel, warnings

    if (
        normalized_model["model_role"] in {"inpaint", "fill", "edit"}
        and normalized_model["supports_inpaint_native"]
    ):
        return "native_fill", warnings

    if scope != "canvas" and mask_area_ratio <= 0.20 and task_hint in _TASK_HINT_LATENT_MASK:
        return "latent_mask", warnings

    return "vae_inpaint", warnings


def recommend_denoise(task_hint: str, scope: str, kernel: str, auto_recommend: str) -> float:
    if auto_recommend == "disabled":
        if scope == "canvas":
            return 0.90
        if scope == "crop":
            return 0.50
        if kernel == "latent_mask":
            return 0.45
        return 0.55

    base = _TASK_DENOISE_VALUES.get(str(task_hint or "auto"), 0.55)
    if scope == "canvas":
        return max(0.90, base)
    if scope == "crop":
        return max(0.30, min(0.65, base))
    return base


def clamp_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    x1 = max(0, min(_coerce_int(x1, 0), image_width))
    y1 = max(0, min(_coerce_int(y1, 0), image_height))
    x2 = max(x1, min(_coerce_int(x2, x1), image_width))
    y2 = max(y1, min(_coerce_int(y2, y1), image_height))
    return x1, y1, x2, y2


def compute_crop_box(
    mask_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    crop_context: int,
    crop_context_factor: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = mask_bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    expanded_width = (box_width + (2 * max(0, crop_context))) * max(1.0, float(crop_context_factor))
    expanded_height = (box_height + (2 * max(0, crop_context))) * max(1.0, float(crop_context_factor))
    half_width = expanded_width / 2.0
    half_height = expanded_height / 2.0
    return clamp_box(
        (
            round(center_x - half_width),
            round(center_y - half_height),
            round(center_x + half_width),
            round(center_y + half_height),
        ),
        image_size,
    )


def resolve_work_size(
    crop_size: tuple[int, int],
    min_size: int,
    max_size: int,
    resize_mode: str,
) -> tuple[int, int, float]:
    crop_width = max(1, _coerce_int(crop_size[0], 1))
    crop_height = max(1, _coerce_int(crop_size[1], 1))
    target_min = max(1, _coerce_int(min_size, crop_width))
    target_max = max(target_min, _coerce_int(max_size, target_min))
    mode = _RESIZE_MODE_ALIASES.get(str(resize_mode or "fit"), "keep_aspect")

    if mode == "force_square":
        side = min(target_max, max(target_min, max(crop_width, crop_height)))
        scale = side / float(max(crop_width, crop_height))
        return side, side, scale

    longest = max(crop_width, crop_height)
    shortest = min(crop_width, crop_height)

    if mode == "ranged_size" and target_min <= shortest and longest <= target_max:
        return crop_width, crop_height, 1.0

    if mode == "keep_aspect" and target_min <= longest <= target_max:
        return crop_width, crop_height, 1.0

    if longest < target_min:
        scale = target_min / float(longest)
    elif longest > target_max:
        scale = target_max / float(longest)
    else:
        scale = 1.0

    width = max(1, int(round(crop_width * scale)))
    height = max(1, int(round(crop_height * scale)))
    return width, height, scale


def build_canvas_info(
    image_size: tuple[int, int],
    expand_left: int,
    expand_right: int,
    expand_top: int,
    expand_bottom: int,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int]]:
    image_width, image_height = image_size
    left = max(0, _coerce_int(expand_left, 0))
    right = max(0, _coerce_int(expand_right, 0))
    top = max(0, _coerce_int(expand_top, 0))
    bottom = max(0, _coerce_int(expand_bottom, 0))

    return {
        "work_size": (image_width + left + right, image_height + top + bottom),
        "original_box": (left, top, left + image_width, top + image_height),
    }


def get_image_size(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise RuntimeError("[LLS] image must have shape [batch, height, width, channels].")

    _, height, width, _ = shape
    width = _coerce_int(width, 0)
    height = _coerce_int(height, 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("[LLS] image width and height must be positive.")
    return width, height


def get_mask_metrics(mask: Any, image_size: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, float]:
    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")

    raw_bbox = getattr(mask, "mask_bbox", None)
    mask_bbox = None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        mask_bbox = clamp_box(tuple(_coerce_int(value, 0) for value in raw_bbox), image_size)

    raw_area_ratio = getattr(mask, "mask_area_ratio", None)
    if raw_area_ratio is not None:
        mask_area_ratio = _coerce_float(raw_area_ratio, 0.0)
        mask_area_ratio = max(0.0, min(1.0, mask_area_ratio))
        return mask_bbox, mask_area_ratio

    if _is_torch_tensor(mask):
        active_map = _normalize_mask_tensor(mask) > 0.0
        if active_map.ndim == 3:
            active_map = active_map.any(dim=0)
        if not bool(active_map.any().item()):
            return None, 0.0
        points = active_map.nonzero(as_tuple=False)
        y1 = int(points[:, 0].min().item())
        y2 = int(points[:, 0].max().item()) + 1
        x1 = int(points[:, 1].min().item())
        x2 = int(points[:, 1].max().item()) + 1
        mask_bbox = clamp_box((x1, y1, x2, y2), image_size)
        return mask_bbox, float(active_map.float().mean().item())

    mask_area_ratio = 0.0
    return mask_bbox, mask_area_ratio


def resize_image_to(image: Any, width: int, height: int) -> Any:
    if hasattr(image, "resized"):
        return image.resized(width, height)
    if _is_torch_tensor(image):
        return _resize_image_tensor(image, width, height)
    current_width, current_height = get_image_size(image)
    if (current_width, current_height) == (width, height):
        return image
    raise RuntimeError("[LLS] image object does not support resizing.")


def resize_mask_to(mask: Any, width: int, height: int) -> Any:
    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")
    if hasattr(mask, "resized"):
        return mask.resized(width, height)
    if _is_torch_tensor(mask):
        return _resize_mask_tensor(mask, width, height)
    _, current_height, current_width = shape
    if (_coerce_int(current_width, 0), _coerce_int(current_height, 0)) == (width, height):
        return mask
    raise RuntimeError("[LLS] mask object does not support resizing.")


def crop_image_to(image: Any, box: tuple[int, int, int, int]) -> Any:
    cropped_box = clamp_box(box, get_image_size(image))
    if hasattr(image, "cropped"):
        return image.cropped(*cropped_box)
    if _is_torch_tensor(image):
        x1, y1, x2, y2 = cropped_box
        return image[:, y1:y2, x1:x2, :]

    current_width, current_height = get_image_size(image)
    if cropped_box == (0, 0, current_width, current_height):
        return image
    raise RuntimeError("[LLS] image object does not support cropping.")


def crop_mask_to(mask: Any, box: tuple[int, int, int, int]) -> Any:
    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")

    _, current_height, current_width = shape
    cropped_box = clamp_box(box, (_coerce_int(current_width, 0), _coerce_int(current_height, 0)))
    if hasattr(mask, "cropped"):
        return mask.cropped(*cropped_box)
    if _is_torch_tensor(mask):
        x1, y1, x2, y2 = cropped_box
        return mask[:, y1:y2, x1:x2]

    if cropped_box == (0, 0, _coerce_int(current_width, 0), _coerce_int(current_height, 0)):
        return mask
    raise RuntimeError("[LLS] mask object does not support cropping.")


def expand_canvas_image(
    image: Any,
    width: int,
    height: int,
    *,
    fill_mode: str = "edge",
    original_box: tuple[int, int, int, int] | None = None,
) -> Any:
    if hasattr(image, "canvas_expanded"):
        return image.canvas_expanded(width, height, fill_mode=fill_mode, original_box=original_box)
    if _is_torch_tensor(image):
        return _expand_canvas_image_tensor(image, width, height, fill_mode=fill_mode, original_box=original_box)
    current_width, current_height = get_image_size(image)
    if (current_width, current_height) == (width, height):
        return image
    raise RuntimeError("[LLS] image object does not support canvas expansion.")


def expand_canvas_mask(
    mask: Any,
    width: int,
    height: int,
    *,
    original_box: tuple[int, int, int, int] | None = None,
) -> Any:
    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")
    if hasattr(mask, "canvas_expanded"):
        return mask.canvas_expanded(width, height, original_box=original_box)
    if _is_torch_tensor(mask):
        return _expand_canvas_mask_tensor(mask, width, height, original_box=original_box)
    _, current_height, current_width = shape
    if (_coerce_int(current_width, 0), _coerce_int(current_height, 0)) == (width, height):
        return mask
    raise RuntimeError("[LLS] mask object does not support canvas expansion.")


def preprocess_mask(
    mask: Any,
    image_size: tuple[int, int],
    *,
    invert_mask: bool,
    mask_threshold: float,
    mask_grow: int,
    mask_blur: float,
) -> Any:
    width, height = image_size
    processed = resize_mask_to(mask, width, height)
    processed = normalize_mask_values(processed)
    if invert_mask:
        processed = invert_mask_values(processed, image_size)
    processed = apply_mask_threshold(processed, mask_threshold)
    if _coerce_int(mask_grow, 0) > 0:
        processed = apply_mask_grow(processed, _coerce_int(mask_grow, 0), image_size)
    if _coerce_float(mask_blur, 0.0) > 0.0:
        processed = apply_mask_blur(processed, _coerce_float(mask_blur, 0.0), image_size)
    return clamp_mask_values(processed)


def normalize_mask_values(mask: Any) -> Any:
    if hasattr(mask, "normalized"):
        return mask.normalized()
    if _is_torch_tensor(mask):
        return _normalize_mask_tensor(mask)
    return mask


def invert_mask_values(mask: Any, image_size: tuple[int, int]) -> Any:
    if hasattr(mask, "inverted"):
        return mask.inverted(image_size)
    if _is_torch_tensor(mask):
        return 1.0 - _normalize_mask_tensor(mask)
    raise RuntimeError("[LLS] mask object does not support invert preprocessing.")


def apply_mask_threshold(mask: Any, threshold: float) -> Any:
    if hasattr(mask, "thresholded"):
        return mask.thresholded(float(threshold))
    if _is_torch_tensor(mask):
        normalized = _normalize_mask_tensor(mask)
        return (normalized >= float(threshold)).to(normalized.dtype)
    return mask


def apply_mask_grow(mask: Any, amount: int, image_size: tuple[int, int]) -> Any:
    if hasattr(mask, "grown"):
        return mask.grown(int(amount), image_size)
    if _is_torch_tensor(mask):
        return _grow_mask_tensor(mask, int(amount))
    return mask


def apply_mask_blur(mask: Any, radius: float, image_size: tuple[int, int]) -> Any:
    if hasattr(mask, "blurred"):
        return mask.blurred(float(radius), image_size)
    if _is_torch_tensor(mask):
        return _blur_mask_tensor(mask, float(radius))
    return mask


def clamp_mask_values(mask: Any) -> Any:
    if _is_torch_tensor(mask):
        return _normalize_mask_tensor(mask)
    return mask


def build_canvas_repair_mask(
    mask: Any,
    width: int,
    height: int,
    *,
    original_box: tuple[int, int, int, int],
) -> Any:
    canvas_shape = (int(getattr(mask, "shape", [1])[0]), height, width)
    canvas_region_factory = getattr(type(mask), "canvas_region", None)
    if callable(canvas_region_factory):
        return canvas_region_factory(canvas_shape, original_box, label="canvas-region")
    if _is_torch_tensor(mask):
        repair_mask = torch.ones(canvas_shape, dtype=_normalize_mask_tensor(mask).dtype, device=mask.device)
        x1, y1, x2, y2 = original_box
        repair_mask[:, y1:y2, x1:x2] = 0.0
        return repair_mask
    raise RuntimeError("[LLS] mask object does not support canvas repair mask creation.")


def merge_masks(mask: Any, other: Any, image_size: tuple[int, int]) -> Any:
    if hasattr(mask, "merged_with"):
        return mask.merged_with(other, image_size)
    if _is_torch_tensor(mask) and _is_torch_tensor(other):
        return torch.maximum(_normalize_mask_tensor(mask), _normalize_mask_tensor(other))
    raise RuntimeError("[LLS] mask objects do not support merging.")


def make_noise_mask(mask: Any, latent_samples: Any) -> Any:
    latent_shape = getattr(latent_samples, "shape", None)
    if not isinstance(latent_shape, (list, tuple)) or len(latent_shape) != 4:
        raise RuntimeError("[LLS] latent samples must have shape [batch, channels, height, width].")

    _, _, latent_height, latent_width = latent_shape
    latent_width = _coerce_int(latent_width, 0)
    latent_height = _coerce_int(latent_height, 0)
    if latent_width <= 0 or latent_height <= 0:
        raise RuntimeError("[LLS] latent samples must have positive height and width.")
    return resize_mask_to(mask, latent_width, latent_height)


def build_preview_image(original_image: Any, final_image: Any, work_mask: Any, preview_mode: str) -> Any:
    del original_image, work_mask

    if preview_mode in {"compare", "before_after"} and hasattr(final_image, "canvas_expanded"):
        final_width, final_height = get_image_size(final_image)
        return final_image.canvas_expanded(final_width * 2, final_height)

    return final_image


def compose_region_result(
    original_image: Any,
    generated_image: Any,
    work_mask: Any,
    repair_info: dict[str, Any],
    feather: float,
    color_match: str,
    brightness_match: str,
    blend_strength: float,
    restore_unmasked_area: bool,
    edge_fix: str,
) -> Any:
    del work_mask, repair_info, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix

    original_width, original_height = get_image_size(original_image)
    return resize_image_to(generated_image, original_width, original_height)


def compose_crop_result(
    original_image: Any,
    generated_image: Any,
    work_mask: Any,
    repair_info: dict[str, Any],
    feather: float,
    color_match: str,
    brightness_match: str,
    blend_strength: float,
    restore_unmasked_area: bool,
    edge_fix: str,
) -> Any:
    del work_mask, repair_info, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix

    original_width, original_height = get_image_size(original_image)
    return resize_image_to(generated_image, original_width, original_height)


def compose_canvas_result(
    original_image: Any,
    generated_image: Any,
    work_mask: Any,
    repair_info: dict[str, Any],
    feather: float,
    color_match: str,
    brightness_match: str,
    blend_strength: float,
    restore_unmasked_area: bool,
    edge_fix: str,
) -> Any:
    del original_image, work_mask, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix

    work_size = repair_info.get("work_size")
    if not isinstance(work_size, (list, tuple)) or len(work_size) != 2:
        raise RuntimeError("[LLS] canvas repair_info must include work_size.")
    work_width = _coerce_int(work_size[0], 0)
    work_height = _coerce_int(work_size[1], 0)
    if work_width <= 0 or work_height <= 0:
        raise RuntimeError("[LLS] canvas work_size must be positive.")
    return resize_image_to(generated_image, work_width, work_height)


def resolve_adapter_mode(adapter_mode: str, model_family: str) -> str:
    mode = str(adapter_mode or "auto")
    if mode != "auto":
        return mode

    family = _normalize_model_family(model_family)
    if family.startswith("FLUX"):
        return "flux"
    if family.startswith("SD3"):
        return "sd3"
    if family.startswith("QWEN"):
        return "qwen"
    if family.startswith("ZIMAGE"):
        return "zimage"
    return "sd_classic"


def _is_torch_tensor(value: Any) -> bool:
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _resize_image_tensor(image: Any, width: int, height: int) -> Any:
    current_width, current_height = get_image_size(image)
    if (current_width, current_height) == (width, height):
        return image
    channel_first = image.movedim(-1, 1)
    resized = _upscale_tensor(channel_first, width, height, mode="bilinear")
    return resized.movedim(1, -1)


def _resize_mask_tensor(mask: Any, width: int, height: int) -> Any:
    _, current_height, current_width = tuple(mask.shape)
    if (int(current_width), int(current_height)) == (width, height):
        return mask
    channel_first = _normalize_mask_tensor(mask).unsqueeze(1)
    resized = _upscale_tensor(channel_first, width, height, mode="bilinear")
    return resized.squeeze(1)


def _upscale_tensor(channel_first: Any, width: int, height: int, *, mode: str) -> Any:
    if comfy_utils is not None:
        try:
            return comfy_utils.common_upscale(channel_first, width, height, mode, "disabled")
        except Exception:
            pass
    if torch_nn_functional is None:
        raise RuntimeError(
            "[LLS] torch or comfy.utils is required for tensor resizing outside fake-object tests."
        ) from _TORCH_ERR
    interpolate_kwargs = {
        "size": (height, width),
        "mode": mode,
    }
    if mode in {"bilinear", "bicubic"}:
        interpolate_kwargs["align_corners"] = False
    return torch_nn_functional.interpolate(channel_first, **interpolate_kwargs)


def _expand_canvas_mask_tensor(
    mask: Any,
    width: int,
    height: int,
    *,
    original_box: tuple[int, int, int, int] | None,
) -> Any:
    if original_box is None:
        original_box = (0, 0, mask.shape[2], mask.shape[1])
    expanded = torch.zeros((mask.shape[0], height, width), dtype=_normalize_mask_tensor(mask).dtype, device=mask.device)
    x1, y1, x2, y2 = original_box
    expanded[:, y1:y2, x1:x2] = _normalize_mask_tensor(mask)
    return expanded


def _expand_canvas_image_tensor(
    image: Any,
    width: int,
    height: int,
    *,
    fill_mode: str,
    original_box: tuple[int, int, int, int] | None,
) -> Any:
    if original_box is None:
        original_box = (0, 0, image.shape[2], image.shape[1])
    x1, y1, x2, y2 = original_box
    left = x1
    top = y1
    right = max(0, width - x2)
    bottom = max(0, height - y2)

    if fill_mode == "edge":
        channel_first = image.movedim(-1, 1)
        return torch_nn_functional.pad(channel_first, (left, right, top, bottom), mode="replicate").movedim(1, -1)

    if fill_mode == "blur":
        channel_first = image.movedim(-1, 1)
        padded = torch_nn_functional.pad(channel_first, (left, right, top, bottom), mode="replicate")
        blur_radius = max(1, min(8, max(left, right, top, bottom)))
        blurred = torch_nn_functional.avg_pool2d(
            padded,
            kernel_size=(blur_radius * 2) + 1,
            stride=1,
            padding=blur_radius,
        ).movedim(1, -1)
        blurred[:, y1:y2, x1:x2, :] = image
        return blurred

    fill_value = 0.0
    if fill_mode == "white":
        fill_value = 1.0
    elif fill_mode == "neutral":
        fill_value = 0.5
    canvas = image.new_full((image.shape[0], height, width, image.shape[3]), fill_value)
    canvas[:, y1:y2, x1:x2, :] = image
    return canvas


def _normalize_mask_tensor(mask: Any) -> Any:
    if not _is_torch_tensor(mask):
        return mask
    normalized = mask.float()
    if normalized.numel() == 0:
        return normalized
    max_value = float(normalized.max().item())
    min_value = float(normalized.min().item())
    if max_value > 1.0 and max_value <= 255.0 and min_value >= 0.0:
        normalized = normalized / 255.0
    return normalized.clamp(0.0, 1.0)


def _grow_mask_tensor(mask: Any, amount: int) -> Any:
    if amount <= 0:
        return _normalize_mask_tensor(mask)
    if torch_nn_functional is None:
        raise RuntimeError("[LLS] torch is required for tensor mask grow operations.") from _TORCH_ERR
    normalized = _normalize_mask_tensor(mask).unsqueeze(1)
    grown = torch_nn_functional.max_pool2d(normalized, kernel_size=(amount * 2) + 1, stride=1, padding=amount)
    return grown.squeeze(1)


def _blur_mask_tensor(mask: Any, radius: float) -> Any:
    blur_amount = max(0, int(round(radius)))
    if blur_amount <= 0:
        return _normalize_mask_tensor(mask)
    if torch_nn_functional is None:
        raise RuntimeError("[LLS] torch is required for tensor mask blur operations.") from _TORCH_ERR
    normalized = _normalize_mask_tensor(mask).unsqueeze(1)
    blurred = torch_nn_functional.avg_pool2d(
        normalized,
        kernel_size=(blur_amount * 2) + 1,
        stride=1,
        padding=blur_amount,
    )
    return blurred.squeeze(1)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none"}:
            return default
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return bool(value)


def _normalize_box(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(_coerce_int(item, 0) for item in value)
    return None


def _normalize_size_pair(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(_coerce_int(item, 0) for item in value)
    return default


def _normalize_expand(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(_coerce_int(item, 0) for item in value)
    return (0, 0, 0, 0)


def _normalize_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return _coerce_float(value, 0.0)


def _normalize_warning_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _normalize_model_family(value: Any) -> str:
    if value in (None, ""):
        return "UNKNOWN"
    family = str(value)
    return family if family == "UNKNOWN" else canonicalize_family(family)
