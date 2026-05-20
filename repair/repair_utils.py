from __future__ import annotations

from typing import Any

from ..utils.model_info import canonicalize_family, parse_jsonish_info


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


def resolve_adapter_mode(adapter_mode: str, model_family: str) -> str:
    mode = str(adapter_mode or "auto")
    if mode != "auto":
        return mode

    family = _normalize_model_family(model_family)
    if family in {"SD1.5", "SDXL", "SDXL_TURBO"}:
        return "sd_classic"
    if family in {"FLUX", "FLUX_DEV", "FLUX_SCHNELL"}:
        return "flux"
    if family == "SD3":
        return "sd3"
    if family == "QWEN":
        return "qwen"
    if family == "ZIMAGE":
        return "zimage"
    return "auto"


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
