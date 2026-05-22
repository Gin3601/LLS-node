from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as torch_nn_functional
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    torch_nn_functional = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    _TORCH_ERR = None


MASK_INFO_TYPE = "LLS_MASK_INFO"
SHAPE_TYPE_CHOICES = ["rectangle", "square", "circle", "ellipse"]
COORDINATE_MODE_CHOICES = ["pixel", "percent"]
COMBINE_MODE_CHOICES = ["replace", "union", "subtract", "intersect"]
OVERLAY_COLOR_CHOICES = ["red", "green", "blue"]

_OVERLAY_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def get_image_size(image) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise RuntimeError("[LLS] image must have shape [batch, height, width, channels].")
    _, height, width, _ = tuple(shape)
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise RuntimeError("[LLS] image width and height must be positive.")
    return width, height


def create_shape_mask(
    image,
    *,
    shape_type: str,
    coordinate_mode: str,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    radius: float,
):
    _require_torch("[LLS] torch is required for mask generation.")

    image_width, image_height = get_image_size(image)
    batch = int(image.shape[0])
    dtype = image.dtype
    device = image.device

    shape_type = str(shape_type or "rectangle")
    coordinate_mode = str(coordinate_mode or "percent")
    center_px_x = _resolve_axis_value(center_x, image_width, coordinate_mode)
    center_px_y = _resolve_axis_value(center_y, image_height, coordinate_mode)
    width_px = _resolve_length_value(width, image_width, coordinate_mode)
    height_px = _resolve_length_value(height, image_height, coordinate_mode)
    radius_px = _resolve_radius_value(radius, min(image_width, image_height), coordinate_mode)

    if shape_type == "square":
        width_px = max(1.0, width_px)
        height_px = width_px
    if shape_type == "circle":
        radius_px = max(1.0, radius_px)
    if shape_type not in {"rectangle", "square", "circle", "ellipse"}:
        raise RuntimeError(f"[LLS] Unsupported shape_type '{shape_type}'.")

    xs = torch.arange(image_width, dtype=dtype, device=device) + 0.5
    ys = torch.arange(image_height, dtype=dtype, device=device) + 0.5
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # pragma: no cover - older torch fallback
        yy, xx = torch.meshgrid(ys, xs)

    if shape_type in {"rectangle", "square"}:
        half_width = width_px / 2.0
        half_height = height_px / 2.0
        x1 = center_px_x - half_width
        y1 = center_px_y - half_height
        x2 = center_px_x + half_width
        y2 = center_px_y + half_height
        inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        geometric_area_px = float(width_px * height_px)
        unclipped_bbox = (x1, y1, x2, y2)
    elif shape_type == "circle":
        inside = ((xx - center_px_x) ** 2) + ((yy - center_px_y) ** 2) <= (radius_px ** 2)
        geometric_area_px = float(math.pi * radius_px * radius_px)
        unclipped_bbox = (
            center_px_x - radius_px,
            center_px_y - radius_px,
            center_px_x + radius_px,
            center_px_y + radius_px,
        )
    else:
        rx = max(0.5, width_px / 2.0)
        ry = max(0.5, height_px / 2.0)
        inside = (((xx - center_px_x) / rx) ** 2) + (((yy - center_px_y) / ry) ** 2) <= 1.0
        geometric_area_px = float(math.pi * rx * ry)
        unclipped_bbox = (
            center_px_x - rx,
            center_px_y - ry,
            center_px_x + rx,
            center_px_y + ry,
        )

    mask_2d = inside.to(dtype=dtype)
    mask = mask_2d.unsqueeze(0).repeat(batch, 1, 1)
    clipped_bbox, clipped_by_image = _clip_bbox(unclipped_bbox, image_width, image_height)

    geometry_info = {
        "image_size": [image_width, image_height],
        "shape_type": shape_type,
        "coordinate_mode": coordinate_mode,
        "center": [float(center_px_x), float(center_px_y)],
        "width": float(width_px) if shape_type in {"rectangle", "square", "ellipse"} else None,
        "height": float(height_px) if shape_type in {"rectangle", "square", "ellipse"} else None,
        "radius": float(radius_px) if shape_type == "circle" else None,
        "bbox": clipped_bbox,
        "geometric_area_px": float(geometric_area_px),
        "clipped_by_image": bool(clipped_by_image),
    }
    return mask, geometry_info


def align_mask_to_image(mask, image):
    _require_torch("[LLS] torch is required for mask resizing.")

    if mask is None:
        return None

    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")

    image_batch = int(image.shape[0])
    image_width, image_height = get_image_size(image)
    normalized = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    if int(normalized.shape[0]) != image_batch:
        if int(normalized.shape[0]) == 1 and image_batch > 1:
            normalized = normalized.repeat(image_batch, 1, 1)
        else:
            raise RuntimeError(
                f"[LLS] mask batch size must match image batch size or be 1; got mask batch {int(normalized.shape[0])} and image batch {image_batch}."
            )

    if tuple(normalized.shape[1:]) == (image_height, image_width):
        return normalized

    resized = torch_nn_functional.interpolate(
        normalized.unsqueeze(1),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1).clamp(0.0, 1.0)


def combine_masks(created_mask, input_mask, image, combine_mode: str):
    aligned_input = align_mask_to_image(input_mask, image) if input_mask is not None else None
    if aligned_input is None:
        return created_mask

    combine_mode = str(combine_mode or "replace")
    if combine_mode == "replace":
        return created_mask
    if combine_mode == "union":
        return torch.maximum(aligned_input, created_mask)
    if combine_mode == "subtract":
        return torch.clamp(aligned_input - created_mask, 0.0, 1.0)
    if combine_mode == "intersect":
        return aligned_input * created_mask
    raise RuntimeError(f"[LLS] Unsupported combine_mode '{combine_mode}'.")


def apply_invert(mask, invert_mask: bool):
    if not invert_mask:
        return mask
    return 1.0 - mask


def apply_softening(mask, *, feather: float, blur: float):
    feather_radius = _coerce_radius(feather, limit=128.0)
    blur_radius = _coerce_radius(blur, limit=128.0)
    softened = mask
    if feather_radius > 0.0:
        softened = _box_blur_mask(softened, feather_radius)
    if blur_radius > 0.0:
        softened = _box_blur_mask(softened, blur_radius)
    return softened.clamp(0.0, 1.0)


def build_preview_image(image, mask, *, overlay_alpha: float, overlay_color: str):
    _require_torch("[LLS] torch is required for preview image generation.")

    overlay_alpha = max(0.0, min(1.0, float(overlay_alpha)))
    overlay_rgb = _OVERLAY_COLORS.get(str(overlay_color or "red"), _OVERLAY_COLORS["red"])
    overlay = torch.zeros_like(image)
    overlay[..., 0] = overlay_rgb[0]
    overlay[..., 1] = overlay_rgb[1]
    overlay[..., 2] = overlay_rgb[2]

    mask_rgb = align_mask_to_image(mask, image).unsqueeze(-1)
    strength = mask_rgb * overlay_alpha
    return (image * (1.0 - strength) + overlay * strength).clamp(0.0, 1.0)


def build_area_info(
    mask,
    geometry_info: dict,
    *,
    feather: float,
    blur: float,
    invert_mask: bool,
    combine_mode: str,
):
    _require_torch("[LLS] torch is required for area_info calculation.")

    image_width, image_height = geometry_info["image_size"]
    binary_area_px = int((mask > 0.5).sum().item())
    effective_area_px = float(mask.sum().item())
    total_area = max(1, image_width * image_height)

    info = dict(geometry_info)
    info.update(
        {
            "binary_area_px": binary_area_px,
            "effective_area_px": effective_area_px,
            "area_ratio": float(binary_area_px / float(total_area)),
            "feather": float(_coerce_radius(feather, limit=128.0)),
            "blur": float(_coerce_radius(blur, limit=128.0)),
            "invert_mask": bool(invert_mask),
            "combine_mode": str(combine_mode or "replace"),
        }
    )
    return info


def _box_blur_mask(mask, radius: float):
    kernel_radius = max(1, int(round(float(radius))))
    kernel_size = (kernel_radius * 2) + 1
    blurred = torch_nn_functional.avg_pool2d(
        mask.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_radius,
    )
    return blurred.squeeze(1)


def _clip_bbox(unclipped_bbox, image_width: int, image_height: int):
    x1, y1, x2, y2 = [float(value) for value in unclipped_bbox]
    clipped_x1 = max(0, min(int(math.floor(x1)), image_width))
    clipped_y1 = max(0, min(int(math.floor(y1)), image_height))
    clipped_x2 = max(clipped_x1, min(int(math.ceil(x2)), image_width))
    clipped_y2 = max(clipped_y1, min(int(math.ceil(y2)), image_height))
    clipped_by_image = (
        clipped_x1 != int(math.floor(x1))
        or clipped_y1 != int(math.floor(y1))
        or clipped_x2 != int(math.ceil(x2))
        or clipped_y2 != int(math.ceil(y2))
    )
    return [clipped_x1, clipped_y1, clipped_x2, clipped_y2], clipped_by_image


def _resolve_axis_value(value: float, image_extent: int, coordinate_mode: str) -> float:
    value = float(value)
    if coordinate_mode == "percent":
        return value * float(image_extent)
    return value


def _resolve_length_value(value: float, image_extent: int, coordinate_mode: str) -> float:
    value = float(value)
    if coordinate_mode == "percent":
        return max(1.0, value * float(image_extent))
    return max(1.0, value)


def _resolve_radius_value(value: float, image_extent: int, coordinate_mode: str) -> float:
    value = float(value)
    if coordinate_mode == "percent":
        return max(1.0, value * float(image_extent))
    return max(1.0, value)


def _coerce_radius(value: float, *, limit: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(limit, normalized))


def _require_torch(message: str):
    if torch is None or torch_nn_functional is None:
        raise RuntimeError(message) from _TORCH_ERR
