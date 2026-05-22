from __future__ import annotations

from typing import Any

from ..repair.repair_utils import (
    build_canvas_info,
    build_canvas_repair_mask,
    compute_crop_box,
    crop_image_to,
    crop_mask_to,
    expand_canvas_image,
    expand_canvas_mask,
    get_image_size,
    get_mask_metrics,
    make_noise_mask,
    merge_masks,
    preprocess_mask,
    recommend_denoise,
    resize_image_to,
    resize_mask_to,
    resolve_repair_scope,
    resolve_work_size,
)
from ..utils.model_info import parse_jsonish_info

try:
    import node_helpers
except Exception:
    node_helpers = None

try:
    import torch
except Exception:
    torch = None


EDIT_INFO_TYPE = "LLS_EDIT_INFO"
BACKEND_MODE_CHOICES = ["auto", "sdxl", "flux"]
EDIT_SCOPE_CHOICES = ["auto", "region", "crop", "canvas"]
PREVIEW_MODE_CHOICES = ["final", "compare", "mask", "before_after"]
AUTO_RECOMMEND_CHOICES = ["enabled", "disabled"]
CANVAS_FILL_CHOICES = ["edge", "blur", "black", "white", "neutral"]
COLOR_MATCH_CHOICES = ["disabled", "mean_std", "histogram_simple"]
BRIGHTNESS_MATCH_CHOICES = ["disabled", "enabled"]
EDGE_FIX_CHOICES = ["none", "soft", "strong"]


def normalize_edit_info(edit_info):
    info = parse_jsonish_info(edit_info)
    info.setdefault("backend_name", "")
    info.setdefault("routing_reason", "")
    info.setdefault("model_family", "SD1.5")
    info.setdefault("model_role", "base")
    info.setdefault("profile_id", "")
    info.setdefault("backend_type", "none")
    info.setdefault("sampler_strategy", "standard_k")
    info.setdefault("loader_strategy", "")
    info.setdefault("edit_scope", "region")
    info.setdefault("edit_payload_version", "1.0")
    return info


def set_conditioning_values(conditioning, values: dict[str, Any]):
    if node_helpers is not None:
        try:
            return node_helpers.conditioning_set_values(conditioning, values)
        except Exception:
            pass

    updated = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            updated.append(entry)
            continue
        token, meta = entry
        merged = dict(meta or {})
        merged.update(values)
        updated.append([token, merged])
    return updated


def build_masked_pixel_image(image, mask, *, fill_value: float):
    if hasattr(image, "masked_fill"):
        return image.masked_fill(mask, fill_value)
    if torch is not None and isinstance(image, torch.Tensor):
        mask_4d = mask.unsqueeze(-1).clamp(0.0, 1.0)
        return (image * (1.0 - mask_4d)) + (float(fill_value) * mask_4d)
    raise RuntimeError("[LLS] image object does not support masked pixel preprocessing.")


def build_native_conditioning_payload(vae, work_image, work_mask, *, latent_source: str, masked_fill_value: float):
    latent_samples = vae.encode(work_image)
    masked_pixels = build_masked_pixel_image(work_image, work_mask, fill_value=masked_fill_value)
    concat_latent_image = vae.encode(masked_pixels)
    concat_mask = make_noise_mask(work_mask, latent_samples)
    latent = {
        "samples": latent_samples,
        "noise_mask": concat_mask,
        "source": latent_source,
    }
    return latent, concat_latent_image, concat_mask


def build_workspace(
    image,
    mask,
    *,
    edit_scope,
    mask_grow,
    mask_blur,
    mask_threshold,
    invert_mask,
    crop_context,
    crop_context_factor,
    min_size,
    max_size,
    resize_mode,
    expand_left,
    expand_right,
    expand_top,
    expand_bottom,
    canvas_fill,
    auto_recommend,
):
    original_width, original_height = get_image_size(image)
    original_size = (original_width, original_height)
    processed_mask = preprocess_mask(
        mask,
        original_size,
        invert_mask=bool(invert_mask),
        mask_threshold=float(mask_threshold),
        mask_grow=int(mask_grow),
        mask_blur=float(mask_blur),
    )
    mask_bbox, mask_area_ratio = get_mask_metrics(processed_mask, original_size)
    canvas_expand = (
        max(0, int(expand_left)),
        max(0, int(expand_right)),
        max(0, int(expand_top)),
        max(0, int(expand_bottom)),
    )
    scope = resolve_repair_scope(
        edit_scope,
        mask_area_ratio=mask_area_ratio,
        mask_bbox=mask_bbox,
        image_size=original_size,
        canvas_expand=canvas_expand,
    )

    if scope == "region":
        return {
            "edit_scope": "region",
            "work_image": image,
            "work_mask": resize_mask_to(processed_mask, original_width, original_height),
            "original_size": original_size,
            "work_size": original_size,
            "crop_box": None,
            "crop_scale": 1.0,
            "canvas_expand": list(canvas_expand),
            "original_box_in_canvas": None,
            "mask_bbox": list(mask_bbox) if mask_bbox is not None else None,
            "mask_area_ratio": mask_area_ratio,
            "recommended_denoise": recommend_denoise("replace", "region", "native_fill", auto_recommend),
            "canvas_fill": canvas_fill,
        }

    if scope == "crop":
        if mask_bbox is None or mask_area_ratio <= 0.0:
            raise RuntimeError("[LLS] crop edit scope requires a non-empty mask.")
        crop_box = compute_crop_box(mask_bbox, original_size, int(crop_context), float(crop_context_factor))
        crop_width = max(1, crop_box[2] - crop_box[0])
        crop_height = max(1, crop_box[3] - crop_box[1])
        work_width, work_height, crop_scale = resolve_work_size(
            (crop_width, crop_height),
            int(min_size),
            int(max_size),
            resize_mode,
        )
        work_image = resize_image_to(crop_image_to(image, crop_box), work_width, work_height)
        work_mask = resize_mask_to(crop_mask_to(processed_mask, crop_box), work_width, work_height)
        return {
            "edit_scope": "crop",
            "work_image": work_image,
            "work_mask": work_mask,
            "original_size": original_size,
            "work_size": (work_width, work_height),
            "crop_box": list(crop_box),
            "crop_scale": crop_scale,
            "canvas_expand": list(canvas_expand),
            "original_box_in_canvas": None,
            "mask_bbox": None,
            "mask_area_ratio": mask_area_ratio,
            "recommended_denoise": recommend_denoise("replace", "crop", "native_fill", auto_recommend),
            "canvas_fill": canvas_fill,
        }

    canvas_info = build_canvas_info(
        original_size,
        int(expand_left),
        int(expand_right),
        int(expand_top),
        int(expand_bottom),
    )
    work_width, work_height = canvas_info["work_size"]
    original_box = canvas_info["original_box"]
    user_mask = expand_canvas_mask(processed_mask, work_width, work_height, original_box=original_box)
    canvas_mask = build_canvas_repair_mask(user_mask, work_width, work_height, original_box=original_box)
    work_mask = merge_masks(user_mask, canvas_mask, (work_width, work_height))
    return {
        "edit_scope": "canvas",
        "work_image": expand_canvas_image(
            image,
            work_width,
            work_height,
            fill_mode=canvas_fill,
            original_box=original_box,
        ),
        "work_mask": work_mask,
        "original_size": original_size,
        "work_size": (work_width, work_height),
        "crop_box": None,
        "crop_scale": 1.0,
        "canvas_expand": list(canvas_expand),
        "original_box_in_canvas": list(original_box),
        "mask_bbox": None,
        "mask_area_ratio": mask_area_ratio,
        "recommended_denoise": recommend_denoise("fill", "canvas", "native_fill", auto_recommend),
        "canvas_fill": canvas_fill,
    }


def build_edit_info(workspace: dict[str, Any], routing, *, backend_hints: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = routing.profile
    info = {
        "backend_name": routing.backend_name,
        "routing_reason": routing.routing_reason,
        "model_family": profile["family"],
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
        "edit_scope": workspace["edit_scope"],
        "original_size": list(workspace["original_size"]),
        "work_size": list(workspace["work_size"]),
        "crop_box": workspace["crop_box"],
        "crop_scale": workspace["crop_scale"],
        "canvas_expand": workspace["canvas_expand"],
        "original_box_in_canvas": workspace["original_box_in_canvas"],
        "recommended_denoise": float(workspace["recommended_denoise"]),
        "edit_payload_version": "1.0",
    }
    if backend_hints:
        info.update(backend_hints)
    return info


def _ensure_torch_available():
    if torch is None:
        raise RuntimeError("[LLS] Pro image edit compositing requires PyTorch.")


def _resize_mask_for_image(mask, image):
    width, height = get_image_size(image)
    return resize_mask_to(mask, width, height)


def _build_canvas_base(original_image, canvas_image, original_box):
    _ensure_torch_available()
    x1, y1, x2, y2 = original_box
    out = torch.zeros_like(canvas_image)
    out[:, y1:y2, x1:x2, :] = resize_image_to(original_image, x2 - x1, y2 - y1)
    return out


def project_work_mask(work_mask, edit_info, reference_image):
    _ensure_torch_available()
    scope = str(edit_info.get("edit_scope") or "region")
    if scope == "region":
        return _resize_mask_for_image(work_mask, reference_image)

    if scope == "crop":
        crop_box = edit_info.get("crop_box")
        if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
            raise RuntimeError("[LLS] crop edit_info must include crop_box.")
        x1, y1, x2, y2 = [int(value) for value in crop_box]
        projected = torch.zeros(
            (work_mask.shape[0], reference_image.shape[1], reference_image.shape[2]),
            dtype=work_mask.dtype,
            device=work_mask.device,
        )
        projected[:, y1:y2, x1:x2] = resize_mask_to(work_mask, x2 - x1, y2 - y1)
        return projected

    return _resize_mask_for_image(work_mask, reference_image)


def _blend_torch_images(original, generated, mask, blend_strength: float):
    _ensure_torch_available()
    mask_4d = _resize_mask_for_image(mask, original).unsqueeze(-1).clamp(0.0, 1.0) * float(blend_strength)
    return (original * (1.0 - mask_4d)) + (generated * mask_4d)


def overlay_mask_preview(image, mask, *, alpha: float = 0.4):
    if torch is None or not isinstance(image, torch.Tensor):
        return image
    mask_4d = _resize_mask_for_image(mask, image).unsqueeze(-1).clamp(0.0, 1.0) * float(alpha)
    red = torch.zeros_like(image)
    red[..., 0] = 1.0
    return (image * (1.0 - mask_4d)) + (red * mask_4d)


def _paste_torch_patch(base, patch, box):
    _ensure_torch_available()
    x1, y1, x2, y2 = box
    out = base.clone()
    out[:, y1:y2, x1:x2, :] = patch
    return out


def compose_region_result(
    original_image,
    generated_image,
    work_mask,
    edit_info,
    feather,
    color_match,
    brightness_match,
    blend_strength,
    restore_unmasked_area,
    edge_fix,
):
    del edit_info, feather, color_match, brightness_match, restore_unmasked_area, edge_fix
    original_width, original_height = get_image_size(original_image)
    base_image = resize_image_to(original_image, original_width, original_height)
    edited_image = resize_image_to(generated_image, original_width, original_height)
    return _blend_torch_images(base_image, edited_image, _resize_mask_for_image(work_mask, base_image), blend_strength)


def compose_crop_result(
    original_image,
    generated_image,
    work_mask,
    edit_info,
    feather,
    color_match,
    brightness_match,
    blend_strength,
    restore_unmasked_area,
    edge_fix,
):
    del feather, color_match, brightness_match, restore_unmasked_area, edge_fix
    crop_box = edit_info.get("crop_box")
    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
        raise RuntimeError("[LLS] crop edit_info must include crop_box.")
    crop_width = max(1, int(crop_box[2]) - int(crop_box[0]))
    crop_height = max(1, int(crop_box[3]) - int(crop_box[1]))
    resized_patch = resize_image_to(generated_image, crop_width, crop_height)
    resized_mask = resize_mask_to(work_mask, crop_width, crop_height)
    original_patch = crop_image_to(original_image, tuple(int(v) for v in crop_box))
    blended_patch = _blend_torch_images(original_patch, resized_patch, resized_mask, blend_strength)
    return _paste_torch_patch(original_image, blended_patch, tuple(int(v) for v in crop_box))


def compose_canvas_result(
    original_image,
    generated_image,
    work_mask,
    edit_info,
    feather,
    color_match,
    brightness_match,
    blend_strength,
    restore_unmasked_area,
    edge_fix,
):
    del feather, color_match, brightness_match, restore_unmasked_area, edge_fix
    original_box = edit_info.get("original_box_in_canvas")
    if not isinstance(original_box, (list, tuple)) or len(original_box) != 4:
        raise RuntimeError("[LLS] canvas edit_info must include original_box_in_canvas.")
    base_canvas = _build_canvas_base(
        original_image,
        generated_image,
        tuple(int(value) for value in original_box),
    )
    return _blend_torch_images(base_canvas, generated_image, _resize_mask_for_image(work_mask, generated_image), blend_strength)


def build_preview_image(original_image, final_image, work_mask, edit_info, preview_mode):
    _ensure_torch_available()
    scope = str(edit_info.get("edit_scope") or "region")
    if preview_mode == "final":
        return final_image
    if preview_mode == "mask":
        if scope == "canvas":
            preview_base = _build_canvas_base(
                original_image,
                final_image,
                tuple(int(value) for value in edit_info["original_box_in_canvas"]),
            )
            projected_mask = _resize_mask_for_image(work_mask, final_image)
            return overlay_mask_preview(preview_base, projected_mask)
        projected_mask = project_work_mask(work_mask, edit_info, original_image)
        return overlay_mask_preview(original_image, projected_mask)
    if scope == "canvas":
        compare_left = _build_canvas_base(
            original_image,
            final_image,
            tuple(int(value) for value in edit_info["original_box_in_canvas"]),
        )
    else:
        compare_left = resize_image_to(original_image, get_image_size(final_image)[0], get_image_size(final_image)[1])
    return torch.cat([compare_left, final_image], dim=2)
