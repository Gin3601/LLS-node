from __future__ import annotations

from .backends.registry import resolve_backend
from .repair_utils import (
    build_canvas_info,
    build_canvas_repair_mask,
    compute_crop_box,
    crop_image_to,
    crop_mask_to,
    expand_canvas_image,
    expand_canvas_mask,
    get_image_size,
    get_mask_metrics,
    merge_masks,
    preprocess_mask,
    recommend_denoise,
    resize_image_to,
    resize_mask_to,
    resolve_repair_kernel,
    resolve_repair_scope,
    resolve_work_size,
)
from ..utils.model_info import resolve_edit_capabilities


_REPAIR_SCOPE_CHOICES = ["auto", "region", "crop", "canvas"]
_REPAIR_KERNEL_CHOICES = ["auto", "latent_mask", "vae_inpaint", "native_fill"]
_TASK_HINT_CHOICES = [
    "auto",
    "repair",
    "remove",
    "replace",
    "fill",
    "appearance",
    "content",
    "structure",
    "dehaze",
    "deshadow",
    "recolor",
]
_RESIZE_MODE_CHOICES = ["fit", "pad", "stretch"]
_AUTO_RECOMMEND_CHOICES = ["enabled", "disabled"]
_CANVAS_FILL_CHOICES = ["edge", "blur", "black", "white", "neutral"]


def _build_repair_info(
    *,
    repair_scope: str,
    repair_kernel: str,
    task_hint: str,
    original_size: tuple[int, int],
    work_size: tuple[int, int],
    crop_box,
    crop_scale,
    canvas_expand: list[int],
    mask_grow: int,
    mask_blur: float,
    mask_threshold: float,
    invert_mask: bool,
    recommended_denoise: float,
    routing,
    warnings: list[str],
    original_box_in_canvas=None,
    backend_hints: dict | None = None,
):
    profile = routing.profile
    info = {
        "repair_scope": repair_scope,
        "repair_kernel": repair_kernel,
        "task_hint": str(task_hint or "auto"),
        "original_size": original_size,
        "work_size": work_size,
        "crop_box": crop_box,
        "crop_scale": crop_scale,
        "canvas_expand": list(canvas_expand),
        "original_box_in_canvas": original_box_in_canvas,
        "mask_grow": int(mask_grow),
        "mask_blur": float(mask_blur),
        "mask_threshold": float(mask_threshold),
        "invert_mask": bool(invert_mask),
        "recommended_denoise": float(recommended_denoise),
        "model_family": profile["family"],
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
        "backend_name": routing.backend_name,
        "routing_reason": routing.routing_reason,
        "execution_path": routing.execution_path,
        "model_patch": "",
        "model_patch_strength": 1.0,
        "repair_payload_version": "1.0",
        "has_mask": False,
        "mask_area_ratio": 0.0,
        "mask_bbox": None,
        "warnings": list(warnings),
    }
    if backend_hints:
        info.update(backend_hints)
    return info


class LLSSimpleRepairPrepare:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise", "positive", "negative")
    DESCRIPTION = "Prepare repair inputs, work area, and repair metadata."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "repair_scope": (_REPAIR_SCOPE_CHOICES, {"default": "auto"}),
                "repair_kernel": (_REPAIR_KERNEL_CHOICES, {"default": "auto"}),
                "task_hint": (_TASK_HINT_CHOICES, {"default": "auto"}),
                "mask_grow": ("INT", {"default": 24, "min": 0, "max": 2048, "step": 1}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("INT", {"default": 64, "min": 0, "max": 512}),
                "crop_context_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "min_size": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (_RESIZE_MODE_CHOICES, {"default": "fit"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "canvas_fill": (_CANVAS_FILL_CHOICES, {"default": "edge"}),
                "auto_recommend": (_AUTO_RECOMMEND_CHOICES, {"default": "enabled"}),
            },
            "optional": {
                "model": ("MODEL",),
                "model_info": ("STRING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    def prepare(
        self,
        image,
        mask,
        vae,
        repair_scope,
        repair_kernel,
        task_hint,
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
        model=None,
        model_info=None,
        positive=None,
        negative=None,
    ):
        if vae is None:
            raise RuntimeError("[LLS] Missing VAE for LLS Simple Repair Prepare.")

        image_width, image_height = get_image_size(image)
        original_size = (image_width, image_height)
        capabilities = resolve_edit_capabilities(model=model, model_info=model_info)
        processed_mask = preprocess_mask(
            mask,
            original_size,
            invert_mask=bool(invert_mask),
            mask_threshold=float(mask_threshold),
            mask_grow=int(mask_grow),
            mask_blur=float(mask_blur),
        )
        mask_bbox, mask_area_ratio = get_mask_metrics(processed_mask, original_size)
        canvas_expand = [
            max(0, int(expand_left)),
            max(0, int(expand_right)),
            max(0, int(expand_top)),
            max(0, int(expand_bottom)),
        ]

        effective_scope = resolve_repair_scope(
            repair_scope,
            mask_area_ratio=mask_area_ratio,
            mask_bbox=mask_bbox,
            image_size=original_size,
            canvas_expand=tuple(canvas_expand),
        )
        effective_kernel, warnings = resolve_repair_kernel(
            repair_kernel,
            scope=effective_scope,
            task_hint=task_hint,
            mask_area_ratio=mask_area_ratio,
            model_info=capabilities,
        )
        backend, routing = resolve_backend(
            effective_kernel,
            model=model,
            model_info=capabilities,
        )
        recommended = recommend_denoise(task_hint, effective_scope, effective_kernel, auto_recommend)

        def finalize(work_image, work_mask, *, work_size, crop_box=None, crop_scale=1.0, original_box_in_canvas=None):
            workspace = {
                "repair_scope": effective_scope,
                "original_size": original_size,
                "work_size": work_size,
                "crop_box": crop_box,
                "crop_scale": crop_scale,
                "canvas_expand": list(canvas_expand),
                "original_box_in_canvas": original_box_in_canvas,
                "work_image": work_image,
                "work_mask": work_mask,
            }
            prepared = backend.prepare(
                model=model,
                vae=vae,
                work_image=work_image,
                work_mask=work_mask,
                positive=positive,
                negative=negative,
                workspace=workspace,
                routing=routing,
                repair_kernel=effective_kernel,
            )
            info = _build_repair_info(
                repair_scope=effective_scope,
                repair_kernel=effective_kernel,
                task_hint=task_hint,
                original_size=original_size,
                work_size=work_size,
                crop_box=None if crop_box is None else list(crop_box),
                crop_scale=crop_scale,
                canvas_expand=canvas_expand,
                mask_grow=mask_grow,
                mask_blur=mask_blur,
                mask_threshold=mask_threshold,
                invert_mask=invert_mask,
                recommended_denoise=recommended,
                routing=routing,
                warnings=list(warnings),
                original_box_in_canvas=None if original_box_in_canvas is None else list(original_box_in_canvas),
                backend_hints=prepared.get("backend_hints"),
            )
            final_bbox, final_area_ratio = get_mask_metrics(work_mask, work_size)
            info["has_mask"] = final_bbox is not None and final_area_ratio > 0.0
            info["mask_bbox"] = list(final_bbox) if final_bbox is not None else None
            info["mask_area_ratio"] = final_area_ratio
            return (
                prepared["latent"],
                work_image,
                work_mask,
                info,
                recommended,
                prepared["positive"],
                prepared["negative"],
            )

        if effective_scope == "region":
            work_image = image
            work_mask = resize_mask_to(processed_mask, image_width, image_height)
            return finalize(work_image, work_mask, work_size=original_size)

        if effective_scope == "crop":
            if mask_bbox is None or mask_area_ratio <= 0.0:
                raise RuntimeError("[LLS] crop scope requires a non-empty mask.")

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
            return finalize(
                work_image,
                work_mask,
                work_size=(work_width, work_height),
                crop_box=crop_box,
                crop_scale=crop_scale,
            )

        if effective_scope == "canvas":
            canvas_info = build_canvas_info(
                original_size,
                int(expand_left),
                int(expand_right),
                int(expand_top),
                int(expand_bottom),
            )
            work_width, work_height = canvas_info["work_size"]
            original_box = canvas_info["original_box"]
            work_image = expand_canvas_image(
                image,
                work_width,
                work_height,
                fill_mode=canvas_fill,
                original_box=original_box,
            )
            user_mask = expand_canvas_mask(
                processed_mask,
                work_width,
                work_height,
                original_box=original_box,
            )
            canvas_mask = build_canvas_repair_mask(
                user_mask,
                work_width,
                work_height,
                original_box=original_box,
            )
            work_mask = merge_masks(user_mask, canvas_mask, (work_width, work_height))
            return finalize(
                work_image,
                work_mask,
                work_size=(work_width, work_height),
                original_box_in_canvas=original_box,
            )

        raise RuntimeError(f"[LLS] Unsupported repair scope '{effective_scope}'.")


NODE_CLASS_MAPPINGS = {
    "LLSSimpleRepairPrepare": LLSSimpleRepairPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSSimpleRepairPrepare": "LLS Simple Repair Prepare",
}
