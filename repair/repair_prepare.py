from __future__ import annotations

from .repair_utils import (
    build_canvas_info,
    compute_crop_box,
    expand_canvas_image,
    expand_canvas_mask,
    get_image_size,
    get_mask_metrics,
    make_noise_mask,
    normalize_model_info,
    recommend_denoise,
    resize_image_to,
    resize_mask_to,
    resolve_repair_kernel,
    resolve_repair_scope,
    resolve_work_size,
)


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


class LLSSimpleRepairPrepare:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise")
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
        model_info=None,
        positive=None,
        negative=None,
    ):
        del canvas_fill, positive, negative

        if vae is None:
            raise RuntimeError("[LLS] Missing VAE for LLS Simple Repair Prepare.")

        image_width, image_height = get_image_size(image)
        original_size = (image_width, image_height)
        normalized_model = normalize_model_info(model_info)
        mask_bbox, mask_area_ratio = get_mask_metrics(mask, original_size)
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
            model_info=normalized_model,
        )
        warnings = list(warnings)
        if effective_kernel == "native_fill":
            effective_kernel = "vae_inpaint"
            warnings.append(
                "native_fill requested but runtime sampler support is not implemented yet; using vae_inpaint"
            )

        recommended = recommend_denoise(task_hint, effective_scope, effective_kernel, auto_recommend)
        repair_info = {
            "repair_scope": effective_scope,
            "repair_kernel": effective_kernel,
            "task_hint": str(task_hint or "auto"),
            "original_size": original_size,
            "work_size": original_size,
            "crop_box": None,
            "crop_scale": 1.0,
            "canvas_expand": canvas_expand,
            "mask_grow": int(mask_grow),
            "mask_blur": float(mask_blur),
            "mask_threshold": float(mask_threshold),
            "invert_mask": bool(invert_mask),
            "recommended_denoise": recommended,
            "model_family": normalized_model["model_family"],
            "model_role": normalized_model["model_role"],
            "repair_payload_version": "1.0",
            "has_mask": mask_bbox is not None and mask_area_ratio > 0.0,
            "mask_area_ratio": mask_area_ratio,
            "mask_bbox": list(mask_bbox) if mask_bbox is not None else None,
            "warnings": warnings,
        }

        if effective_scope == "region":
            work_image = image
            work_mask = resize_mask_to(mask, image_width, image_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_region"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)
            return latent, work_image, work_mask, repair_info, recommended

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
            work_image = resize_image_to(image, work_width, work_height)
            work_mask = resize_mask_to(mask, work_width, work_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_crop"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)

            repair_info["crop_box"] = list(crop_box)
            repair_info["crop_scale"] = crop_scale
            repair_info["work_size"] = (work_width, work_height)
            return latent, work_image, work_mask, repair_info, recommended

        if effective_scope == "canvas":
            canvas_info = build_canvas_info(
                original_size,
                int(expand_left),
                int(expand_right),
                int(expand_top),
                int(expand_bottom),
            )
            work_width, work_height = canvas_info["work_size"]
            work_image = expand_canvas_image(image, work_width, work_height)
            work_mask = expand_canvas_mask(mask, work_width, work_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_canvas"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)

            repair_info["work_size"] = (work_width, work_height)
            repair_info["original_box_in_canvas"] = list(canvas_info["original_box"])
            return latent, work_image, work_mask, repair_info, recommended

        raise RuntimeError(f"[LLS] Unsupported repair scope '{effective_scope}'.")


NODE_CLASS_MAPPINGS = {
    "LLSSimpleRepairPrepare": LLSSimpleRepairPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSSimpleRepairPrepare": "LLS Simple Repair Prepare",
}
