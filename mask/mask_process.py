from __future__ import annotations

from .utils import (
    apply_blur,
    apply_grow,
    apply_shrink,
    align_mask_batch,
    fill_mask_holes,
    normalize_mask,
    remove_small_mask_regions,
    resize_mask_to_hw,
    resolve_image_size,
)


PROCESS_OPERATION_CHOICES = [
    "passthrough",
    "threshold",
    "invert",
    "grow",
    "shrink",
    "blur",
    "feather",
    "fill_holes",
    "remove_small_regions",
    "smooth",
    "clamp",
    "resize_to_image",
]


class LLSMaskProcess:
    CATEGORY = "LLS/Mask"
    FUNCTION = "process"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = "Apply common processing operations to a single mask."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (PROCESS_OPERATION_CHOICES, {"default": "passthrough"}),
                "value_float": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "value_int": ("INT", {"default": 8, "min": -512, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def process(self, mask, operation, value_float, value_int, image=None):
        normalized = normalize_mask(mask)
        operation = str(operation or "passthrough")

        if operation == "passthrough":
            processed = normalized
        elif operation == "threshold":
            processed = (normalized >= float(value_float)).to(dtype=normalized.dtype)
        elif operation == "invert":
            processed = 1.0 - normalized
        elif operation == "grow":
            processed = apply_grow(normalized, value_int)
        elif operation == "shrink":
            processed = apply_shrink(normalized, value_int)
        elif operation == "blur":
            processed = apply_blur(normalized, value_int)
        elif operation == "feather":
            processed = apply_blur(normalized, value_int)
        elif operation == "fill_holes":
            try:
                processed = fill_mask_holes(normalized)
            except Exception:
                processed = normalized
        elif operation == "remove_small_regions":
            try:
                processed = remove_small_mask_regions(normalized, value_int)
            except Exception:
                processed = normalized
        elif operation == "smooth":
            processed = apply_blur(normalized, max(1, int(value_int) if int(value_int) > 0 else 1))
        elif operation == "clamp":
            processed = normalized
        elif operation == "resize_to_image":
            image_size = resolve_image_size(image)
            if image_size is None:
                processed = normalized
            else:
                batch, height, width = image_size
                processed = align_mask_batch(resize_mask_to_hw(normalized, height, width), batch)
        else:
            raise RuntimeError(f"[LLS] Unsupported mask operation '{operation}'.")

        return (processed.clamp(0.0, 1.0).to(dtype=normalized.dtype),)


NODE_CLASS_MAPPINGS = {"LLSMaskProcess": LLSMaskProcess}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSMaskProcess": "LLS Mask Process"}
