from __future__ import annotations

from .mask_utils import OVERLAY_COLOR_CHOICES, build_preview_image


class LLSSimpleMaskPreview:
    CATEGORY = "LLS/Mask"
    FUNCTION = "preview_mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_image",)
    DESCRIPTION = "Overlay a mask on top of an image for preview only."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color": (OVERLAY_COLOR_CHOICES, {"default": "red"}),
            }
        }

    def preview_mask(self, image, mask, overlay_alpha, overlay_color):
        if image is None:
            raise RuntimeError("[LLS] Missing image for LLS Simple Mask Preview.")
        if mask is None:
            raise RuntimeError("[LLS] Missing mask for LLS Simple Mask Preview.")
        preview_image = build_preview_image(
            image,
            mask,
            overlay_alpha=float(overlay_alpha),
            overlay_color=str(overlay_color or "red"),
        )
        return (preview_image,)


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskPreview": LLSSimpleMaskPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskPreview": "LLS Simple Mask Preview"}
