from __future__ import annotations

from .composite_utils import (
    ANCHOR_MODE_CHOICES,
    BLEND_MODE_CHOICES,
    ROTATION_ORIGIN_MODE_CHOICES,
    composite_images,
)


class LLSSimpleImageComposite:
    CATEGORY = "LLS/Image"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    DESCRIPTION = "Composite an overlay image onto a background image with translation, scale, rotation, and opacity."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "anchor_mode": (ANCHOR_MODE_CHOICES, {"default": "top_left"}),
                "rotation_origin_mode": (ROTATION_ORIGIN_MODE_CHOICES, {"default": "center"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (BLEND_MODE_CHOICES, {"default": "normal"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 32.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "keep_aspect": ("BOOLEAN", {"default": True}),
            }
        }

    def composite(
        self,
        background_image,
        overlay_image,
        x_offset,
        y_offset,
        anchor_mode,
        rotation_origin_mode,
        opacity,
        blend_mode,
        scale,
        rotation,
        keep_aspect,
    ):
        output_image = composite_images(
            background_image,
            overlay_image,
            x_offset=x_offset,
            y_offset=y_offset,
            anchor_mode=anchor_mode,
            rotation_origin_mode=rotation_origin_mode,
            opacity=opacity,
            blend_mode=blend_mode,
            scale=scale,
            rotation=rotation,
            keep_aspect=keep_aspect,
        )
        return (output_image,)


NODE_CLASS_MAPPINGS = {"LLSSimpleImageComposite": LLSSimpleImageComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleImageComposite": "LLS Simple Image Composite"}
