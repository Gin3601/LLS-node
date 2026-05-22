from __future__ import annotations

from .mask_utils import (
    COMBINE_MODE_CHOICES,
    COORDINATE_MODE_CHOICES,
    MASK_INFO_TYPE,
    OVERLAY_COLOR_CHOICES,
    SHAPE_TYPE_CHOICES,
    apply_invert,
    apply_softening,
    build_area_info,
    build_preview_image,
    combine_masks,
    create_shape_mask,
)


class LLSSimpleMaskCreate:
    CATEGORY = "LLS/Mask"
    FUNCTION = "create_mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", MASK_INFO_TYPE)
    RETURN_NAMES = ("image", "mask", "preview_image", "area_info")
    DESCRIPTION = "Create a geometric repair mask, preview overlay, and area metadata from an input image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shape_type": (SHAPE_TYPE_CHOICES, {"default": "rectangle"}),
                "coordinate_mode": (COORDINATE_MODE_CHOICES, {"default": "percent"}),
                "center_x": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "width": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 8192.0, "step": 0.01}),
                "height": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 8192.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 8192.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "combine_mode": (COMBINE_MODE_CHOICES, {"default": "replace"}),
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color": (OVERLAY_COLOR_CHOICES, {"default": "red"}),
            },
            "optional": {
                "input_mask": ("MASK",),
            },
        }

    def create_mask(
        self,
        image,
        shape_type,
        coordinate_mode,
        center_x,
        center_y,
        width,
        height,
        radius,
        feather,
        blur,
        invert_mask,
        combine_mode,
        overlay_alpha,
        overlay_color,
        input_mask=None,
    ):
        if image is None:
            raise RuntimeError("[LLS] Missing image for LLS Simple Mask Create.")

        created_mask, geometry_info = create_shape_mask(
            image,
            shape_type=shape_type,
            coordinate_mode=coordinate_mode,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            radius=radius,
        )
        combined_mask = combine_masks(created_mask, input_mask, image, combine_mode)
        final_mask = apply_invert(combined_mask, bool(invert_mask))
        final_mask = apply_softening(final_mask, feather=float(feather), blur=float(blur))
        preview_image = build_preview_image(
            image,
            final_mask,
            overlay_alpha=float(overlay_alpha),
            overlay_color=str(overlay_color or "red"),
        )
        area_info = build_area_info(
            final_mask,
            geometry_info,
            feather=float(feather),
            blur=float(blur),
            invert_mask=bool(invert_mask),
            combine_mode=str(combine_mode or "replace"),
        )
        return image, final_mask, preview_image, area_info


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskCreate": LLSSimpleMaskCreate}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskCreate": "LLS Simple Mask Create"}
