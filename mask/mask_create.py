from __future__ import annotations

from .mask_utils import (
    COMBINE_MODE_CHOICES,
    COORDINATE_MODE_CHOICES,
    MASK_INFO_TYPE,
    SHAPE_TYPE_CHOICES,
    apply_invert,
    apply_softening,
    build_area_info,
    combine_masks,
    create_reference_image,
    create_shape_mask,
    mask_to_image,
)


def _resolve_batch_for_mask_creation(input_mask) -> int:
    if input_mask is None:
        return 1
    shape = getattr(input_mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")
    return max(1, int(shape[0]))


class LLSSimpleMaskCreate:
    CATEGORY = "LLS/Mask"
    FUNCTION = "create_mask"
    RETURN_TYPES = ("MASK", "IMAGE", MASK_INFO_TYPE)
    RETURN_NAMES = ("mask", "mask_image", "area_info")
    DESCRIPTION = "Create a geometric mask image from width, height, and shape parameters."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "image_height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
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
            },
            "optional": {
                "input_mask": ("MASK",),
            },
        }

    def create_mask(
        self,
        image_width,
        image_height,
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
        input_mask=None,
    ):
        batch = _resolve_batch_for_mask_creation(input_mask)
        reference_image = create_reference_image(
            image_width=int(image_width),
            image_height=int(image_height),
            batch=batch,
            dtype=getattr(input_mask, "dtype", None),
            device=getattr(input_mask, "device", None),
        )

        created_mask, geometry_info = create_shape_mask(
            reference_image,
            shape_type=shape_type,
            coordinate_mode=coordinate_mode,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            radius=radius,
        )
        combined_mask = combine_masks(created_mask, input_mask, reference_image, combine_mode)
        final_mask = apply_invert(combined_mask, bool(invert_mask))
        final_mask = apply_softening(final_mask, feather=float(feather), blur=float(blur))
        mask_image = mask_to_image(final_mask)
        area_info = build_area_info(
            final_mask,
            geometry_info,
            feather=float(feather),
            blur=float(blur),
            invert_mask=bool(invert_mask),
            combine_mode=str(combine_mode or "replace"),
        )
        return final_mask, mask_image, area_info


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskCreate": LLSSimpleMaskCreate}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskCreate": "LLS Simple Mask Create"}
