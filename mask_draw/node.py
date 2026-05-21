from __future__ import annotations

from .utils import build_preview_image, resolve_output_mask


_DRAW_MODE_CHOICES = ["brush", "erase"]


class LLSSimpleMaskDraw:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "draw_mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "preview_image")
    DESCRIPTION = "Resolve the edited mask output and build a preview overlay."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "draw_mode": (_DRAW_MODE_CHOICES, {"default": "brush"}),
                "brush_size": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "brush_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_state_json": ("STRING", {"default": "{}", "multiline": False, "advanced": True}),
            },
            "optional": {
                "input_mask": ("MASK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    def draw_mask(
        self,
        image,
        draw_mode,
        brush_size,
        brush_softness,
        overlay_alpha,
        invert_mask,
        mask_state_json,
        node_id,
        input_mask=None,
    ):
        del draw_mode, brush_size, brush_softness, node_id

        mask = resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json=mask_state_json,
            invert_mask=invert_mask,
        )
        preview_image = build_preview_image(
            image=image,
            mask=mask,
            overlay_alpha=overlay_alpha,
        )
        return image, mask, preview_image


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskDraw": LLSSimpleMaskDraw}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskDraw": "LLS Simple Mask Draw"}
