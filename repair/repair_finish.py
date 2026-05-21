from __future__ import annotations

from .repair_utils import (
    build_preview_image,
    compose_canvas_result,
    compose_crop_result,
    compose_region_result,
    normalize_repair_info,
)


_COLOR_MATCH_CHOICES = ["disabled", "mean_std", "histogram_simple"]
_BRIGHTNESS_MATCH_CHOICES = ["disabled", "enabled"]
_EDGE_FIX_CHOICES = ["none", "soft", "strong"]
_PREVIEW_MODE_CHOICES = ["final", "compare", "mask", "before_after"]


class LLSSimpleRepairFinish:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")
    DESCRIPTION = "Composite repaired content back into the original image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "repair_info": ("LLS_REPAIR_INFO",),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "color_match": (_COLOR_MATCH_CHOICES, {"default": "disabled"}),
                "brightness_match": (_BRIGHTNESS_MATCH_CHOICES, {"default": "enabled"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (_EDGE_FIX_CHOICES, {"default": "soft"}),
                "preview_mode": (_PREVIEW_MODE_CHOICES, {"default": "final"}),
            },
            "optional": {
                "work_mask": ("MASK",),
                "sample_info": ("STRING",),
            },
        }

    def finish(self, *args, **kwargs):
        del args

        original_image = kwargs["original_image"]
        generated_image = kwargs["generated_image"]
        repair_info = kwargs["repair_info"]
        feather = kwargs["feather"]
        color_match = kwargs["color_match"]
        brightness_match = kwargs["brightness_match"]
        blend_strength = kwargs["blend_strength"]
        restore_unmasked_area = kwargs["restore_unmasked_area"]
        edge_fix = kwargs["edge_fix"]
        preview_mode = kwargs["preview_mode"]
        work_mask = kwargs.get("work_mask")
        sample_info = kwargs.get("sample_info")
        del sample_info

        info = normalize_repair_info(repair_info)
        scope = info["repair_scope"]

        if scope == "region":
            final_image = compose_region_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        elif scope == "crop":
            final_image = compose_crop_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        elif scope == "canvas":
            final_image = compose_canvas_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        else:
            raise RuntimeError(f"[LLS] Unsupported repair_scope '{scope}'.")

        preview_image = build_preview_image(original_image, final_image, work_mask, preview_mode)
        return final_image, preview_image


NODE_CLASS_MAPPINGS = {
    "LLSSimpleRepairFinish": LLSSimpleRepairFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSSimpleRepairFinish": "LLS Simple Repair Finish",
}
