from .pro_edit_utils import (
    BRIGHTNESS_MATCH_CHOICES,
    COLOR_MATCH_CHOICES,
    EDGE_FIX_CHOICES,
    PREVIEW_MODE_CHOICES,
    build_preview_image,
    compose_canvas_result,
    compose_crop_result,
    compose_region_result,
    normalize_edit_info,
)


class LLSProImageEditFinish:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "edit_info": ("LLS_EDIT_INFO",),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "color_match": (COLOR_MATCH_CHOICES, {"default": "disabled"}),
                "brightness_match": (BRIGHTNESS_MATCH_CHOICES, {"default": "enabled"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (EDGE_FIX_CHOICES, {"default": "soft"}),
                "preview_mode": (PREVIEW_MODE_CHOICES, {"default": "final"}),
            },
            "optional": {
                "work_mask": ("MASK",),
                "sample_info": ("STRING",),
            },
        }

    def finish(
        self,
        original_image,
        generated_image,
        edit_info,
        feather,
        color_match,
        brightness_match,
        blend_strength,
        restore_unmasked_area,
        edge_fix,
        preview_mode,
        work_mask=None,
        sample_info=None,
    ):
        del sample_info

        info = normalize_edit_info(edit_info)
        scope = info["edit_scope"]
        if work_mask is None:
            raise RuntimeError("[LLS] LLS Pro Image Edit Finish requires work_mask.")

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
            raise RuntimeError(f"[LLS] Unsupported edit_scope '{scope}'.")

        preview_image = build_preview_image(original_image, final_image, work_mask, info, preview_mode)
        return final_image, preview_image


NODE_CLASS_MAPPINGS = {"LLSProImageEditFinish": LLSProImageEditFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProImageEditFinish": "LLS Pro Image Edit Finish"}
