from __future__ import annotations


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
        raise RuntimeError("[LLS] LLS Simple Repair Finish is not implemented yet.")


NODE_CLASS_MAPPINGS = {
    "LLSSimpleRepairFinish": LLSSimpleRepairFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSSimpleRepairFinish": "LLS Simple Repair Finish",
}
