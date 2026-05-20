from __future__ import annotations


_EDGE_FIX_CHOICES = ["off", "feather", "color_blend"]
_PREVIEW_MODE_CHOICES = ["final", "preview", "split"]


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
                "color_match": ("BOOLEAN", {"default": True}),
                "brightness_match": ("BOOLEAN", {"default": True}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (_EDGE_FIX_CHOICES, {"default": "feather"}),
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
