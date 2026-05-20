from __future__ import annotations


_REPAIR_SCOPE_CHOICES = ["masked_area", "bounding_box", "full_image"]
_REPAIR_KERNEL_CHOICES = ["gaussian", "box", "dilate"]
_TASK_HINT_CHOICES = ["general", "cleanup", "object_removal", "detail_fix"]
_RESIZE_MODE_CHOICES = ["fit", "pad", "stretch"]
_CANVAS_FILL_CHOICES = ["original", "mask_mean", "solid"]


class LLSSimpleRepairPrepare:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise")
    DESCRIPTION = "Prepare repair inputs, work area, and repair metadata."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "repair_scope": (_REPAIR_SCOPE_CHOICES, {"default": "masked_area"}),
                "repair_kernel": (_REPAIR_KERNEL_CHOICES, {"default": "gaussian"}),
                "task_hint": (_TASK_HINT_CHOICES, {"default": "general"}),
                "mask_grow": ("INT", {"default": 24, "min": 0, "max": 2048, "step": 1}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("BOOLEAN", {"default": True}),
                "crop_context_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "min_size": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (_RESIZE_MODE_CHOICES, {"default": "fit"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "canvas_fill": (_CANVAS_FILL_CHOICES, {"default": "original"}),
                "auto_recommend": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_info": ("STRING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    def prepare(self, *args, **kwargs):
        raise RuntimeError("[LLS] LLS Simple Repair Prepare is not implemented yet.")


NODE_CLASS_MAPPINGS = {
    "LLSSimpleRepairPrepare": LLSSimpleRepairPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLSSimpleRepairPrepare": "LLS Simple Repair Prepare",
}
