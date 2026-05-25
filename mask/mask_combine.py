from __future__ import annotations

from .utils import align_mask_batch, normalize_mask, resize_mask_to_hw, torch


COMBINE_MODE_CHOICES = ["add", "subtract", "intersect", "xor", "max", "min"]


class LLSMaskCombine:
    CATEGORY = "LLS/Mask"
    FUNCTION = "combine"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = "Combine two masks with logical-style operations."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "mode": (COMBINE_MODE_CHOICES, {"default": "add"}),
            }
        }

    def combine(self, mask_a, mask_b, mode):
        mask_a = normalize_mask(mask_a)
        mask_b = align_mask_batch(
            resize_mask_to_hw(normalize_mask(mask_b), int(mask_a.shape[1]), int(mask_a.shape[2])),
            int(mask_a.shape[0]),
        )
        mode = str(mode or "add")

        if mode in {"add", "max"}:
            combined = torch.maximum(mask_a, mask_b)
        elif mode == "subtract":
            combined = torch.clamp(mask_a - mask_b, 0.0, 1.0)
        elif mode in {"intersect", "min"}:
            combined = torch.minimum(mask_a, mask_b)
        elif mode == "xor":
            combined = torch.abs(mask_a - mask_b)
        else:
            raise RuntimeError(f"[LLS] Unsupported mask combine mode '{mode}'.")

        return (combined.clamp(0.0, 1.0).to(dtype=mask_a.dtype),)


NODE_CLASS_MAPPINGS = {"LLSMaskCombine": LLSMaskCombine}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSMaskCombine": "LLS Mask Combine"}
