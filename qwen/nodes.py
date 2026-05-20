"""
Minimal high-level Qwen nodes.
"""
from __future__ import annotations

import importlib

from . import discovery, runtime

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None

_MAX_RESOLUTION = int(getattr(comfy_core_nodes, "MAX_RESOLUTION", 8192))


class LLSQwenTextToImage:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Minimal Qwen text-to-image node. Internally loads the Qwen model, text encoder, "
        "and VAE, then runs the official ComfyUI Qwen generation flow."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (discovery.get_qwen_text_model_choices(),),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 16, "max": _MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": _MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    def generate(
        self,
        model_name: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        batch_size: int,
    ):
        return (
            runtime.run_qwen_text_to_image(
                model_name=model_name,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                batch_size=batch_size,
            ),
        )


class LLSQwenImageEdit:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Minimal Qwen image-edit node. Internally loads the Qwen edit model, text encoder, "
        "and VAE, then runs the official ComfyUI Qwen image-edit flow."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (discovery.get_qwen_edit_model_choices(),),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def generate(
        self,
        model_name: str,
        image,
        prompt: str,
        steps: int,
        seed: int,
    ):
        return (
            runtime.run_qwen_image_edit(
                model_name=model_name,
                image=image,
                prompt=prompt,
                steps=steps,
                seed=seed,
            ),
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSQwenTextToImage": LLSQwenTextToImage,
    "LLSQwenImageEdit": LLSQwenImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSQwenTextToImage": "LLS Qwen Text To Image",
    "LLSQwenImageEdit": "LLS Qwen Image Edit",
}
