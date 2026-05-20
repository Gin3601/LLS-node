"""
High-level Qwen nodes.
"""
from __future__ import annotations

import importlib

from ..sampling.nodes import _get_samplers, _get_schedulers
from . import discovery, runtime

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None

_MAX_RESOLUTION = int(getattr(comfy_core_nodes, "MAX_RESOLUTION", 8192))
_QWEN_LORA_STACK_TYPE = "LLS_QWEN_LORA_STACK"


class LLSQwenLoRAStack:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "build"
    RETURN_TYPES = (_QWEN_LORA_STACK_TYPE,)
    RETURN_NAMES = ("lora_stack",)
    DESCRIPTION = (
        "Builds an ordered model-side LoRA stack for the Qwen nodes. "
        "Chain multiple nodes to apply multiple LoRAs serially."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (discovery.get_qwen_lora_choices(),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "lora_stack": (_QWEN_LORA_STACK_TYPE,),
            },
        }

    def build(self, lora_name: str, strength_model: float, lora_stack=None):
        if not lora_name or lora_name == discovery.NO_LORA_PLACEHOLDER:
            raise RuntimeError("[LLS] Missing Qwen LoRA selection.")

        stack = []
        if lora_stack is not None:
            if not isinstance(lora_stack, list):
                raise RuntimeError("[LLS] Qwen lora_stack must be a list.")
            stack.extend(lora_stack)

        stack.append(
            {
                "lora_name": str(lora_name),
                "strength_model": float(strength_model),
            }
        )
        return (stack,)


class LLSQwenTextToImage:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Compressed Qwen text-to-image node. Internally loads the Qwen model, text encoder, "
        "and VAE, supports advanced sampling controls, optional ordered LoRA stacks, "
        "and optional turbo/lightning LoRAs."
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
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "advanced": True}),
                "sampler_name": (_get_samplers(), {"default": "euler", "advanced": True}),
                "scheduler": (_get_schedulers(), {"default": "simple", "advanced": True}),
                "shift": ("FLOAT", {"default": 3.1, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "enable_turbo_mode": ("BOOLEAN", {"default": False, "advanced": True}),
                "turbo_lora_name": (
                    discovery.get_qwen_text_turbo_lora_choices(),
                    {"default": discovery.AUTO_TURBO_LORA_CHOICE, "advanced": True},
                ),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "advanced": True}),
            },
            "optional": {
                "lora_stack": (_QWEN_LORA_STACK_TYPE,),
            },
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
        negative_prompt: str,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        shift: float,
        enable_turbo_mode: bool,
        turbo_lora_name: str,
        turbo_strength: float,
        lora_stack=None,
    ):
        return (
            runtime.run_qwen_text_to_image(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                batch_size=batch_size,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                shift=shift,
                enable_turbo_mode=enable_turbo_mode,
                turbo_lora_name=turbo_lora_name,
                turbo_strength=turbo_strength,
                lora_stack=lora_stack,
            ),
        )


class LLSQwenImageEdit:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Compressed Qwen image-edit node. Internally loads the Qwen edit model, text encoder, "
        "and VAE, supports official multi-image edit conditioning, ordered LoRA stacks, "
        "and optional turbo/lightning LoRAs."
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
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "advanced": True}),
                "sampler_name": (_get_samplers(), {"default": "euler", "advanced": True}),
                "scheduler": (_get_schedulers(), {"default": "simple", "advanced": True}),
                "shift": ("FLOAT", {"default": 3.1, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "cfg_norm_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "reference_latents_method": (
                    discovery.REFERENCE_LATENTS_METHOD_CHOICES,
                    {"default": "index_timestep_zero", "advanced": True},
                ),
                "enable_turbo_mode": ("BOOLEAN", {"default": False, "advanced": True}),
                "turbo_lora_name": (
                    discovery.get_qwen_edit_turbo_lora_choices(),
                    {"default": discovery.AUTO_TURBO_LORA_CHOICE, "advanced": True},
                ),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "advanced": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "lora_stack": (_QWEN_LORA_STACK_TYPE,),
            },
        }

    def generate(
        self,
        model_name: str,
        image,
        prompt: str,
        steps: int,
        seed: int,
        image2=None,
        image3=None,
        negative_prompt: str = "",
        cfg: float = 4.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        shift: float = 3.1,
        cfg_norm_strength: float = 1.0,
        reference_latents_method: str = "index_timestep_zero",
        enable_turbo_mode: bool = False,
        turbo_lora_name: str = discovery.AUTO_TURBO_LORA_CHOICE,
        turbo_strength: float = 1.0,
        lora_stack=None,
    ):
        return (
            runtime.run_qwen_image_edit(
                model_name=model_name,
                image=image,
                image2=image2,
                image3=image3,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                shift=shift,
                cfg_norm_strength=cfg_norm_strength,
                reference_latents_method=reference_latents_method,
                enable_turbo_mode=enable_turbo_mode,
                turbo_lora_name=turbo_lora_name,
                turbo_strength=turbo_strength,
                lora_stack=lora_stack,
            ),
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSQwenLoRAStack": LLSQwenLoRAStack,
    "LLSQwenTextToImage": LLSQwenTextToImage,
    "LLSQwenImageEdit": LLSQwenImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSQwenLoRAStack": "LLS Qwen LoRA Stack",
    "LLSQwenTextToImage": "LLS Qwen Text To Image",
    "LLSQwenImageEdit": "LLS Qwen Image Edit",
}
