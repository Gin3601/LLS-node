"""
LLS / Task
==========
功能域：统一任务控制与上下文检查。
"""
from __future__ import annotations

from ..utils.task_context import (
    LLS_TASK_CONTEXT_TYPE,
    QUALITY_PRESET_CHOICES,
    TASK_CONTROLLER_MODEL_FAMILY_CHOICES,
    TASK_MODE_CHOICES,
    WORKFLOW_PRESET_CHOICES,
    create_task_context,
    task_context_to_json,
)


class LLSTaskController:
    CATEGORY = "LLS-node"
    FUNCTION = "execute"
    RETURN_TYPES = (LLS_TASK_CONTEXT_TYPE,)
    RETURN_NAMES = ("task_context",)
    DESCRIPTION = "Create a unified LLS task_context object for txt2img, img2img, and related workflows."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_mode": (TASK_MODE_CHOICES, {"default": "txt2img"}),
                "model_family": (TASK_CONTROLLER_MODEL_FAMILY_CHOICES, {"default": "auto"}),
                "workflow_preset": (WORKFLOW_PRESET_CHOICES, {"default": "simple"}),
                "quality_preset": (QUALITY_PRESET_CHOICES, {"default": "balanced"}),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "enable_controlnet": ("BOOLEAN", {"default": False}),
                "enable_reference": ("BOOLEAN", {"default": False}),
                "use_external_vae": ("BOOLEAN", {"default": False}),
                "use_external_text_encoder": ("BOOLEAN", {"default": False}),
            }
        }

    def execute(
        self,
        task_mode: str,
        model_family: str,
        workflow_preset: str,
        quality_preset: str,
        enable_upscale: bool,
        enable_controlnet: bool,
        enable_reference: bool,
        use_external_vae: bool,
        use_external_text_encoder: bool,
    ):
        return (
            create_task_context(
                task_mode=task_mode,
                model_family=model_family,
                workflow_preset=workflow_preset,
                quality_preset=quality_preset,
                enable_upscale=enable_upscale,
                enable_controlnet=enable_controlnet,
                enable_reference=enable_reference,
                use_external_vae=use_external_vae,
                use_external_text_encoder=use_external_text_encoder,
            ),
        )


class LLSTaskInspector:
    CATEGORY = "LLS-node"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("task_summary",)
    DESCRIPTION = "Render task_context as a JSON string for debugging."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_context": (LLS_TASK_CONTEXT_TYPE,),
            }
        }

    def inspect(self, task_context):
        return (task_context_to_json(task_context),)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSTaskController": LLSTaskController,
    "LLSTaskInspector": LLSTaskInspector,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSTaskController": "LLS Task Controller",
    "LLSTaskInspector": "LLS Task Inspector",
}
