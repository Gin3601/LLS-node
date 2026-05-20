"""
LLS / Utils
===========
功能域：数据类型与逻辑工具（对应功能分类总览第 14 节）

职责：封装工作流编排所需的基础类型节点、数学运算、逻辑判断、
      类型转换、分辨率辅助、曲线编辑、NOP 占位等节点。

开发规范
--------
- CATEGORY = "LLS/Utils"
- 节点尽量纯 Python 实现，减少对 ComfyUI 内部模块的依赖
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSStringLiteral          : 字符串字面量节点
  - LLSIntLiteral             : 整数字面量节点
  - LLSFloatLiteral           : 浮点数字面量节点
  - LLSBoolLiteral            : 布尔字面量节点
  - LLSMathOp                 : 数学运算（加减乘除/取模/幂/对数）
  - LLSLogicOp                : 逻辑判断（与/或/非/比较）
  - LLSIfElse                 : If/Else 条件分支
  - LLSTypeConvert            : 类型转换（数字 ↔ 字符串）
  - LLSResolutionSelector     : 常用分辨率选择器（含宽高比计算）
  - LLSBezierCurve            : 贝塞尔曲线数值生成
  - LLSStringFormat           : 字符串格式化（f-string 风格）
  - LLSStringConcat           : 字符串拼接
  - LLSRegexReplace           : 正则替换
  - LLSNop                    : 透传占位节点（调试用）
  - LLSPreviewAny             : 预览任意数据类型（String/Number/Dict）
"""
from __future__ import annotations

from .model_info import (
    FAMILY_DEFAULT_PRESET,
    MODEL_FAMILY_CHOICES,
    SIZE_PRESET_AUTO,
    get_family_defaults,
    get_sampling_preset,
    info_to_json,
    resolve_model_family,
)


# ---------- 节点类示例（基础类型，纯 Python 无外部依赖） ----------

class LLSStringLiteral:
    """输出一个字符串常量，可复用于多个下游节点。"""

    CATEGORY = "LLS/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    DESCRIPTION = "Output a constant string value."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, value: str):
        return (value,)


class LLSIntLiteral:
    """输出一个整数常量。"""

    CATEGORY = "LLS/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    DESCRIPTION = "Output a constant integer value."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2**31, "max": 2**31 - 1}),
            }
        }

    def execute(self, value: int):
        return (value,)


class LLSFloatLiteral:
    """输出一个浮点数常量。"""

    CATEGORY = "LLS/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    DESCRIPTION = "Output a constant float value."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
            }
        }

    def execute(self, value: float):
        return (value,)


class LLSResolutionSelector:
    """常用分辨率选择器，返回宽和高。"""

    CATEGORY = "LLS/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    DESCRIPTION = "Select from common resolutions or enter custom width/height."

    _PRESETS = [
        "Custom",
        "512x512",
        "768x768",
        "1024x1024",
        "512x768",
        "768x512",
        "768x1024",
        "1024x768",
        "1024x576",
        "576x1024",
        "1280x720",
        "720x1280",
        "1920x1080",
        "1080x1920",
        "2048x2048",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (cls._PRESETS, {"default": "1024x1024"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            }
        }

    def execute(self, preset: str, custom_width: int, custom_height: int):
        if preset == "Custom":
            return (custom_width, custom_height)
        w, h = preset.split("x")
        return (int(w), int(h))


class LLSGenerationConfig:
    """按模型家族输出推荐分辨率与采样配置。"""

    CATEGORY = "LLS/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "guidance", "sampler_name", "scheduler", "denoise", "config_info")
    DESCRIPTION = "Generate family-aware default width/height and sampling parameters."

    _QUALITY_PRESETS = [FAMILY_DEFAULT_PRESET, "Manual", "Fast", "Balanced", "High Quality"]
    _SIZE_PRESETS = [SIZE_PRESET_AUTO, "Custom", "512x512", "768x768", "1024x1024"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_preset": (cls._QUALITY_PRESETS, {"default": FAMILY_DEFAULT_PRESET}),
                "size_preset": (cls._SIZE_PRESETS, {"default": SIZE_PRESET_AUTO}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    def execute(self, quality_preset: str, size_preset: str, model_family: str = "Auto", model=None, clip=None):
        family = resolve_model_family(model_family, model=model, clip=clip)
        defaults = get_family_defaults(family)
        preset = get_sampling_preset(defaults, quality_preset) or {
            "steps": defaults["default_steps"],
            "cfg": defaults["default_cfg"],
            "guidance": defaults["default_guidance"],
            "sampler_name": defaults["default_sampler"],
            "scheduler": defaults["default_scheduler"],
            "denoise": defaults["default_denoise"],
        }

        if size_preset == SIZE_PRESET_AUTO:
            width = defaults["default_width"]
            height = defaults["default_height"]
        elif size_preset == "Custom":
            width = defaults["default_width"]
            height = defaults["default_height"]
        else:
            width, height = [int(part) for part in size_preset.split("x", 1)]

        guidance = preset.get("guidance")
        guidance = 0.0 if guidance is None else float(guidance)
        config_info = info_to_json(
            {
                "family": family,
                "width": width,
                "height": height,
                "steps": int(preset["steps"]),
                "cfg": float(preset["cfg"]),
                "guidance": guidance,
                "sampler_name": str(preset["sampler_name"]),
                "scheduler": str(preset["scheduler"]),
                "denoise": float(preset["denoise"]),
                "quality_preset": quality_preset,
                "size_preset": size_preset,
            }
        )
        return (
            int(width),
            int(height),
            int(preset["steps"]),
            float(preset["cfg"]),
            guidance,
            str(preset["sampler_name"]),
            str(preset["scheduler"]),
            float(preset["denoise"]),
            config_info,
        )


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSStringLiteral": LLSStringLiteral,
    "LLSIntLiteral": LLSIntLiteral,
    "LLSFloatLiteral": LLSFloatLiteral,
    "LLSResolutionSelector": LLSResolutionSelector,
    "LLSGenerationConfig": LLSGenerationConfig,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSStringLiteral": "LLS String Literal",
    "LLSIntLiteral": "LLS Int Literal",
    "LLSFloatLiteral": "LLS Float Literal",
    "LLSResolutionSelector": "LLS Resolution Selector",
    "LLSGenerationConfig": "LLS Generation Config",
}
