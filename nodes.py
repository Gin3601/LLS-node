"""
node/nodes.py — 根级共享工具与统一门面节点
==========================================

职责：
1. 提供可被各子包复用的通用工具函数。
2. 注册根级的 `LLS Universal Image Generator` 节点。

现有子包仍可继续从本文件导入工具函数；新增的统一节点只作为门面，
具体的 SD1.5 / SDXL / FLUX 差异全部下沉到 `lls_universal/` 后端适配器。
"""
from __future__ import annotations

import contextlib
import inspect
import importlib
from typing import Any, Iterable

# ---------- 防御性导入 ----------

try:
    import torch
except Exception as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import folder_paths
except Exception as exc:
    folder_paths = None
    _FOLDER_PATHS_IMPORT_ERROR = exc
else:
    _FOLDER_PATHS_IMPORT_ERROR = None

try:
    import comfy.model_management as model_management
    import comfy.utils as comfy_utils
except Exception as exc:
    model_management = None
    comfy_utils = None
    _COMFY_IMPORT_ERROR = exc
else:
    _COMFY_IMPORT_ERROR = None

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None

try:
    from .lls_universal.dispatcher import get_backend
    from .lls_universal.request import (
        MODEL_FAMILY_CHOICES,
        TASK_MODE_CHOICES,
        LLSUniversalGenerationRequest,
    )
except Exception as exc:
    get_backend = None
    MODEL_FAMILY_CHOICES = ("SD1.5", "SDXL", "FLUX")
    TASK_MODE_CHOICES = ("txt2img",)
    LLSUniversalGenerationRequest = None
    _UNIVERSAL_IMPORT_ERROR = exc
else:
    _UNIVERSAL_IMPORT_ERROR = None

try:
    from .sampling.nodes import _get_samplers, _get_schedulers
except Exception:
    _get_samplers = None
    _get_schedulers = None


_DEFAULT_SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "ddim",
    "uni_pc",
]
_DEFAULT_SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
_MAX_RESOLUTION = int(getattr(comfy_core_nodes, "MAX_RESOLUTION", 8192))


# ---------- 共享工具函数 ----------

def _torch_inference_mode():
    """返回 torch.inference_mode() context manager； torch 不可用时返回空上下文。"""
    if torch is None or not hasattr(torch, "inference_mode"):
        return contextlib.nullcontext()
    return torch.inference_mode()


def _require_torch() -> None:
    """断言 torch 可用，否则抛出 RuntimeError。"""
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available in the current environment. "
            "Load this custom node inside ComfyUI's Python environment."
        ) from _TORCH_IMPORT_ERROR


def _require_comfy_modules() -> None:
    """断言 ComfyUI 核心模块均可用，否则抛出 RuntimeError。"""
    if folder_paths is None:
        raise RuntimeError(
            "ComfyUI folder_paths could not be imported. "
            "Place this plugin directory under ComfyUI/custom_nodes/ and restart ComfyUI."
        ) from _FOLDER_PATHS_IMPORT_ERROR
    if comfy_utils is None or model_management is None:
        raise RuntimeError(
            "ComfyUI runtime modules could not be imported. "
            "This node must run inside a ComfyUI environment."
        ) from _COMFY_IMPORT_ERROR


def _clamp_image(image: Any) -> Any:
    """将图像 tensor clamp 到 [0.0, 1.0]。"""
    if torch is None:
        return image
    return torch.clamp(image, 0.0, 1.0)


def _sorted_unique(items: Iterable[str]) -> list[str]:
    """去重并排序字符串列表（保持确定性顺序）。"""
    return sorted(dict.fromkeys(items))


def _is_oom_exception(exc: Exception) -> bool:
    """判断异常是否为显存不足（OOM）。"""
    oom_types: list[type[BaseException]] = []
    if model_management is not None:
        oom_type = getattr(model_management, "OOM_EXCEPTION", None)
        if isinstance(oom_type, type):
            oom_types.append(oom_type)
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        oom_type = getattr(cuda, "OutOfMemoryError", None)
        if isinstance(oom_type, type):
            oom_types.append(oom_type)
    if any(isinstance(exc, t) for t in oom_types):
        return True
    return "out of memory" in str(exc).lower()


def _make_progress_bar(steps: int):
    """创建 ComfyUI ProgressBar；comfy_utils 不可用时返回 None。"""
    if comfy_utils is None or not hasattr(comfy_utils, "ProgressBar"):
        return None
    return comfy_utils.ProgressBar(steps)


def _supports_argument(fn: Any, argument_name: str) -> bool:
    """检查函数签名中是否包含指定参数名。"""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return argument_name in signature.parameters


def _get_checkpoint_names() -> list[str]:
    """读取 ComfyUI checkpoints 列表；脱离 ComfyUI 时返回占位提示。"""
    if folder_paths is None:
        return ["(ComfyUI not available)"]
    try:
        names = folder_paths.get_filename_list("checkpoints")
    except Exception:
        names = []
    return names if names else ["(no checkpoints found)"]


def _get_sampler_choices() -> list[str]:
    if callable(_get_samplers):
        try:
            names = list(_get_samplers())
            if names:
                return names
        except Exception:
            pass
    return _DEFAULT_SAMPLERS


def _get_scheduler_choices() -> list[str]:
    if callable(_get_schedulers):
        try:
            names = list(_get_schedulers())
            if names:
                return names
        except Exception:
            pass
    return _DEFAULT_SCHEDULERS


# ---------- 根级统一节点 ----------

class LLSUniversalImageGenerator:
    """
    统一 txt2img 入口。

    节点本身不拼接任何“万能采样流程”；
    它只负责收集请求并交给 dispatcher 选择对应 backend。
    """

    CATEGORY = "LLS/Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Unified txt2img node that dispatches SD1.5, SDXL, and FLUX generation "
        "to family-specific backend adapters."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_family": (list(MODEL_FAMILY_CHOICES), {"default": "SD1.5"}),
                "task_mode": (list(TASK_MODE_CHOICES), {"default": "txt2img"}),
                "model_name": (_get_checkpoint_names(),),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": _MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": _MAX_RESOLUTION, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "sampler_name": (_get_sampler_choices(), {"default": "euler"}),
                "scheduler": (_get_scheduler_choices(), {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def generate(
        self,
        model_family: str,
        task_mode: str,
        model_name: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        sampler_name: str,
        scheduler: str,
        denoise: float,
    ):
        if get_backend is None or LLSUniversalGenerationRequest is None:
            raise RuntimeError(
                "[LLS] Universal generator modules could not be imported. "
                "Check the lls_universal package for syntax or import errors."
            ) from _UNIVERSAL_IMPORT_ERROR

        request = LLSUniversalGenerationRequest(
            model_family=model_family,
            task_mode=task_mode,
            model_name=model_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        ).validate()

        backend = get_backend(request.model_family)
        image = backend.generate(request)
        return (image,)


# ---------- 根级注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSUniversalImageGenerator": LLSUniversalImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSUniversalImageGenerator": "LLS Universal Image Generator",
}
