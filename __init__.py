"""
LLS ComfyUI Custom Nodes — 插件入口
====================================

目录结构
--------
node/
├── __init__.py           ← 本文件：自动扫描并汇总所有子包的注册表
├── nodes.py              ← 共享工具函数（不含节点类）
├── upscale/              ← 图像超分（已实现：LLSUpscaleSwitcher）
├── sampling/             ← 采样与去噪
├── latent/               ← Latent 空间操作
├── image/                ← 图像处理与后处理
├── mask/                 ← 遮罩操作
├── conditioning/         ← 文本编码与条件生成
├── model_loader/         ← 模型加载与管理
├── controlnet/           ← ControlNet / 控制引导
├── lora/                 ← LoRA / Adapter
├── video/                ← 视频生成与帧处理
├── audio/                ← 音频处理
└── utils/                ← 数据类型与逻辑工具

扩展方式
--------
1. 在对应子包的 nodes.py 中添加节点类
2. 将类名注册到该文件的 NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
3. 无需修改本文件，下次 ComfyUI 重启时自动生效

新增功能域
----------
在 node/ 下新建目录并包含 __init__.py（暴露两个映射字典）即可，
无需修改本文件——动态扫描会自动发现。
"""
from __future__ import annotations

import importlib
import types

# ---------- 子包白名单（控制加载顺序，同时作为文档） ----------
# 按功能分类总览章节顺序排列；新增子包追加到列表末尾即可。
_SUBPACKAGES: list[str] = [
    "model_loader",   # 第 1 节：模型加载与管理
    "conditioning",   # 第 2 节：文本编码与条件生成
    "sampling",       # 第 3 节：图像采样与去噪
    "qwen",           # 第 4 节：Qwen 一体化节点
    "latent",         # 第 4 节：Latent 空间操作
    "image",          # 第 5 节：图像处理与后处理
    "upscale",        # 第 5 节（超分子域）：图像超分
    "mask",           # 第 6 节：遮罩操作
    "controlnet",     # 第 7 节：ControlNet / 控制引导
    "lora",           # 第 8 节：LoRA / Adapter
    "video",          # 第 10 节：视频生成与帧处理
    "audio",          # 第 11 节：音频处理
    "utils",          # 第 14 节：数据类型与逻辑工具
]


def _merge_subpackage(pkg_name: str, classes: dict, display_names: dict) -> None:
    """
    动态导入子包并将其注册表合并到传入的字典中。

    冲突检测：若同名节点已被前面的子包注册，发出警告后跳过，
    保持先注册优先（first-wins）语义，避免静默覆盖。
    """
    full_name = f"{__name__}.{pkg_name}"
    try:
        mod: types.ModuleType = importlib.import_module(full_name)
    except Exception as exc:  # pragma: no cover
        # 子包导入失败不应阻断整个插件加载
        print(f"[LLS] WARNING: Failed to import subpackage '{pkg_name}': {exc}")
        return

    sub_classes: dict = getattr(mod, "NODE_CLASS_MAPPINGS", {})
    sub_names: dict = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for key, cls in sub_classes.items():
        if key in classes:
            existing_mod = getattr(classes[key], "__module__", "unknown")
            print(
                f"[LLS] WARNING: Node key '{key}' from '{pkg_name}' conflicts with "
                f"existing registration from '{existing_mod}'. Skipping."
            )
            continue
        classes[key] = cls
        if key in sub_names:
            display_names[key] = sub_names[key]


def _merge_root_nodes(classes: dict, display_names: dict) -> None:
    """合并根级 nodes.py 中定义的注册表。"""
    try:
        mod: types.ModuleType = importlib.import_module(f"{__name__}.nodes")
    except Exception as exc:  # pragma: no cover
        print(f"[LLS] WARNING: Failed to import root nodes module: {exc}")
        return

    root_classes: dict = getattr(mod, "NODE_CLASS_MAPPINGS", {})
    root_names: dict = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for key, cls in root_classes.items():
        if key in classes:
            existing_mod = getattr(classes[key], "__module__", "unknown")
            print(
                f"[LLS] WARNING: Root node key '{key}' conflicts with existing registration "
                f"from '{existing_mod}'. Skipping."
            )
            continue
        classes[key] = cls
        if key in root_names:
            display_names[key] = root_names[key]


# ---------- 构建全局注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

_merge_root_nodes(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

for _pkg in _SUBPACKAGES:
    _merge_subpackage(_pkg, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
