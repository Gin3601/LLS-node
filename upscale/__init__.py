"""
node.upscale
============
功能域：图像超分辨率放大（对应功能分类总览第 5 节：图像处理与后处理 → 超分）

暴露给顶层 __init__.py 的标准接口：
  NODE_CLASS_MAPPINGS        : dict[str, type]
  NODE_DISPLAY_NAME_MAPPINGS : dict[str, str]
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
