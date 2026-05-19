"""
LLS / ControlNet
================
功能域：ControlNet / 控制引导（对应功能分类总览第 7 节）

职责：封装 ControlNet Apply、T2I-Adapter、多路叠加、深度/法线/姿态等
      预处理节点，以及 Lotus 深度估计节点。

开发规范
--------
- CATEGORY = "LLS/ControlNet"
- 防御性导入所有 ComfyUI 内部模块
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSControlNetApply        : 标准 ControlNet 注入（conditioning + hint image）
  - LLSControlNetApplyAdv     : 高级版（支持起止步、强度渐变）
  - LLST2IAdapterApply        : T2I-Adapter 轻量级控制
  - LLSControlNetStack        : 多路 ControlNet 链式叠加
  - LLSCannyPreprocess        : Canny 边缘检测预处理
  - LLSDepthPreprocess        : 深度图预处理
  - LLSHEDPreprocess          : HED 边缘检测预处理
  - LLSLotusDepth             : Lotus 扩散式深度图生成
  - LLSLotusNormal            : Lotus 扩散式法线图生成
"""
from __future__ import annotations


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
