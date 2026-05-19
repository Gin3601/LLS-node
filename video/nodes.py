"""
LLS / Video
===========
功能域：视频生成与帧处理（对应功能分类总览第 10 节）

职责：封装视频帧加载/保存、帧插值、SVD/CogVideoX/HunyuanVideo/WanVideo
      等视频生成模型节点，以及相机轨迹控制、视频超分节点。

开发规范
--------
- CATEGORY = "LLS/Video"
- 防御性导入所有 ComfyUI 内部模块
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSLoadVideoFrames        : 从文件加载图像序列或视频文件
  - LLSSaveVideo              : 导出 GIF / WebP / MP4（调用 ffmpeg）
  - LLSFrameInterpolation     : 基于 FILM 模型的光流插值补帧
  - LLSSVDSampler             : Stable Video Diffusion 图像驱动视频生成
  - LLSCogVideoXSampler       : CogVideoX 文本/图像驱动视频生成
  - LLSHunyuanVideoSampler    : Hunyuan Video 多模态视频生成
  - LLSWanVideoSampler        : Wan Video 文生/图生视频
  - LLSLTXVideoSampler        : LTX Video 实时视频生成
  - LLSVideoUpscale           : 视频帧超分（LTX Upsampler）
  - LLSCameraControl          : 6DOF 相机路径驱动视频生成
"""
from __future__ import annotations


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
