"""
LLS / Audio
===========
功能域：音频处理（对应功能分类总览第 11 节）

职责：封装音频加载/保存、裁剪/拼接、重采样、幅度调整、
      音频编码器、TTS（CosyVoice）、多模态音频生成等节点。

开发规范
--------
- CATEGORY = "LLS/Audio"
- 防御性导入所有 ComfyUI 内部模块
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSLoadAudio              : 加载 WAV / MP3 / FLAC 等格式
  - LLSSaveAudio              : 导出音频文件
  - LLSAudioCrop              : 音频时间轴裁剪
  - LLSAudioConcat            : 音频拼接
  - LLSAudioResample          : 采样率转换
  - LLSAudioVolumeAdjust      : 音量缩放
  - LLSAudioBatch             : 多段音频打包
  - LLSAudioEncode            : 将音频编码为条件向量
  - LLSCosyVoiceTTS           : 文本转语音（CosyVoice）
  - LLSHunyuanAudio           : 多模态音频生成（Hunyuan Audio）
  - LLSSpectrogram            : 频谱可视化（生成频谱图像）
"""
from __future__ import annotations


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
