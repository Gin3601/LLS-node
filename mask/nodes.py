"""
LLS / Mask
==========
功能域：遮罩（Mask）操作（对应功能分类总览第 6 节）

职责：封装遮罩加载/保存、反转、裁剪/填充/缩放、腐蚀/膨胀/模糊、
      布尔运算（与/或/差）、分割、遮罩转 Latent 等节点。

开发规范
--------
- CATEGORY = "LLS/Mask"
- 防御性导入所有 ComfyUI 内部模块
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSLoadMask           : 从图像 Alpha/灰度通道提取遮罩
  - LLSSaveMask           : 导出遮罩为图像
  - LLSMaskInvert         : 遮罩反转（逻辑非）
  - LLSMaskCrop           : 遮罩裁剪
  - LLSMaskPad            : 遮罩填充
  - LLSMaskResize         : 遮罩缩放
  - LLSMaskErode          : 遮罩腐蚀
  - LLSMaskDilate         : 遮罩膨胀
  - LLSMaskBlur           : 遮罩模糊（软边缘）
  - LLSMaskAnd            : 遮罩逻辑与
  - LLSMaskOr             : 遮罩逻辑或
  - LLSMaskDiff           : 遮罩差集
  - LLSMaskToLatent       : 遮罩转 Latent（用于 Latent 遮罩合成）
  - LLSImageToMask        : 图像 → 遮罩（指定通道）
"""
from __future__ import annotations


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
