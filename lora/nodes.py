"""
LLS / LoRA
==========
功能域：LoRA / Hypernetwork / 模型微调适配器（对应功能分类总览第 8 节）

职责：封装 LoRA 加载与应用、LoRA Stack 叠加、Hypernetwork、
      LoRA 提取（模型差分）、Hook-based 动态 LoRA、IP-Adapter 等节点。

开发规范
--------
- CATEGORY = "LLS/LoRA"
- 防御性导入所有 ComfyUI 内部模块
- 对外只暴露 NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS

节点列表（待实现）
------------------
  - LLSLoRALoader             : 加载 LoRA 文件（不直接应用）
  - LLSLoRAApply              : 将 LoRA 应用到 model + clip（支持双路强度）
  - LLSLoRAStack              : 多个 LoRA 叠加（返回 stack 列表）
  - LLSLoRAStackApply         : 将 LoRA stack 批量应用
  - LLSHypernetworkApply      : 旧式超网络注入
  - LLSLoRAExtract            : 从两个 Checkpoint 提取 LoRA（模型差分）
  - LLSIPAdapterApply         : IP-Adapter 图像提示适配器
  - LLSHookLoRA               : Hook-based 条件级动态 LoRA 切换
  - LLSWeightAdapter          : DoRA / LoCon 等变体格式适配
"""
from __future__ import annotations


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
