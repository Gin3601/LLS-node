"""
LLS / Conditioning
==================
功能域：文本编码与条件生成（对应功能分类总览第 2 节）

CATEGORY = "LLS/Conditioning"
"""
from __future__ import annotations

# ---------- 防御性导入 ----------
# （本节点直接调用 clip 对象方法，无需额外导入 comfy.sd）


# ---------- 节点类 ----------

class LLSSimplePromptEncode:
    """
    简化版提示词编码节点。
    内部复用 ComfyUI 原生 clip.encode_from_tokens_scheduled()，
    自动适配 SD1.5 / SDXL（CLIP 对象内部根据模型类型处理 DualCLIP 路由）。
    """

    CATEGORY = "LLS/Conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt_info")
    DESCRIPTION = (
        "Encode positive and negative prompts into CONDITIONING tensors. "
        "Reuses ComfyUI native encode_from_tokens_scheduled for SD1.5/SDXL compatibility."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            }
        }

    def encode(self, clip, positive_prompt: str, negative_prompt: str, clip_skip: int):
        if clip is None:
            raise RuntimeError(
                "[LLS] CLIP is None. Make sure a checkpoint is loaded before connecting to this node."
            )

        # 应用 clip_skip（ComfyUI CLIP 对象支持 clip_layer）
        if clip_skip != -1:
            try:
                clip = clip.clone()
                clip.clip_layer(clip_skip)
            except Exception:
                # clip_layer 不可用时静默忽略，不阻断流程
                pass

        # 编码 positive — 复用 ComfyUI 原生编码入口
        try:
            tokens_pos = clip.tokenize(positive_prompt)
            positive = clip.encode_from_tokens_scheduled(tokens_pos)
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] Failed to encode positive prompt: {exc}"
            ) from exc

        # 编码 negative — 复用 ComfyUI 原生编码入口
        try:
            tokens_neg = clip.tokenize(negative_prompt)
            negative = clip.encode_from_tokens_scheduled(tokens_neg)
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] Failed to encode negative prompt: {exc}"
            ) from exc

        pos_preview = positive_prompt[:60].replace("\n", " ")
        neg_preview = negative_prompt[:40].replace("\n", " ")
        prompt_info = (
            f"clip_skip={clip_skip} "
            f"| pos=\"{pos_preview}...\" | neg=\"{neg_preview}...\""
        )

        return (positive, negative, prompt_info)


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimplePromptEncode": LLSSimplePromptEncode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimplePromptEncode": "LLS Simple Prompt Encode",
}
