"""
LLS / Conditioning
==================
功能域：文本编码与条件生成（对应功能分类总览第 2 节）

CATEGORY = "LLS/Conditioning"
"""
from __future__ import annotations

# ---------- 防御性导入 ----------
# （本节点直接调用 clip 对象方法，无需额外导入 comfy.sd）

_CLIP_SKIP_CHOICES = [None] + list(range(-1, -25, -1))


def _encode_conditioning(clip, prompt: str):
    """
    兼容新旧 ComfyUI 的文本编码入口。

    新版本优先使用 `encode_from_tokens_scheduled()`；
    若运行环境较旧，仅存在 `encode_from_tokens()`，则手动包装成
    ComfyUI 约定的 CONDITIONING 结构。
    """
    tokens = clip.tokenize(prompt)

    scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
    if callable(scheduled):
        return scheduled(tokens)

    plain_encode = getattr(clip, "encode_from_tokens", None)
    if not callable(plain_encode):
        raise AttributeError(
            "CLIP object does not expose encode_from_tokens_scheduled() "
            "or encode_from_tokens()."
        )

    try:
        pooled_dict = plain_encode(tokens, return_pooled=True, return_dict=True)
    except TypeError:
        encoded = plain_encode(tokens, return_pooled=True)
        if isinstance(encoded, tuple):
            cond, pooled = encoded[:2]
            pooled_dict = {"cond": cond, "pooled_output": pooled}
        else:
            pooled_dict = {"cond": encoded}

    if not isinstance(pooled_dict, dict) or "cond" not in pooled_dict:
        raise RuntimeError("[LLS] Legacy CLIP encode_from_tokens() returned an unsupported value.")

    cond = pooled_dict.pop("cond")
    return [[cond, pooled_dict]]


def _normalize_clip_skip(value) -> int:
    """
    兼容旧工作流中的 null / None，以及字符串形式的 clip_skip。
    """
    if value in (None, "", "None"):
        return -1
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[LLS] Invalid clip_skip value: {value!r}") from exc

    if normalized < -24 or normalized > -1:
        raise RuntimeError(f"[LLS] clip_skip out of supported range [-24, -1]: {normalized}")
    return normalized


# ---------- 节点类 ----------

class LLSSimplePromptEncode:
    """
    简化版提示词编码节点。
    优先复用 ComfyUI 原生 clip.encode_from_tokens_scheduled()，
    旧版环境下自动回退到 encode_from_tokens()，
    以兼容 SD1.5 / SDXL 的常见 CLIP 编码路径。
    """

    CATEGORY = "LLS/Conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt_info")
    DESCRIPTION = (
        "Encode positive and negative prompts into CONDITIONING tensors. "
        "Prefers ComfyUI native encode_from_tokens_scheduled and falls back to "
        "encode_from_tokens for older runtimes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "clip_skip": (_CLIP_SKIP_CHOICES, {"default": -1}),
            }
        }

    def encode(self, clip, positive_prompt: str, negative_prompt: str, clip_skip):
        if clip is None:
            raise RuntimeError(
                "[LLS] CLIP is None. Make sure a checkpoint is loaded before connecting to this node."
            )

        clip_skip = _normalize_clip_skip(clip_skip)

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
            positive = _encode_conditioning(clip, positive_prompt)
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] Failed to encode positive prompt: {exc}"
            ) from exc

        # 编码 negative — 复用 ComfyUI 原生编码入口
        try:
            negative = _encode_conditioning(clip, negative_prompt)
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
