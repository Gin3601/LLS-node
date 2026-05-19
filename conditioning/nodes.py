"""
LLS / Conditioning
==================
功能域：文本编码与条件生成（对应功能分类总览第 2 节）

CATEGORY = "LLS/Conditioning"
"""
from __future__ import annotations

from ..utils.model_info import is_flux_family, is_sdxl_family, parse_model_info


_CLIP_SKIP_CHOICES = [None] + list(range(-1, -25, -1))


def _normalize_clip_skip(value) -> int:
    if value in (None, "", "None"):
        return -1
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[LLS] Invalid clip_skip value: {value!r}") from exc
    if normalized < -24 or normalized > -1:
        raise RuntimeError(f"[LLS] clip_skip out of supported range [-24, -1]: {normalized}")
    return normalized


def _encode_tokens(clip, tokens, add_dict=None):
    scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
    if callable(scheduled):
        try:
            if add_dict:
                return scheduled(tokens, add_dict=add_dict)
            return scheduled(tokens)
        except TypeError:
            return scheduled(tokens)

    plain_encode = getattr(clip, "encode_from_tokens", None)
    if not callable(plain_encode):
        raise AttributeError(
            "CLIP object does not expose encode_from_tokens_scheduled() or encode_from_tokens()."
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


def _encode_standard_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return _encode_tokens(clip, tokens)


def _build_sdxl_tokens(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    local_tokens = clip.tokenize(prompt)

    if "l" in local_tokens:
        tokens["l"] = local_tokens["l"]
    if "g" not in tokens and "g" in local_tokens:
        tokens["g"] = local_tokens["g"]

    if "l" in tokens and "g" in tokens and len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty.get("l", [])
        while len(tokens["g"]) < len(tokens["l"]):
            tokens["g"] += empty.get("g", [])

    return tokens


def _encode_sdxl_prompt(clip, prompt: str, width: int, height: int):
    tokens = _build_sdxl_tokens(clip, prompt)
    add_dict = {
        "width": width,
        "height": height,
        "crop_w": 0,
        "crop_h": 0,
        "target_width": width,
        "target_height": height,
    }
    return _encode_tokens(clip, tokens, add_dict=add_dict)


def _build_flux_tokens(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    t5_tokens = clip.tokenize(prompt)
    if "t5xxl" in t5_tokens:
        tokens["t5xxl"] = t5_tokens["t5xxl"]
    return tokens


def _encode_flux_prompt(clip, prompt: str, guidance):
    tokens = _build_flux_tokens(clip, prompt)
    add_dict = {}
    if guidance is not None:
        add_dict["guidance"] = guidance
    return _encode_tokens(clip, tokens, add_dict=add_dict or None)


class LLSSimplePromptEncode:
    """
    简化版提示词编码节点。

    - 默认仍兼容原有 SD 风格 CLIP 编码。
    - 若连接 model_info，则按 family 自动切换到 SDXL / FLUX 编码路径。
    """

    CATEGORY = "LLS/Conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt_info")
    DESCRIPTION = (
        "Encode prompts into CONDITIONING tensors. Uses model_info when available to "
        "dispatch between SD15, SDXL, and FLUX text-encoding paths."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("CLIP",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "clip_skip": (_CLIP_SKIP_CHOICES, {"default": -1}),
            },
            "optional": {
                "model_info": ("STRING", {"default": ""}),
            },
        }

    def encode(
        self,
        text_encoder,
        positive_prompt: str,
        negative_prompt: str,
        clip_skip,
        model_info: str | None = None,
    ):
        if text_encoder is None:
            raise RuntimeError(
                "[LLS] text_encoder is None. Connect the loader output or load a compatible text encoder first."
            )

        info = parse_model_info(model_info)
        family = info["family"]
        clip_skip = _normalize_clip_skip(clip_skip)

        if clip_skip != -1:
            try:
                text_encoder = text_encoder.clone()
                text_encoder.clip_layer(clip_skip)
            except Exception:
                pass

        try:
            if is_flux_family(family):
                guidance = info.get("guidance") if info.get("guidance_embed") else None
                positive = _encode_flux_prompt(text_encoder, positive_prompt, guidance)
                negative = _encode_flux_prompt(text_encoder, "", guidance)
                negative_note = "negative_ignored_for_flux"
            elif is_sdxl_family(family):
                width = int(info.get("base_width", 1024))
                height = int(info.get("base_height", 1024))
                positive = _encode_sdxl_prompt(text_encoder, positive_prompt, width, height)
                negative = _encode_sdxl_prompt(text_encoder, negative_prompt, width, height)
                negative_note = "negative_used"
            else:
                positive = _encode_standard_prompt(text_encoder, positive_prompt)
                negative = _encode_standard_prompt(text_encoder, negative_prompt)
                negative_note = "negative_used"
        except Exception as exc:
            raise RuntimeError(f"[LLS] Failed to encode prompts for family {family}: {exc}") from exc

        pos_preview = positive_prompt[:60].replace("\n", " ")
        neg_preview = negative_prompt[:40].replace("\n", " ")
        if is_flux_family(family):
            neg_preview = ""
        prompt_info = (
            f"family={family} | clip_skip={clip_skip} | "
            f"pos=\"{pos_preview}...\" | neg=\"{neg_preview}...\" | {negative_note}"
        )

        return (positive, negative, prompt_info)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimplePromptEncode": LLSSimplePromptEncode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimplePromptEncode": "LLS Simple Prompt Encode",
}
