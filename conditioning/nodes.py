"""
LLS / Conditioning
==================
功能域：文本编码与条件生成（对应功能分类总览第 2 节）

CATEGORY = "LLS/Conditioning"
"""
from __future__ import annotations

from ..utils.model_info import (
    MODEL_FAMILY_CHOICES,
    get_family_defaults,
    info_to_json,
    is_flux_family,
    is_sdxl_family,
    resolve_model_family,
)


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
    家族感知的提示词编码节点。
    - `text_encoder` 与 `clip` 都使用原生 CLIP 端口类型。
    - `model_family` 默认 Auto，从 CLIP tokenizer 或对象标记推断家族。
    - 输出 `prompt_info` JSON，而不是透传上下文对象。
    """

    CATEGORY = "LLS/Conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt_info")
    DESCRIPTION = (
        "Encode prompts into CONDITIONING tensors. Dispatches between SD1.5, SDXL, "
        "SDXL Turbo, and FLUX text-encoding paths using native CLIP inputs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "clip_skip": (_CLIP_SKIP_CHOICES, {"default": -1}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "text_encoder": ("CLIP",),
                "clip": ("CLIP",),
            },
        }

    def encode(
        self,
        positive_prompt: str = "",
        negative_prompt: str = "",
        clip_skip=-1,
        model_family: str = "Auto",
        text_encoder=None,
        clip=None,
    ):
        encoder = text_encoder if text_encoder is not None else clip
        if encoder is None:
            raise RuntimeError(
                "[LLS] Missing text encoder. Connect 'text_encoder' from LLS Simple Checkpoint Loader "
                "or the legacy 'clip' input."
            )

        family = resolve_model_family(model_family, clip=encoder)
        defaults = get_family_defaults(family)
        clip_skip = _normalize_clip_skip(clip_skip)
        if is_flux_family(family):
            clip_skip = -1

        if clip_skip != -1:
            try:
                encoder = encoder.clone()
                encoder.clip_layer(clip_skip)
            except Exception:
                pass

        negative_mode = "standard"
        negative_for_encode = negative_prompt

        if family == "SDXL_TURBO" and negative_prompt.strip():
            negative_for_encode = ""
            negative_mode = "weakened_for_turbo"

        try:
            if is_flux_family(family):
                guidance = defaults.get("default_guidance")
                positive = _encode_flux_prompt(encoder, positive_prompt, guidance)
                negative = _encode_flux_prompt(encoder, "", guidance)
                negative_mode = "ignored_for_flux"
            elif is_sdxl_family(family):
                width = int(defaults.get("default_width", 1024))
                height = int(defaults.get("default_height", 1024))
                positive = _encode_sdxl_prompt(encoder, positive_prompt, width, height)
                negative = _encode_sdxl_prompt(encoder, negative_for_encode, width, height)
            else:
                positive = _encode_standard_prompt(encoder, positive_prompt)
                negative = _encode_standard_prompt(encoder, negative_for_encode)
        except Exception as exc:
            raise RuntimeError(f"[LLS] Failed to encode prompts for family {family}: {exc}") from exc

        prompt_mode = "flux" if is_flux_family(family) else "sdxl" if is_sdxl_family(family) else "clip"
        prompt_info = info_to_json(
            {
                "model_family": family,
                "text_encoder_type": defaults.get("text_encoder_type", "clip"),
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "prompt_mode": prompt_mode,
                "positive_prompt_length": len(positive_prompt),
                "negative_prompt_length": len(negative_prompt),
                "clip_skip": clip_skip,
                "negative_mode": negative_mode,
                "guidance": defaults.get("default_guidance") if is_flux_family(family) else None,
            }
        )
        return (positive, negative, prompt_info)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimplePromptEncode": LLSSimplePromptEncode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimplePromptEncode": "LLS Simple Prompt Encode",
}
