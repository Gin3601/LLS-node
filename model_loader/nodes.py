"""
LLS / ModelLoader
=================
功能域：模型加载与管理（对应功能分类总览第 1 节）

CATEGORY = "LLS/Model Loader"
"""
from __future__ import annotations

import importlib
from typing import Iterable

try:
    import folder_paths
except Exception as exc:
    folder_paths = None
    _FOLDER_PATHS_ERR = exc
else:
    _FOLDER_PATHS_ERR = None

try:
    import comfy.sd as comfy_sd
except Exception as exc:
    comfy_sd = None
    _COMFY_SD_ERR = exc
else:
    _COMFY_SD_ERR = None

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception as exc:
    comfy_core_nodes = None
    _CORE_NODES_ERR = exc
else:
    _CORE_NODES_ERR = None

from ..model_profiles.registry import resolve_model_profile
from ..utils.model_info import (
    MODEL_FAMILY_CHOICES,
    canonicalize_family,
    get_family_defaults,
    info_to_json,
    infer_family_from_name,
    is_flux2_family,
    is_flux_family,
    is_sdxl_family,
    tag_lls_object,
)


AUTO_PLACEHOLDER = "(auto)"
NO_MODELS_PLACEHOLDER = "(no models found)"
NO_VAE_PLACEHOLDER = "(no vae found)"
NO_TEXT_ENCODER_PLACEHOLDER = "(no text encoders found)"
LOAD_MODE_CHOICES = ["simple", "advanced"]
VAE_SOURCE_CHOICES = ["auto", "embedded", "external", "none"]
TEXT_ENCODER_SOURCE_CHOICES = ["auto", "embedded", "external", "manual"]
UNIVERSAL_VAE_SOURCE_CHOICES = ["auto", "embedded", "external"]
UNIVERSAL_TEXT_ENCODER_SOURCE_CHOICES = ["auto", "embedded", "external"]

_FLUX_CLIP_L_PATTERNS = ("clip_l.safetensors", "clip_l", "clip-l")
_FLUX_T5_PATTERNS = ("t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors", "t5xxl", "t5")
_SDXL_CLIP_L_PATTERNS = ("clip_l.safetensors", "clip_l", "clip-l")
_SDXL_CLIP_G_PATTERNS = ("clip_g.safetensors", "clip_g", "clip-g")
_FLUX_VAE_PATTERNS = ("ae.safetensors", "ae", "vae")
_FLUX2_QWEN_PATTERNS = ("qwen_3_8b.safetensors", "qwen_3_8b_fp8mixed.safetensors", "qwen_3_8b")
_FLUX2_VAE_PATTERNS = ("flux2-vae.safetensors", "flux2-vae", "flux2_vae")


def _build_capability_tags(model_name: str, family: str) -> dict[str, object]:
    profile = resolve_model_profile(
        model=None,
        model_info={
            "checkpoint_name": model_name,
            "model_name": model_name,
            "family": family,
        },
        checkpoint_name=model_name,
        family=family,
    )
    return {
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "model_role": profile["role"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }


def _get_filename_list(category: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _with_placeholder(names: list[str], placeholder: str) -> list[str]:
    cleaned = [name for name in names if name]
    if not cleaned:
        return [AUTO_PLACEHOLDER, placeholder]
    return [AUTO_PLACEHOLDER] + cleaned


def _get_model_name_choices() -> list[str]:
    checkpoints = _get_filename_list("checkpoints")
    diffusion_models = [f"diffusion_models/{name}" for name in _get_filename_list("diffusion_models")]
    names = list(dict.fromkeys(checkpoints + diffusion_models))
    return names if names else [NO_MODELS_PLACEHOLDER]


def _get_vae_choices() -> list[str]:
    if comfy_core_nodes is not None:
        vae_loader = getattr(comfy_core_nodes, "VAELoader", None)
        if vae_loader is not None and hasattr(vae_loader, "vae_list"):
            try:
                return _with_placeholder(list(vae_loader.vae_list(vae_loader)), NO_VAE_PLACEHOLDER)
            except Exception:
                pass
    return _with_placeholder(_get_filename_list("vae"), NO_VAE_PLACEHOLDER)


def _get_text_encoder_choices() -> list[str]:
    return _with_placeholder(_get_filename_list("text_encoders"), NO_TEXT_ENCODER_PLACEHOLDER)


def _normalize_choice(value: str | None, placeholder: str) -> str | None:
    if value in (None, "", AUTO_PLACEHOLDER, placeholder):
        return None
    return value


def _get_full_path(category: str, name: str) -> str | None:
    if folder_paths is None:
        return None
    try:
        return folder_paths.get_full_path_or_raise(category, name)
    except AttributeError:
        return folder_paths.get_full_path(category, name)
    except FileNotFoundError:
        return None


def _resolve_model_path(model_name: str, family: str) -> tuple[str, str]:
    if model_name == NO_MODELS_PLACEHOLDER:
        raise RuntimeError(
            "[LLS] No models were found. Put checkpoints in ComfyUI/models/checkpoints/ "
            "or FLUX diffusion models in ComfyUI/models/diffusion_models/."
        )

    if model_name.startswith("diffusion_models/"):
        short_name = model_name.split("/", 1)[1]
        model_path = _get_full_path("diffusion_models", short_name)
        if model_path is None:
            raise RuntimeError(
                f"[LLS] Missing FLUX main model '{short_name}'. "
                "Place it in ComfyUI/models/diffusion_models/."
            )
        return ("diffusion_models", model_path)

    checkpoint_path = _get_full_path("checkpoints", model_name)
    if checkpoint_path is not None:
        return ("checkpoints", checkpoint_path)

    if is_flux_family(family):
        diffusion_path = _get_full_path("diffusion_models", model_name)
        if diffusion_path is not None:
            return ("diffusion_models", diffusion_path)
        raise RuntimeError(
            f"[LLS] Missing FLUX main model '{model_name}'. "
            "Place it in ComfyUI/models/diffusion_models/."
        )

    raise RuntimeError(
        f"[LLS] Checkpoint '{model_name}' was not found. "
        "Place it in ComfyUI/models/checkpoints/."
    )


def _load_checkpoint_bundle(model_path: str):
    try:
        return comfy_sd.load_checkpoint_guess_config(
            model_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
    except Exception as exc:
        raise RuntimeError(f"[LLS] Failed to load checkpoint '{model_path}': {exc}") from exc


def _load_model(model_source: str, model_path: str):
    if model_source == "checkpoints":
        model, text_encoder, vae = _load_checkpoint_bundle(model_path)[:3]
        return model, text_encoder, vae

    try:
        model = comfy_sd.load_diffusion_model(model_path)
    except Exception as exc:
        raise RuntimeError(f"[LLS] Failed to load diffusion model '{model_path}': {exc}") from exc
    return model, None, None


def _load_external_vae(vae_name: str):
    if comfy_core_nodes is None:
        raise RuntimeError(
            "[LLS] ComfyUI core VAELoader is unavailable, so external VAE loading cannot be used."
        ) from _CORE_NODES_ERR

    vae_loader_cls = getattr(comfy_core_nodes, "VAELoader", None)
    if vae_loader_cls is None:
        raise RuntimeError("[LLS] ComfyUI core VAELoader class was not found.")
    return vae_loader_cls().load_vae(vae_name)[0]


def _load_external_text_encoder(family: str, name1: str, name2: str | None):
    if comfy_core_nodes is None:
        raise RuntimeError(
            "[LLS] ComfyUI core CLIP loaders are unavailable, so external text encoder loading cannot be used."
        ) from _CORE_NODES_ERR

    if family == "SD1.5":
        clip_loader_cls = getattr(comfy_core_nodes, "CLIPLoader", None)
        if clip_loader_cls is None:
            raise RuntimeError("[LLS] ComfyUI core CLIPLoader class was not found.")
        return clip_loader_cls().load_clip(name1, type="stable_diffusion")[0]

    if is_flux2_family(family):
        clip_loader_cls = getattr(comfy_core_nodes, "CLIPLoader", None)
        if clip_loader_cls is None:
            raise RuntimeError("[LLS] ComfyUI core CLIPLoader class was not found.")
        return clip_loader_cls().load_clip(name1, type="flux2")[0]

    dual_loader_cls = getattr(comfy_core_nodes, "DualCLIPLoader", None)
    if dual_loader_cls is None:
        raise RuntimeError("[LLS] ComfyUI core DualCLIPLoader class was not found.")
    if is_sdxl_family(family):
        return dual_loader_cls().load_clip(name1, name2, type="sdxl")[0]
    return dual_loader_cls().load_clip(name1, name2, type="flux")[0]


def _match_file(names: Iterable[str], patterns: Iterable[str]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for pattern in patterns:
        for candidate_lower, candidate in lowered.items():
            if pattern.lower() == candidate_lower or pattern.lower() in candidate_lower:
                return candidate
    return None


def _resolve_external_text_encoder_names(
    family: str,
    name1: str | None,
    name2: str | None,
) -> tuple[str | None, str | None]:
    available = _get_filename_list("text_encoders")

    if family == "SD1.5":
        if name1:
            return name1, None
        return (_match_file(available, _SDXL_CLIP_L_PATTERNS) or available[0] if available else None, None)

    if is_flux2_family(family):
        resolved_1 = name1 or _match_file(available, _FLUX2_QWEN_PATTERNS)
        return resolved_1, None

    if is_sdxl_family(family):
        resolved_1 = name1 or _match_file(available, _SDXL_CLIP_L_PATTERNS)
        resolved_2 = name2 or _match_file(available, _SDXL_CLIP_G_PATTERNS)
        return resolved_1, resolved_2

    resolved_1 = name1 or _match_file(available, _FLUX_CLIP_L_PATTERNS)
    resolved_2 = name2 or _match_file(available, _FLUX_T5_PATTERNS)
    return resolved_1, resolved_2


def _resolve_external_vae_name(family: str, name: str | None) -> str | None:
    if name:
        return name
    available = _get_filename_list("vae")
    if is_flux2_family(family):
        return _match_file(available, _FLUX2_VAE_PATTERNS)
    if is_flux_family(family):
        return _match_file(available, _FLUX_VAE_PATTERNS)
    return available[0] if available else None


def _missing_text_encoder_error(family: str, missing_role: str) -> RuntimeError:
    if family == "SD1.5":
        return RuntimeError(
            "[LLS] Missing SD1.5 text encoder. Provide external_text_encoder_1 or use a checkpoint with embedded CLIP."
        )
    if is_sdxl_family(family):
        filename = "clip_l.safetensors" if missing_role == "clip_l" else "clip_g.safetensors"
        return RuntimeError(
            f"[LLS] Missing SDXL text encoder {filename}. "
            "Place it in ComfyUI/models/text_encoders/."
        )
    if is_flux2_family(family):
        return RuntimeError(
            "[LLS] Missing Flux2/Klein text encoder qwen_3_8b.safetensors "
            "or qwen_3_8b_fp8mixed.safetensors. Place one of them in ComfyUI/models/text_encoders/."
        )
    filename = "clip_l.safetensors" if missing_role == "clip_l" else "t5xxl_fp8_e4m3fn.safetensors"
    fallback = (
        "[LLS] Missing FLUX text encoder t5xxl_fp8_e4m3fn.safetensors or t5xxl_fp16.safetensors. "
        "Place one of them in ComfyUI/models/text_encoders/."
        if missing_role == "t5xxl"
        else "[LLS] Missing FLUX text encoder clip_l.safetensors. "
        "Place it in ComfyUI/models/text_encoders/."
    )
    return RuntimeError(fallback)


def _resolve_text_encoder(
    family: str,
    source: str,
    embedded_text_encoder,
    external_name_1: str | None,
    external_name_2: str | None,
) -> tuple[object | None, str, str | None, str | None]:
    if source == "manual":
        return None, "manual", external_name_1, external_name_2

    if source == "embedded":
        if embedded_text_encoder is None:
            raise RuntimeError(
                f"[LLS] {family} was set to use embedded text encoders, but the selected model does not contain them."
            )
        return embedded_text_encoder, "embedded", None, None

    resolved_1, resolved_2 = _resolve_external_text_encoder_names(family, external_name_1, external_name_2)

    if source == "auto" and embedded_text_encoder is not None:
        return embedded_text_encoder, "embedded", None, None

    if family == "SD1.5":
        if not resolved_1:
            if source == "auto" and embedded_text_encoder is not None:
                return embedded_text_encoder, "embedded", None, None
            raise _missing_text_encoder_error(family, "clip")
        return _load_external_text_encoder(family, resolved_1, None), "external", resolved_1, None

    if is_flux2_family(family):
        if not resolved_1:
            raise _missing_text_encoder_error(family, "qwen_3_8b")
        return _load_external_text_encoder(family, resolved_1, None), "external", resolved_1, None

    required_role_1 = "clip_l"
    required_role_2 = "clip_g" if is_sdxl_family(family) else "t5xxl"
    if not resolved_1:
        raise _missing_text_encoder_error(family, required_role_1)
    if not resolved_2:
        raise _missing_text_encoder_error(family, required_role_2)
    return (
        _load_external_text_encoder(family, resolved_1, resolved_2),
        "external",
        resolved_1,
        resolved_2,
    )


def _resolve_vae(
    family: str,
    source: str,
    embedded_vae,
    external_vae_name: str | None,
) -> tuple[object | None, str, str | None]:
    if source == "none":
        if is_flux_family(family):
            raise RuntimeError(
                "[LLS] Missing FLUX AE/VAE ae.safetensors. Place it in ComfyUI/models/vae/."
            )
        return None, "none", None

    if source == "embedded":
        if embedded_vae is None:
            raise RuntimeError(
                f"[LLS] {family} was set to use an embedded VAE, but the selected model does not contain one."
            )
        return embedded_vae, "embedded", None

    resolved_vae_name = _resolve_external_vae_name(family, external_vae_name)

    if source == "auto" and embedded_vae is not None:
        return embedded_vae, "embedded", None

    if resolved_vae_name:
        return _load_external_vae(resolved_vae_name), "external", resolved_vae_name

    if is_flux2_family(family):
        raise RuntimeError(
            "[LLS] Missing Flux2/Klein VAE flux2-vae.safetensors. Place it in ComfyUI/models/vae/."
        )

    if is_flux_family(family):
        raise RuntimeError(
            "[LLS] Missing FLUX AE/VAE ae.safetensors. Place it in ComfyUI/models/vae/."
        )
    return None, "none", None


def _resolve_loader_family(model_name: str, model_family: str) -> str:
    if model_family == "Auto":
        return infer_family_from_name(model_name, "SD1.5")
    return canonicalize_family(model_family)


def _load_tagged_resources(
    *,
    model_name: str,
    model_family: str,
    load_mode: str,
    vae_source: str,
    text_encoder_source: str,
    vae_name: str,
    text_encoder_name_1: str,
    text_encoder_name_2: str,
):
    if folder_paths is None:
        raise RuntimeError(
            "[LLS] folder_paths is not available. Make sure this node runs inside ComfyUI."
        ) from _FOLDER_PATHS_ERR
    if comfy_sd is None:
        raise RuntimeError(
            "[LLS] comfy.sd is not available. Make sure this node runs inside ComfyUI."
        ) from _COMFY_SD_ERR

    family = _resolve_loader_family(model_name, model_family)
    model_source, model_path = _resolve_model_path(model_name, family)
    model, embedded_text_encoder, embedded_vae = _load_model(model_source, model_path)
    if model is None:
        raise RuntimeError(f"[LLS] The selected model '{model_name}' did not produce a valid diffusion model.")

    external_vae_name = _normalize_choice(vae_name, NO_VAE_PLACEHOLDER)
    external_text_encoder_1 = _normalize_choice(text_encoder_name_1, NO_TEXT_ENCODER_PLACEHOLDER)
    external_text_encoder_2 = _normalize_choice(text_encoder_name_2, NO_TEXT_ENCODER_PLACEHOLDER)

    text_encoder, resolved_text_encoder_source, resolved_te_1, resolved_te_2 = _resolve_text_encoder(
        family=family,
        source=text_encoder_source,
        embedded_text_encoder=embedded_text_encoder,
        external_name_1=external_text_encoder_1,
        external_name_2=external_text_encoder_2,
    )
    vae, resolved_vae_source, resolved_vae_name = _resolve_vae(
        family=family,
        source=vae_source,
        embedded_vae=embedded_vae,
        external_vae_name=external_vae_name,
    )

    defaults = get_family_defaults(family)
    capability_tags = _build_capability_tags(model_name, family)
    resolved_text_encoder_name = resolved_te_1 or ("embedded" if text_encoder is not None else None)
    resolved_text_encoder_name_1 = resolved_te_1 or ("embedded" if text_encoder is not None and not resolved_te_2 else None)
    resolved_text_encoder_name_2 = resolved_te_2
    if resolved_te_1 and resolved_te_2:
        resolved_text_encoder_name = ", ".join([resolved_te_1, resolved_te_2])
    resolved_vae_label = resolved_vae_name or ("embedded" if vae is not None else None)

    tag_lls_object(
        model,
        family=family,
        model_name=model_name,
        checkpoint_name=model_name,
        load_mode=load_mode,
        **capability_tags,
    )
    tag_lls_object(
        text_encoder,
        family=family,
        model_name=model_name,
        checkpoint_name=model_name,
        text_encoder_type=defaults["text_encoder_type"],
        text_encoder_source=resolved_text_encoder_source,
        text_encoder_name=resolved_text_encoder_name,
        text_encoder_name_1=resolved_text_encoder_name_1,
        text_encoder_name_2=resolved_text_encoder_name_2,
        **capability_tags,
    )
    tag_lls_object(
        vae,
        family=family,
        model_name=model_name,
        checkpoint_name=model_name,
        vae_name=resolved_vae_label,
        vae_source=resolved_vae_source,
        **capability_tags,
    )

    model_info = {
        "model_family": family,
        "checkpoint_name": model_name,
        "model_name": model_name,
        "load_mode": load_mode,
        "model_source": model_source,
        "profile_id": capability_tags["profile_id"],
        "backend_type": capability_tags["backend_type"],
        "sampler_strategy": capability_tags["sampler_strategy"],
        "loader_strategy": capability_tags["loader_strategy"],
        "model_role": capability_tags["model_role"],
        "supports_inpaint_native": capability_tags["supports_inpaint_native"],
        "supports_image_edit_native": capability_tags["supports_image_edit_native"],
        "preferred_edit_backend": capability_tags["preferred_edit_backend"],
        "text_encoder_type": defaults["text_encoder_type"],
        "text_encoder_source": resolved_text_encoder_source,
        "text_encoder_name": resolved_text_encoder_name or "",
        "text_encoder_name_1": resolved_text_encoder_name_1 or "",
        "text_encoder_name_2": resolved_text_encoder_name_2 or "",
        "vae_name": resolved_vae_label or "",
        "vae_source": resolved_vae_source,
    }
    return model, text_encoder, vae, model_info


class LLSSimpleCheckpointLoader:
    """
    统一封装的基础模型加载器。

    输出结构固定为 model / clip / vae / text_encoder。
    - `clip` 保留旧工作流需要的原生 CLIP 端口类型和位置。
    - `text_encoder` 是同一个原生 CLIP 对象的兼容别名。
    - 轻量级模型信息写入原生对象 `_lls_*` 属性，不再额外输出额外上下文端口。
    """

    CATEGORY = "LLS/Model Loader"
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP")
    RETURN_NAMES = ("model", "clip", "vae", "text_encoder")
    DESCRIPTION = (
        "Load SD1.5, SDXL, SDXL Turbo, and FLUX families with family-aware resource dispatch. "
        "Supports FLUX checkpoint-all-in-one mode and separated diffusion/text-encoder/VAE mode. "
        "Preserves the legacy CLIP output for old workflows and writes lightweight metadata "
        "into native MODEL/CLIP/VAE objects."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_model_name_choices(),),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
                "load_mode": (LOAD_MODE_CHOICES, {"default": "simple"}),
                "vae_source": (VAE_SOURCE_CHOICES, {"default": "auto"}),
                "text_encoder_source": (TEXT_ENCODER_SOURCE_CHOICES, {"default": "auto"}),
                "external_vae_name": (_get_vae_choices(), {"default": AUTO_PLACEHOLDER}),
                "external_text_encoder_1": (_get_text_encoder_choices(), {"default": AUTO_PLACEHOLDER}),
                "external_text_encoder_2": (_get_text_encoder_choices(), {"default": AUTO_PLACEHOLDER}),
            },
        }

    def load_checkpoint(
        self,
        ckpt_name: str,
        model_family: str,
        load_mode: str,
        vae_source: str,
        text_encoder_source: str,
        external_vae_name: str,
        external_text_encoder_1: str,
        external_text_encoder_2: str,
    ):
        model, text_encoder, vae, _model_info = _load_tagged_resources(
            model_name=ckpt_name,
            model_family=model_family,
            load_mode=load_mode,
            vae_source=vae_source,
            text_encoder_source=text_encoder_source,
            vae_name=external_vae_name,
            text_encoder_name_1=external_text_encoder_1,
            text_encoder_name_2=external_text_encoder_2,
        )
        return (model, text_encoder, vae, text_encoder)


class LLSUniversalModelLoader:
    CATEGORY = "LLS/Model Loader"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "text_encoder", "vae", "model_info")
    DESCRIPTION = (
        "Unified model loader for SD1.5, SDXL, and FLUX families. "
        "Internally resolves single or dual text encoders but always exposes one text_encoder output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_model_name_choices(),),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
                "load_mode": (LOAD_MODE_CHOICES, {"default": "simple"}),
                "vae_source": (UNIVERSAL_VAE_SOURCE_CHOICES, {"default": "auto"}),
                "text_encoder_source": (UNIVERSAL_TEXT_ENCODER_SOURCE_CHOICES, {"default": "auto"}),
                "text_encoder_1": (_get_text_encoder_choices(), {"default": AUTO_PLACEHOLDER}),
                "text_encoder_2": (_get_text_encoder_choices(), {"default": AUTO_PLACEHOLDER}),
                "vae_name": (_get_vae_choices(), {"default": AUTO_PLACEHOLDER}),
            },
        }

    def load(
        self,
        model_name: str,
        model_family: str,
        load_mode: str,
        vae_source: str,
        text_encoder_source: str,
        text_encoder_1: str,
        text_encoder_2: str,
        vae_name: str,
    ):
        model, text_encoder, vae, model_info = _load_tagged_resources(
            model_name=model_name,
            model_family=model_family,
            load_mode=load_mode,
            vae_source=vae_source,
            text_encoder_source=text_encoder_source,
            vae_name=vae_name,
            text_encoder_name_1=text_encoder_1,
            text_encoder_name_2=text_encoder_2,
        )
        return model, text_encoder, vae, info_to_json(model_info)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleCheckpointLoader": LLSSimpleCheckpointLoader,
    "LLSUniversalModelLoader": LLSUniversalModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleCheckpointLoader": "LLS Simple Checkpoint Loader",
    "LLSUniversalModelLoader": "LLS Universal Model Loader",
}
