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

from ..utils.model_info import (
    LLS_TEXT_ENCODER_TYPE,
    MODEL_FAMILY_CHOICES,
    build_model_info,
    canonicalize_family,
    family_requires_guidance,
    get_family_defaults,
    get_latent_spec,
    infer_family_from_name,
    is_flux_family,
    is_sdxl_family,
)
from ..utils.task_context import LLS_TASK_CONTEXT_TYPE, merge_task_context, parse_task_context, update_task_context


AUTO_PLACEHOLDER = "(auto)"
NO_MODELS_PLACEHOLDER = "(no models found)"
NO_VAE_PLACEHOLDER = "(no vae found)"
NO_TEXT_ENCODER_PLACEHOLDER = "(no text encoders found)"
LOAD_MODE_CHOICES = ["simple", "advanced"]
VAE_SOURCE_CHOICES = ["auto", "embedded", "external", "none"]
TEXT_ENCODER_SOURCE_CHOICES = ["auto", "embedded", "external", "manual"]

_FLUX_CLIP_L_PATTERNS = ("clip_l.safetensors", "clip_l", "clip-l")
_FLUX_T5_PATTERNS = ("t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors", "t5xxl", "t5")
_SDXL_CLIP_L_PATTERNS = ("clip_l.safetensors", "clip_l", "clip-l")
_SDXL_CLIP_G_PATTERNS = ("clip_g.safetensors", "clip_g", "clip-g")
_FLUX_VAE_PATTERNS = ("ae.safetensors", "ae", "vae")


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

    if is_flux_family(family):
        raise RuntimeError(
            "[LLS] Missing FLUX AE/VAE ae.safetensors. Place it in ComfyUI/models/vae/."
        )
    return None, "none", None


class LLSSimpleCheckpointLoader:
    """
    统一封装的基础模型加载器。

    真实输出结构为 model / text_encoder / vae / task_context。
    text_encoder 端口类型使用 LLS_TEXT_ENCODER，内部承载 ComfyUI 的真实文本编码对象。
    """

    CATEGORY = "LLS/Model Loader"
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", LLS_TEXT_ENCODER_TYPE, "VAE", LLS_TASK_CONTEXT_TYPE)
    RETURN_NAMES = ("model", "text_encoder", "vae", "task_context")
    DESCRIPTION = (
        "Load SD1.5, SDXL, SDXL Turbo, and FLUX families with family-aware resource dispatch. "
        "Supports FLUX checkpoint-all-in-one mode and separated diffusion/text-encoder/VAE mode. "
        "Writes the resolved loader state back into a unified task_context object."
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
            "optional": {
                "task_context": (LLS_TASK_CONTEXT_TYPE,),
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
        task_context=None,
    ):
        if folder_paths is None:
            raise RuntimeError(
                "[LLS] folder_paths is not available. Make sure this node runs inside ComfyUI."
            ) from _FOLDER_PATHS_ERR
        if comfy_sd is None:
            raise RuntimeError(
                "[LLS] comfy.sd is not available. Make sure this node runs inside ComfyUI."
            ) from _COMFY_SD_ERR

        context = parse_task_context(task_context)
        requested_family = str(context.get("model_family") or "auto")
        if model_family != "Auto":
            requested_family = model_family
        if model_family == "Auto":
            fallback_family = context.get("resolved_model_family") or context.get("family") or context.get("model_family") or "SD1.5"
            family = infer_family_from_name(ckpt_name, fallback_family)
        else:
            family = canonicalize_family(requested_family)

        if vae_source == "auto" and context.get("use_external_vae"):
            vae_source = "external"
        if text_encoder_source == "auto" and context.get("use_external_text_encoder"):
            text_encoder_source = "external"

        model_source, model_path = _resolve_model_path(ckpt_name, family)
        model, embedded_text_encoder, embedded_vae = _load_model(model_source, model_path)
        if model is None:
            raise RuntimeError(f"[LLS] The selected model '{ckpt_name}' did not produce a valid diffusion model.")

        external_vae_name = _normalize_choice(external_vae_name, NO_VAE_PLACEHOLDER)
        external_text_encoder_1 = _normalize_choice(external_text_encoder_1, NO_TEXT_ENCODER_PLACEHOLDER)
        external_text_encoder_2 = _normalize_choice(external_text_encoder_2, NO_TEXT_ENCODER_PLACEHOLDER)

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
        latent_spec = get_latent_spec(defaults)
        model_info = build_model_info(
            family=family,
            checkpoint_name=ckpt_name,
            model_name=ckpt_name,
            load_mode=load_mode,
            vae_source=resolved_vae_source,
            text_encoder_source=resolved_text_encoder_source,
            text_encoder_type=defaults["text_encoder_type"],
            has_embedded_vae=embedded_vae is not None,
            has_embedded_text_encoder=embedded_text_encoder is not None,
            required_text_encoders=defaults["required_text_encoders"],
            required_vae=defaults["required_vae"],
            is_turbo=defaults["is_turbo"],
            is_flux=defaults["is_flux"],
            default_width=defaults["default_width"],
            default_height=defaults["default_height"],
            default_steps=defaults["default_steps"],
            default_cfg=defaults["default_cfg"],
            default_guidance=defaults["default_guidance"],
            default_sampler=defaults["default_sampler"],
            default_scheduler=defaults["default_scheduler"],
            default_denoise=defaults["default_denoise"],
            guidance_embed=family_requires_guidance(family),
            latent_channels=latent_spec["latent_channels"],
            downscale_ratio=latent_spec["downscale_ratio"],
            model_source=model_source,
            vae_name=resolved_vae_name or ("embedded" if embedded_vae is not None else None),
            text_encoder_name=resolved_te_1,
            text_encoder_name_1=resolved_te_1,
            text_encoder_name_2=resolved_te_2,
        )

        model_caps = {
            "supports_img2img": vae is not None,
            "supports_inpaint": vae is not None,
            "supports_controlnet": family in {"SD1.5", "SDXL", "SDXL_TURBO"},
            "supports_clip_skip": not is_flux_family(family),
            "supports_flux_guidance": is_flux_family(family),
            "vae_type": "ae" if is_flux_family(family) else "vae",
            "text_encoder_type": model_info["text_encoder_type"],
            "latent_channels": latent_spec["latent_channels"],
            "downscale_ratio": latent_spec["downscale_ratio"],
            "recommended_width": model_info["default_width"],
            "recommended_height": model_info["default_height"],
            "recommended_steps": model_info["default_steps"],
            "recommended_cfg": model_info["default_cfg"],
            "recommended_denoise": model_info["default_denoise"],
            "recommended_guidance": model_info["default_guidance"],
            "recommended_sampler": model_info["default_sampler"],
            "recommended_scheduler": model_info["default_scheduler"],
            "source": "LLS Simple Checkpoint Loader",
        }
        next_context = merge_task_context(context, model_info)
        next_context = update_task_context(
            next_context,
            model_family=requested_family,
            resolved_model_family=family,
            checkpoint_name=ckpt_name,
            model_name=ckpt_name,
            load_mode=load_mode,
            vae_source=resolved_vae_source,
            text_encoder_source=resolved_text_encoder_source,
            vae_name=resolved_vae_name or ("embedded" if embedded_vae is not None else None),
            text_encoder_name=resolved_te_1,
            text_encoder_name_1=resolved_te_1,
            text_encoder_name_2=resolved_te_2,
            use_external_vae=resolved_vae_source == "external",
            use_external_text_encoder=resolved_text_encoder_source == "external",
            **model_caps,
        )

        return (model, text_encoder, vae, next_context)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleCheckpointLoader": LLSSimpleCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleCheckpointLoader": "LLS Simple Checkpoint Loader",
}
