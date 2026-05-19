"""
LLS / ModelLoader
=================
功能域：模型加载与管理（对应功能分类总览第 1 节）

CATEGORY = "LLS/Model Loader"
"""
from __future__ import annotations

import importlib

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
    MODEL_FAMILY_CHOICES,
    build_model_info,
    canonicalize_family,
    family_requires_guidance,
    get_latent_spec,
    infer_family_from_name,
    is_flux_family,
    is_sdxl_family,
)


AUTO_PLACEHOLDER = "(auto)"
NO_MODELS_PLACEHOLDER = "(no models found)"
NO_VAE_PLACEHOLDER = "(no vae found)"
NO_TEXT_ENCODER_PLACEHOLDER = "(no text encoders found)"
LOAD_MODE_CHOICES = ["simple", "advanced"]
VAE_SOURCE_CHOICES = ["auto", "embedded", "external", "none"]
TEXT_ENCODER_SOURCE_CHOICES = ["auto", "embedded", "external", "manual"]


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
                names = list(vae_loader.vae_list(vae_loader))
                return _with_placeholder(names, NO_VAE_PLACEHOLDER)
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
            "[LLS] No models were found in ComfyUI models/checkpoints or models/diffusion_models."
        )

    if model_name.startswith("diffusion_models/"):
        if not is_flux_family(family):
            raise RuntimeError(
                f"[LLS] {family} expects a checkpoint file. Select a checkpoint from models/checkpoints/."
            )
        short_name = model_name.split("/", 1)[1]
        model_path = _get_full_path("diffusion_models", short_name)
        if model_path:
            return ("diffusion_models", model_path)

    checkpoint_path = _get_full_path("checkpoints", model_name)
    if checkpoint_path:
        return ("checkpoints", checkpoint_path)

    if is_flux_family(family):
        diffusion_path = _get_full_path("diffusion_models", model_name)
        if diffusion_path:
            return ("diffusion_models", diffusion_path)

    raise RuntimeError(
        f"[LLS] Model '{model_name}' was not found in the expected ComfyUI model directories."
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
        loaded = _load_checkpoint_bundle(model_path)
        return loaded[0], loaded[1], loaded[2]

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


def _load_external_text_encoder(family: str, name1: str | None, name2: str | None):
    if comfy_core_nodes is None:
        raise RuntimeError(
            "[LLS] ComfyUI core CLIP loaders are unavailable, so external text encoder loading cannot be used."
        ) from _CORE_NODES_ERR

    if family == "SD15":
        if not name1:
            raise RuntimeError(
                "[LLS] SD15 external text encoder loading requires external_text_encoder_1."
            )
        clip_loader_cls = getattr(comfy_core_nodes, "CLIPLoader", None)
        if clip_loader_cls is None:
            raise RuntimeError("[LLS] ComfyUI core CLIPLoader class was not found.")
        return clip_loader_cls().load_clip(name1, type="stable_diffusion")[0]

    if is_sdxl_family(family):
        if not name1 or not name2:
            raise RuntimeError(
                "[LLS] SDXL external text encoder loading requires clip_l and clip_g via "
                "external_text_encoder_1/external_text_encoder_2."
            )
        dual_loader_cls = getattr(comfy_core_nodes, "DualCLIPLoader", None)
        if dual_loader_cls is None:
            raise RuntimeError("[LLS] ComfyUI core DualCLIPLoader class was not found.")
        return dual_loader_cls().load_clip(name1, name2, type="sdxl")[0]

    if not name1 or not name2:
        raise RuntimeError(
            "[LLS] FLUX text encoder loading requires clip_l and t5xxl via "
            "external_text_encoder_1/external_text_encoder_2."
        )
    dual_loader_cls = getattr(comfy_core_nodes, "DualCLIPLoader", None)
    if dual_loader_cls is None:
        raise RuntimeError("[LLS] ComfyUI core DualCLIPLoader class was not found.")
    return dual_loader_cls().load_clip(name1, name2, type="flux")[0]


def _resolve_text_encoder(
    family: str,
    source: str,
    embedded_text_encoder,
    external_name_1: str | None,
    external_name_2: str | None,
) -> tuple[object | None, str]:
    if source == "manual":
        return None, "manual"

    if source == "embedded":
        if embedded_text_encoder is None:
            raise RuntimeError(
                f"[LLS] {family} was set to use embedded text encoders, but the selected model did not provide them."
            )
        return embedded_text_encoder, "embedded"

    if source == "external":
        return _load_external_text_encoder(family, external_name_1, external_name_2), "external"

    if embedded_text_encoder is not None:
        return embedded_text_encoder, "embedded"
    if external_name_1:
        return _load_external_text_encoder(family, external_name_1, external_name_2), "external"

    if is_flux_family(family):
        raise RuntimeError(
            "[LLS] FLUX models need text encoders. Provide clip_l and t5xxl via "
            "external_text_encoder_1/external_text_encoder_2, or use a checkpoint that embeds them."
        )

    raise RuntimeError(
        f"[LLS] {family} did not expose an embedded text encoder. "
        "Switch text_encoder_source to 'external' and choose the required encoder files."
    )


def _resolve_vae(
    family: str,
    source: str,
    embedded_vae,
    external_vae_name: str | None,
) -> tuple[object | None, str]:
    if source == "none":
        if is_flux_family(family):
            raise RuntimeError(
                "[LLS] FLUX generation requires a VAE/AE. Provide external_vae_name or use a model with an embedded VAE."
            )
        return None, "none"

    if source == "embedded":
        if embedded_vae is None:
            raise RuntimeError(
                f"[LLS] {family} was set to use an embedded VAE, but the selected model did not provide one."
            )
        return embedded_vae, "embedded"

    if source == "external":
        if not external_vae_name:
            raise RuntimeError("[LLS] External VAE loading requires external_vae_name.")
        return _load_external_vae(external_vae_name), "external"

    if embedded_vae is not None:
        return embedded_vae, "embedded"
    if external_vae_name:
        return _load_external_vae(external_vae_name), "external"
    if is_flux_family(family):
        raise RuntimeError(
            "[LLS] FLUX models usually need an external AE/VAE. Provide external_vae_name or use a model with an embedded VAE."
        )
    return None, "none"


def _text_encoder_type_for_family(family: str) -> str:
    if family == "SD15":
        return "clip"
    if is_sdxl_family(family):
        return "sdxl_dual_clip"
    return "flux_clip_l_t5xxl"


def _required_text_encoders_for_family(family: str) -> list[str]:
    if family == "SD15":
        return ["clip"]
    if is_sdxl_family(family):
        return ["clip_l", "clip_g"]
    return ["clip_l", "t5xxl"]


def _required_vae_for_family(family: str) -> str:
    return "required" if is_flux_family(family) else "optional"


class LLSSimpleCheckpointLoader:
    """
    统一外观的模型加载节点。

    对外仍然输出 MODEL / CLIP / VAE / STRING 以兼容旧工作流；
    但第二个端口实际语义已经升级为 text_encoder，资源分派策略则由
    model_family + source 选项控制。
    """

    CATEGORY = "LLS/Model Loader"
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "text_encoder", "vae", "model_info")
    DESCRIPTION = (
        "Load SD15, SDXL, SDXL Turbo, and FLUX families with family-specific resource "
        "dispatch. The CLIP-typed output now represents the resolved text encoder bundle."
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
            }
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
        if folder_paths is None:
            raise RuntimeError(
                "[LLS] folder_paths is not available. Make sure this node runs inside ComfyUI."
            ) from _FOLDER_PATHS_ERR
        if comfy_sd is None:
            raise RuntimeError(
                "[LLS] comfy.sd is not available. Make sure this node runs inside ComfyUI."
            ) from _COMFY_SD_ERR

        family = canonicalize_family(model_family)
        if model_family == "Auto":
            family = infer_family_from_name(ckpt_name, family)

        model_source, model_path = _resolve_model_path(ckpt_name, family)
        model, embedded_text_encoder, embedded_vae = _load_model(model_source, model_path)

        if model is None:
            raise RuntimeError(f"[LLS] The selected model '{ckpt_name}' did not produce a valid MODEL.")

        external_vae_name = _normalize_choice(external_vae_name, NO_VAE_PLACEHOLDER)
        external_text_encoder_1 = _normalize_choice(external_text_encoder_1, NO_TEXT_ENCODER_PLACEHOLDER)
        external_text_encoder_2 = _normalize_choice(external_text_encoder_2, NO_TEXT_ENCODER_PLACEHOLDER)

        text_encoder, resolved_text_encoder_source = _resolve_text_encoder(
            family=family,
            source=text_encoder_source,
            embedded_text_encoder=embedded_text_encoder,
            external_name_1=external_text_encoder_1,
            external_name_2=external_text_encoder_2,
        )
        vae, resolved_vae_source = _resolve_vae(
            family=family,
            source=vae_source,
            embedded_vae=embedded_vae,
            external_vae_name=external_vae_name,
        )

        latent_spec = get_latent_spec({"family": family})
        model_info = build_model_info(
            family=family,
            model_name=ckpt_name,
            model_source=model_source,
            load_mode=load_mode,
            text_encoder_source=resolved_text_encoder_source,
            vae_source=resolved_vae_source,
            text_encoder_type=_text_encoder_type_for_family(family),
            has_embedded_vae=embedded_vae is not None,
            has_embedded_text_encoder=embedded_text_encoder is not None,
            required_text_encoders=_required_text_encoders_for_family(family),
            required_vae=_required_vae_for_family(family),
            is_turbo=(family == "SDXL_TURBO"),
            guidance_embed=family_requires_guidance(family),
            latent_channels=latent_spec["latent_channels"],
            downscale_ratio=latent_spec["downscale_ratio"],
        )

        return (model, text_encoder, vae, model_info)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleCheckpointLoader": LLSSimpleCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleCheckpointLoader": "LLS Simple Checkpoint Loader",
}
