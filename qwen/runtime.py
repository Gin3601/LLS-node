"""
Runtime wrappers for official ComfyUI Qwen image pipelines.
"""
from __future__ import annotations

import importlib
from typing import Any

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception as exc:
    comfy_core_nodes = None
    _CORE_NODES_ERR = exc
else:
    _CORE_NODES_ERR = None

try:
    nodes_sd3 = importlib.import_module("comfy_extras.nodes_sd3")
except Exception as exc:
    nodes_sd3 = None
    _SD3_ERR = exc
else:
    _SD3_ERR = None

try:
    nodes_model_advanced = importlib.import_module("comfy_extras.nodes_model_advanced")
except Exception as exc:
    nodes_model_advanced = None
    _MODEL_ADVANCED_ERR = exc
else:
    _MODEL_ADVANCED_ERR = None

try:
    nodes_qwen = importlib.import_module("comfy_extras.nodes_qwen")
except Exception as exc:
    nodes_qwen = None
    _QWEN_ERR = exc
else:
    _QWEN_ERR = None

try:
    nodes_flux = importlib.import_module("comfy_extras.nodes_flux")
except Exception as exc:
    nodes_flux = None
    _FLUX_ERR = exc
else:
    _FLUX_ERR = None

try:
    nodes_cfg = importlib.import_module("comfy_extras.nodes_cfg")
except Exception as exc:
    nodes_cfg = None
    _CFG_ERR = exc
else:
    _CFG_ERR = None

from . import discovery


_QWEN_CLIP_TYPE = "qwen_image"
_QWEN_MODEL_SHIFT = 3.1
_QWEN_CFG = 4.0
_QWEN_SAMPLER = "euler"
_QWEN_SCHEDULER = "simple"
_QWEN_CFG_NORM_STRENGTH = 1.0
_QWEN_REFERENCE_LATENTS_METHOD = "index_timestep_zero"


def _unwrap_first(result: Any):
    if hasattr(result, "result"):
        values = result.result
        if values is None:
            return None
        return values[0]
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, list):
        return result[0]
    return result


def _require_module(module: Any, label: str, import_error: Exception | None = None):
    if module is None:
        raise RuntimeError(
            f"[LLS] Required ComfyUI runtime component '{label}' is unavailable. "
            "Use a ComfyUI build that includes official Qwen image support."
        ) from import_error
    return module


def _require_class(module: Any, class_name: str, import_error: Exception | None = None):
    _require_module(module, class_name, import_error)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"[LLS] Required ComfyUI runtime component '{class_name}' is unavailable. "
            "Use a ComfyUI build that includes official Qwen image support."
        ) from import_error
    return cls


def _load_qwen_clip(clip_name: str):
    clip_loader_cls = _require_class(comfy_core_nodes, "CLIPLoader", _CORE_NODES_ERR)
    loader = clip_loader_cls()
    try:
        return loader.load_clip(clip_name, type=_QWEN_CLIP_TYPE, device="default")[0]
    except TypeError:
        return loader.load_clip(clip_name, type=_QWEN_CLIP_TYPE)[0]


def _load_qwen_vae(vae_name: str):
    vae_loader_cls = _require_class(comfy_core_nodes, "VAELoader", _CORE_NODES_ERR)
    return vae_loader_cls().load_vae(vae_name)[0]


def _load_qwen_model(model_name: str):
    unet_loader_cls = _require_class(comfy_core_nodes, "UNETLoader", _CORE_NODES_ERR)
    return unet_loader_cls().load_unet(model_name, "default")[0]


def _patch_qwen_model_sampling(model):
    patcher_cls = _require_class(nodes_model_advanced, "ModelSamplingAuraFlow", _MODEL_ADVANCED_ERR)
    return patcher_cls().patch_aura(model, _QWEN_MODEL_SHIFT)[0]


def _apply_cfg_norm(model):
    cfg_cls = _require_class(nodes_cfg, "CFGNorm", _CFG_ERR)
    return _unwrap_first(cfg_cls.execute(model, _QWEN_CFG_NORM_STRENGTH))


def _encode_clip_text(clip, prompt: str):
    encoder_cls = _require_class(comfy_core_nodes, "CLIPTextEncode", _CORE_NODES_ERR)
    return encoder_cls().encode(clip, prompt)[0]


def _create_empty_qwen_latent(width: int, height: int, batch_size: int):
    latent_cls = _require_class(nodes_sd3, "EmptySD3LatentImage", _SD3_ERR)
    latent_node = latent_cls()
    if hasattr(latent_node, "generate"):
        return latent_node.generate(width, height, batch_size)[0]
    return latent_node.execute(width, height, batch_size)[0]


def _encode_qwen_edit_conditioning(clip, prompt: str, vae, image):
    encoder_cls = _require_class(nodes_qwen, "TextEncodeQwenImageEditPlus", _QWEN_ERR)
    return _unwrap_first(encoder_cls.execute(clip, prompt, vae=vae, image1=image))


def _scale_qwen_edit_image(image):
    scaler_cls = _require_class(nodes_flux, "FluxKontextImageScale", _FLUX_ERR)
    return _unwrap_first(scaler_cls.execute(image))


def _apply_reference_latents_method(conditioning):
    method_cls = _require_class(nodes_flux, "FluxKontextMultiReferenceLatentMethod", _FLUX_ERR)
    return _unwrap_first(method_cls.execute(conditioning, _QWEN_REFERENCE_LATENTS_METHOD))


def _encode_image_to_latent(vae, image):
    encoder_cls = _require_class(comfy_core_nodes, "VAEEncode", _CORE_NODES_ERR)
    return encoder_cls().encode(vae, image)[0]


def _sample_qwen(model, positive, negative, latent, steps: int, seed: int):
    sampler_cls = _require_class(comfy_core_nodes, "KSampler", _CORE_NODES_ERR)
    return sampler_cls().sample(
        model,
        int(seed),
        int(steps),
        _QWEN_CFG,
        _QWEN_SAMPLER,
        _QWEN_SCHEDULER,
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]


def _decode_qwen_latent(vae, latent):
    decoder_cls = _require_class(comfy_core_nodes, "VAEDecode", _CORE_NODES_ERR)
    return decoder_cls().decode(vae, latent)[0]


def _resolve_qwen_resources(model_name: str, validator):
    validator(model_name)
    clip_name = discovery.resolve_qwen_text_encoder_name()
    vae_name = discovery.resolve_qwen_vae_name()

    if clip_name is None:
        raise RuntimeError(
            "[LLS] Missing Qwen text encoder. Place a compatible Qwen VL text encoder in "
            "ComfyUI/models/text_encoders/."
        )
    if vae_name is None:
        raise RuntimeError(
            "[LLS] Missing Qwen VAE. Place qwen_image_vae.safetensors in ComfyUI/models/vae/."
        )

    return model_name, clip_name, vae_name


def run_qwen_text_to_image(
    model_name: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    batch_size: int,
):
    try:
        model_name, clip_name, vae_name = _resolve_qwen_resources(
            model_name,
            discovery.validate_qwen_text_model_name,
        )
        model = _patch_qwen_model_sampling(_load_qwen_model(model_name))
        clip = _load_qwen_clip(clip_name)
        vae = _load_qwen_vae(vae_name)
        positive = _encode_clip_text(clip, prompt)
        negative = _encode_clip_text(clip, "")
        latent = _create_empty_qwen_latent(int(width), int(height), int(batch_size))
        sampled = _sample_qwen(model, positive, negative, latent, int(steps), int(seed))
        return _decode_qwen_latent(vae, sampled)
    except Exception as exc:
        if str(exc).startswith("[LLS]"):
            raise
        raise RuntimeError(f"[LLS] Qwen text-to-image failed: {exc}") from exc


def run_qwen_image_edit(
    model_name: str,
    image,
    prompt: str,
    steps: int,
    seed: int,
):
    try:
        if image is None:
            raise RuntimeError("[LLS] Missing source image. Connect an IMAGE input.")

        model_name, clip_name, vae_name = _resolve_qwen_resources(
            model_name,
            discovery.validate_qwen_edit_model_name,
        )
        model = _patch_qwen_model_sampling(_load_qwen_model(model_name))
        clip = _load_qwen_clip(clip_name)
        vae = _load_qwen_vae(vae_name)
        scaled = _scale_qwen_edit_image(image)
        positive = _encode_qwen_edit_conditioning(clip, prompt, vae, scaled)
        negative = _encode_qwen_edit_conditioning(clip, "", vae, scaled)
        positive = _apply_reference_latents_method(positive)
        negative = _apply_reference_latents_method(negative)
        latent = _encode_image_to_latent(vae, scaled)
        model = _apply_cfg_norm(model)
        sampled = _sample_qwen(model, positive, negative, latent, int(steps), int(seed))
        return _decode_qwen_latent(vae, sampled)
    except Exception as exc:
        if str(exc).startswith("[LLS]"):
            raise
        raise RuntimeError(f"[LLS] Qwen image edit failed: {exc}") from exc
