from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import folder_paths
except Exception as exc:
    folder_paths = None
    _FOLDER_PATHS_IMPORT_ERROR = exc
else:
    _FOLDER_PATHS_IMPORT_ERROR = None

try:
    import comfy.model_management as model_management
except Exception as exc:
    model_management = None
    _MODEL_MANAGEMENT_IMPORT_ERROR = exc
else:
    _MODEL_MANAGEMENT_IMPORT_ERROR = None

try:
    import comfy.sd as comfy_sd
except Exception as exc:
    comfy_sd = None
    _COMFY_SD_IMPORT_ERROR = exc
else:
    _COMFY_SD_IMPORT_ERROR = None

from ..sampling.nodes import _common_ksampler
from .request import LLSUniversalGenerationRequest


@dataclass(frozen=True, slots=True)
class LoadedModels:
    model: Any
    clip: Any
    vae: Any


class UniversalBackendBase(ABC):
    """统一 backend 接口。"""

    family_name = "BASE"

    @abstractmethod
    def load_models(self, request: LLSUniversalGenerationRequest) -> LoadedModels:
        raise NotImplementedError

    @abstractmethod
    def encode_prompt(self, request: LLSUniversalGenerationRequest, clip: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_latent(self, request: LLSUniversalGenerationRequest, model: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        request: LLSUniversalGenerationRequest,
        model: Any,
        positive: Any,
        negative: Any,
        latent: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, request: LLSUniversalGenerationRequest, vae: Any, samples: dict[str, Any]) -> Any:
        raise NotImplementedError

    def generate(self, request: LLSUniversalGenerationRequest) -> Any:
        request = request.validate()
        loaded = self.load_models(request)
        positive, negative = self.encode_prompt(request, loaded.clip)
        latent = self.prepare_latent(request, loaded.model)
        sampled = self.sample(request, loaded.model, positive, negative, latent)
        return self.decode(request, loaded.vae, sampled)

    # ---------- 共用辅助 ----------

    def _require_torch(self) -> None:
        if torch is None:
            raise RuntimeError(
                "[LLS] PyTorch is not available. "
                "Run this node inside ComfyUI's Python environment."
            ) from _TORCH_IMPORT_ERROR

    def _require_comfy_runtime(self) -> None:
        if folder_paths is None:
            raise RuntimeError(
                "[LLS] folder_paths is not available. "
                "Place this plugin under ComfyUI/custom_nodes/ and restart ComfyUI."
            ) from _FOLDER_PATHS_IMPORT_ERROR
        if comfy_sd is None:
            raise RuntimeError(
                "[LLS] comfy.sd is not available. "
                "Run this node inside a ComfyUI environment."
            ) from _COMFY_SD_IMPORT_ERROR
        if model_management is None:
            raise RuntimeError(
                "[LLS] comfy.model_management is not available. "
                "Run this node inside a ComfyUI environment."
            ) from _MODEL_MANAGEMENT_IMPORT_ERROR

    def _resolve_checkpoint_path(self, model_name: str) -> str:
        self._require_comfy_runtime()
        try:
            return folder_paths.get_full_path_or_raise("checkpoints", model_name)
        except (AttributeError, TypeError):
            ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
            if ckpt_path:
                return ckpt_path
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"[LLS] Checkpoint '{model_name}' was not found in ComfyUI checkpoints."
            ) from exc
        raise RuntimeError(
            f"[LLS] Checkpoint '{model_name}' was not found in ComfyUI checkpoints."
        )

    def _load_checkpoint_bundle(self, request: LLSUniversalGenerationRequest) -> LoadedModels:
        ckpt_path = self._resolve_checkpoint_path(request.model_name)
        try:
            embedding_directory = folder_paths.get_folder_paths("embeddings")
        except Exception:
            embedding_directory = []

        try:
            model, clip, vae = comfy_sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=embedding_directory,
            )[:3]
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] Failed to load checkpoint '{request.model_name}' for {self.family_name}: {exc}"
            ) from exc

        if model is None or clip is None or vae is None:
            raise RuntimeError(
                f"[LLS] Checkpoint '{request.model_name}' did not produce MODEL/CLIP/VAE required by {self.family_name}."
            )

        return LoadedModels(model=model, clip=clip, vae=vae)

    def _infer_loaded_family(self, model: Any) -> str:
        model_impl = getattr(model, "model", model)
        latent_format = getattr(model_impl, "latent_format", None)
        latent_name = type(latent_format).__name__
        config_name = type(getattr(model_impl, "model_config", None)).__name__
        combined = f"{latent_name} {config_name}".lower()

        if "flux" in combined:
            return "FLUX"
        if "sdxl" in combined:
            return "SDXL"
        if "sd15" in combined:
            return "SD1.5"
        return "UNKNOWN"

    def _validate_loaded_family(self, model: Any, expected: str) -> None:
        loaded_family = self._infer_loaded_family(model)
        if loaded_family == "UNKNOWN":
            return
        if loaded_family != expected:
            raise RuntimeError(
                f"[LLS] Selected backend '{expected}' does not match loaded checkpoint family '{loaded_family}'."
            )

    def _get_latent_spec(self, model: Any) -> tuple[int, int]:
        model_impl = getattr(model, "model", model)
        latent_format = getattr(model_impl, "latent_format", None)
        channels = int(getattr(latent_format, "latent_channels", 4))
        downscale = int(getattr(latent_format, "spacial_downscale_ratio", 8))
        return channels, downscale

    def _prepare_empty_latent(
        self,
        request: LLSUniversalGenerationRequest,
        model: Any,
    ) -> dict[str, Any]:
        self._require_torch()
        self._require_comfy_runtime()

        latent_channels, downscale = self._get_latent_spec(model)
        aligned_request, changed = request.align_to(downscale)
        if changed:
            print(
                f"[LLS] {self.family_name}: adjusted size "
                f"{request.width}x{request.height} -> {aligned_request.width}x{aligned_request.height} "
                f"to match latent downscale ratio {downscale}."
            )

        latent = torch.zeros(
            [1, latent_channels, aligned_request.height // downscale, aligned_request.width // downscale],
            device=model_management.intermediate_device(),
            dtype=model_management.intermediate_dtype(),
        )
        return {"samples": latent, "downscale_ratio_spacial": downscale}

    def _sample_latent(
        self,
        request: LLSUniversalGenerationRequest,
        model: Any,
        positive: Any,
        negative: Any,
        latent: dict[str, Any],
    ) -> dict[str, Any]:
        actual_seed = request.resolve_seed()
        try:
            return _common_ksampler(
                model=model,
                seed=actual_seed,
                steps=request.steps,
                cfg=request.cfg,
                sampler_name=request.sampler_name,
                scheduler=request.scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=request.denoise,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] {self.family_name} sampling failed for checkpoint '{request.model_name}': {exc}"
            ) from exc

    def _decode_latent(self, vae: Any, samples: dict[str, Any]) -> Any:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(getattr(images, "shape", ())) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def _encode_standard_prompt(self, clip: Any, prompt: str, add_dict: dict[str, Any] | None = None) -> Any:
        tokens = clip.tokenize(prompt)
        return clip.encode_from_tokens_scheduled(tokens, add_dict=add_dict or {})

