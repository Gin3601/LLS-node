from __future__ import annotations

from .backend_base import LoadedModels, UniversalBackendBase
from .request import LLSUniversalGenerationRequest


class SD15Backend(UniversalBackendBase):
    family_name = "SD1.5"

    def load_models(self, request: LLSUniversalGenerationRequest) -> LoadedModels:
        loaded = self._load_checkpoint_bundle(request)
        loaded_family = self._infer_loaded_family(loaded.model)
        if loaded_family in {"SDXL", "FLUX"}:
            raise RuntimeError(
                f"[LLS] Checkpoint '{request.model_name}' looks like {loaded_family}, not SD1.5."
            )
        return loaded

    def encode_prompt(self, request: LLSUniversalGenerationRequest, clip):
        positive = self._encode_standard_prompt(clip, request.positive_prompt)
        negative = self._encode_standard_prompt(clip, request.negative_prompt)
        return positive, negative

    def prepare_latent(self, request: LLSUniversalGenerationRequest, model):
        return self._prepare_empty_latent(request, model)

    def sample(self, request: LLSUniversalGenerationRequest, model, positive, negative, latent):
        return self._sample_latent(request, model, positive, negative, latent)

    def decode(self, request: LLSUniversalGenerationRequest, vae, samples):
        return self._decode_latent(vae, samples)
