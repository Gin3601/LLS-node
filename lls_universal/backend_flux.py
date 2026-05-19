from __future__ import annotations

from .backend_base import LoadedModels, UniversalBackendBase
from .request import LLSUniversalGenerationRequest


class FluxBackend(UniversalBackendBase):
    family_name = "FLUX"

    def load_models(self, request: LLSUniversalGenerationRequest) -> LoadedModels:
        loaded = self._load_checkpoint_bundle(request)
        self._validate_loaded_family(loaded.model, "FLUX")
        return loaded

    def encode_prompt(self, request: LLSUniversalGenerationRequest, clip):
        positive = self._encode_flux_prompt(clip, request.positive_prompt, request.cfg)
        negative = self._encode_flux_prompt(clip, request.negative_prompt, request.cfg)
        return positive, negative

    def prepare_latent(self, request: LLSUniversalGenerationRequest, model):
        return self._prepare_empty_latent(request, model)

    def sample(self, request: LLSUniversalGenerationRequest, model, positive, negative, latent):
        return self._sample_latent(request, model, positive, negative, latent)

    def decode(self, request: LLSUniversalGenerationRequest, vae, samples):
        return self._decode_latent(vae, samples)

    def _encode_flux_prompt(self, clip, prompt: str, guidance: float):
        tokens = clip.tokenize(prompt)
        return clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
