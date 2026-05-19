from __future__ import annotations

from .backend_base import LoadedModels, UniversalBackendBase
from .request import LLSUniversalGenerationRequest


class SDXLBackend(UniversalBackendBase):
    family_name = "SDXL"

    def load_models(self, request: LLSUniversalGenerationRequest) -> LoadedModels:
        loaded = self._load_checkpoint_bundle(request)
        self._validate_loaded_family(loaded.model, "SDXL")
        return loaded

    def encode_prompt(self, request: LLSUniversalGenerationRequest, clip):
        positive = self._encode_sdxl_prompt(clip, request.positive_prompt, request.width, request.height)
        negative = self._encode_sdxl_prompt(clip, request.negative_prompt, request.width, request.height)
        return positive, negative

    def prepare_latent(self, request: LLSUniversalGenerationRequest, model):
        return self._prepare_empty_latent(request, model)

    def sample(self, request: LLSUniversalGenerationRequest, model, positive, negative, latent):
        return self._sample_latent(request, model, positive, negative, latent)

    def decode(self, request: LLSUniversalGenerationRequest, vae, samples):
        return self._decode_latent(vae, samples)

    def _encode_sdxl_prompt(self, clip, prompt: str, width: int, height: int):
        tokens = clip.tokenize(prompt)
        if isinstance(tokens, dict) and "g" in tokens and "l" in tokens:
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["g"]) < len(tokens["l"]):
                tokens["g"] += empty["g"]
            add_dict = {
                "width": width,
                "height": height,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": width,
                "target_height": height,
            }
            return self._encode_tokens(clip, tokens, add_dict=add_dict)
        return self._encode_standard_prompt(
            clip,
            prompt,
            add_dict={"width": width, "height": height},
        )
