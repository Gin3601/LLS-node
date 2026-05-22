from __future__ import annotations

import importlib

from ..sampling.nodes import _common_ksampler, _get_samplers, _get_schedulers

try:
    comfy_core_nodes = importlib.import_module("nodes")
except Exception:
    comfy_core_nodes = None


def _run_native_ksampler_advanced(
    model,
    add_noise,
    noise_seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    start_at_step,
    end_at_step,
    return_with_leftover_noise,
    denoise,
):
    if comfy_core_nodes is None or not hasattr(comfy_core_nodes, "KSamplerAdvanced"):
        return None

    sampler = comfy_core_nodes.KSamplerAdvanced()
    result = sampler.sample(
        model=model,
        add_noise=add_noise,
        noise_seed=int(noise_seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        start_at_step=int(start_at_step),
        end_at_step=int(end_at_step),
        return_with_leftover_noise=return_with_leftover_noise,
        denoise=float(denoise),
    )
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, list):
        return result[0]
    return result


class LLSProKSamplerBridge:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"advanced": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_get_samplers(),),
                "scheduler": (_get_schedulers(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "advanced": True}),
                "return_with_leftover_noise": (["disable", "enable"], {"advanced": True}),
            }
        }

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        result_latent = _run_native_ksampler_advanced(
            model=model,
            add_noise=add_noise,
            noise_seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            denoise=denoise,
        )
        if result_latent is None:
            force_full_denoise = str(return_with_leftover_noise or "disable") != "enable"
            disable_noise = str(add_noise or "enable") == "disable"

            result_latent = _common_ksampler(
                model=model,
                seed=int(noise_seed),
                steps=int(steps),
                cfg=float(cfg),
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent_image,
                denoise=float(denoise),
                disable_noise=disable_noise,
                start_step=int(start_at_step),
                last_step=int(end_at_step),
                force_full_denoise=force_full_denoise,
            )
        return (result_latent,)


NODE_CLASS_MAPPINGS = {"LLSProKSamplerBridge": LLSProKSamplerBridge}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProKSamplerBridge": "LLS Pro KSampler Bridge"}
