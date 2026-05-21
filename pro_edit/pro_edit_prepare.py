from .backends.registry import resolve_backend
from .pro_edit_utils import (
    AUTO_RECOMMEND_CHOICES,
    BACKEND_MODE_CHOICES,
    CANVAS_FILL_CHOICES,
    EDIT_SCOPE_CHOICES,
    build_edit_info,
    build_workspace,
)


class LLSProImageEditPrepare:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_EDIT_INFO", "FLOAT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "edit_info", "recommended_denoise", "positive", "negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "edit_scope": (EDIT_SCOPE_CHOICES, {"default": "auto"}),
                "mask_grow": ("INT", {"default": 24, "min": 0, "max": 2048}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("INT", {"default": 64, "min": 0, "max": 512}),
                "crop_context_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "min_size": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (["fit", "pad", "stretch"], {"default": "fit"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "canvas_fill": (CANVAS_FILL_CHOICES, {"default": "edge"}),
                "auto_recommend": (AUTO_RECOMMEND_CHOICES, {"default": "enabled"}),
            },
            "optional": {
                "model": ("MODEL",),
                "model_info": ("STRING",),
            },
        }

    def prepare(
        self,
        image,
        mask,
        vae,
        positive,
        negative,
        backend_mode,
        edit_scope,
        mask_grow,
        mask_blur,
        mask_threshold,
        invert_mask,
        crop_context,
        crop_context_factor,
        min_size,
        max_size,
        resize_mode,
        expand_left,
        expand_right,
        expand_top,
        expand_bottom,
        canvas_fill,
        auto_recommend,
        model=None,
        model_info=None,
    ):
        workspace = build_workspace(
            image,
            mask,
            edit_scope=edit_scope,
            mask_grow=mask_grow,
            mask_blur=mask_blur,
            mask_threshold=mask_threshold,
            invert_mask=invert_mask,
            crop_context=crop_context,
            crop_context_factor=crop_context_factor,
            min_size=min_size,
            max_size=max_size,
            resize_mode=resize_mode,
            expand_left=expand_left,
            expand_right=expand_right,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            canvas_fill=canvas_fill,
            auto_recommend=auto_recommend,
        )
        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=model_info,
        )
        prepared = backend.prepare(
            model=model,
            vae=vae,
            work_image=workspace["work_image"],
            work_mask=workspace["work_mask"],
            positive=positive,
            negative=negative,
            workspace=workspace,
            routing=routing,
        )
        edit_info = build_edit_info(
            workspace,
            routing,
            backend_hints=prepared.get("backend_hints"),
        )
        return (
            prepared["latent"],
            workspace["work_image"],
            workspace["work_mask"],
            edit_info,
            float(workspace["recommended_denoise"]),
            prepared["positive"],
            prepared["negative"],
        )


NODE_CLASS_MAPPINGS = {"LLSProImageEditPrepare": LLSProImageEditPrepare}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProImageEditPrepare": "LLS Pro Image Edit Prepare"}
