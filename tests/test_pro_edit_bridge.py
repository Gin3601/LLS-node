import json
import unittest
from unittest import mock

try:
    from .test_pro_edit_helpers import FakeLatentTensor, FakeModel, import_plugin_submodule, load_plugin_package, make_conditioning
except ImportError:
    from test_pro_edit_helpers import FakeLatentTensor, FakeModel, import_plugin_submodule, load_plugin_package, make_conditioning


class TestProEditBridge(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.bridge_module = import_plugin_submodule(self.plugin, "pro_edit.pro_edit_bridge")
        self.node = self.bridge_module.LLSProKSamplerBridge()

    def test_bridge_uses_auto_from_edit_denoise(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_inpaint",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "pro_edit_prepare_region"}

        with mock.patch.object(self.bridge_module, "_common_ksampler", side_effect=lambda **kwargs: kwargs["latent"]):
            result_latent, sample_info_json = self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.25,
                denoise_mode="auto_from_edit",
                flux_guidance=3.5,
                model_family="Auto",
                edit_info={
                    "backend_name": "sdxl",
                    "model_family": "SDXL",
                    "model_role": "inpaint",
                    "supports_inpaint_native": True,
                    "supports_image_edit_native": False,
                    "preferred_edit_backend": "sdxl",
                    "profile_id": "sdxl_inpaint",
                    "backend_type": "sdxl_native",
                    "sampler_strategy": "standard_k",
                    "recommended_denoise": 0.63,
                },
                model_info=None,
            )

        sample_info = json.loads(sample_info_json)
        self.assertEqual(result_latent["source"], "pro_edit_prepare_region")
        self.assertEqual(sample_info["backend_name"], "sdxl")
        self.assertEqual(sample_info["profile_id"], "sdxl_inpaint")
        self.assertEqual(sample_info["sampler_strategy"], "standard_k")
        self.assertEqual(sample_info["denoise"], 0.63)
        self.assertEqual(sample_info["denoise_mode"], "auto_from_edit")

    def test_bridge_applies_flux_guidance_for_flux_backend(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="flux_guided",
            loader_strategy="flux_split_or_bundle",
        )
        latent = {"samples": FakeLatentTensor((1, 16, 64, 64)), "source": "pro_edit_prepare_region"}

        with mock.patch.object(self.bridge_module, "_common_ksampler", side_effect=lambda **kwargs: kwargs["latent"]):
            _, sample_info_json = self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=9,
                steps=12,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.8,
                denoise_mode="manual",
                flux_guidance=4.2,
                model_family="Auto",
                edit_info={
                    "backend_name": "flux",
                    "model_family": "FLUX_DEV",
                    "model_role": "edit",
                    "supports_inpaint_native": False,
                    "supports_image_edit_native": True,
                    "preferred_edit_backend": "flux",
                    "profile_id": "flux_edit",
                    "backend_type": "flux_edit",
                    "sampler_strategy": "flux_guided",
                    "recommended_denoise": 0.8,
                },
                model_info=None,
            )

        sample_info = json.loads(sample_info_json)
        self.assertEqual(sample_info["backend_name"], "flux")
        self.assertEqual(sample_info["profile_id"], "flux_edit")
        self.assertEqual(sample_info["sampler_strategy"], "flux_guided")
        self.assertEqual(sample_info["guidance"], 4.2)

    def test_bridge_manual_backend_mismatch_raises(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_inpaint",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "pro_edit_prepare_region"}

        with self.assertRaisesRegex(RuntimeError, "backend 'flux' is incompatible"):
            self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="flux",
                quality_preset="Manual",
                seed=11,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                denoise_mode="manual",
                flux_guidance=3.5,
                model_family="Auto",
                edit_info=None,
                model_info=None,
            )

    def test_bridge_rejects_unknown_sampler_strategy(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="mystery_strategy",
            loader_strategy="flux_split_or_bundle",
        )
        latent = {"samples": FakeLatentTensor((1, 16, 64, 64)), "source": "pro_edit_prepare_region"}

        with self.assertRaisesRegex(RuntimeError, "Unsupported sampler_strategy 'mystery_strategy'"):
            self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=9,
                steps=12,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.8,
                denoise_mode="manual",
                flux_guidance=4.2,
                model_family="Auto",
                edit_info=None,
                model_info=None,
            )


if __name__ == "__main__":
    unittest.main()
