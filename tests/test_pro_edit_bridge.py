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

    def test_bridge_prefers_native_advanced_ksampler_when_available(self):
        model = FakeModel()
        positive = make_conditioning("positive")
        negative = make_conditioning("negative")
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "pro_edit_prepare_region"}
        captured = {}
        native_result = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "native"}

        class NativeKSamplerAdvanced:
            def sample(self, **kwargs):
                captured.update(kwargs)
                return (native_result,)

        with mock.patch.object(self.bridge_module, "comfy_core_nodes", mock.Mock(KSamplerAdvanced=NativeKSamplerAdvanced)), mock.patch.object(
            self.bridge_module,
            "_common_ksampler",
            side_effect=AssertionError("_common_ksampler should not be used when native KSamplerAdvanced is available"),
        ):
            (result_latent,) = self.node.sample(
                model=model,
                add_noise="disable",
                noise_seed=11,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=positive,
                negative=negative,
                latent_image=latent,
                start_at_step=3,
                end_at_step=17,
                return_with_leftover_noise="enable",
            )

        self.assertIs(result_latent, native_result)
        self.assertIs(captured["model"], model)
        self.assertIs(captured["positive"], positive)
        self.assertIs(captured["negative"], negative)
        self.assertIs(captured["latent_image"], latent)
        self.assertEqual(captured["noise_seed"], 11)
        self.assertEqual(captured["steps"], 20)
        self.assertEqual(captured["cfg"], 7.0)
        self.assertEqual(captured["sampler_name"], "euler")
        self.assertEqual(captured["scheduler"], "normal")
        self.assertEqual(captured["denoise"], 1.0)
        self.assertEqual(captured["add_noise"], "disable")
        self.assertEqual(captured["start_at_step"], 3)
        self.assertEqual(captured["end_at_step"], 17)
        self.assertEqual(captured["return_with_leftover_noise"], "enable")

    def test_bridge_falls_back_to_common_ksampler_without_native_node(self):
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64))}
        captured = {}

        def fake_common_ksampler(**kwargs):
            captured.update(kwargs)
            return kwargs["latent"]

        with mock.patch.object(self.bridge_module, "comfy_core_nodes", None), mock.patch.object(
            self.bridge_module,
            "_common_ksampler",
            side_effect=fake_common_ksampler,
        ):
            (_result_latent,) = self.node.sample(
                model=FakeModel(),
                add_noise="enable",
                noise_seed=21,
                steps=10,
                cfg=5.5,
                sampler_name="heun",
                scheduler="karras",
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="disable",
            )

        self.assertFalse(captured["disable_noise"])
        self.assertTrue(captured["force_full_denoise"])


if __name__ == "__main__":
    unittest.main()
