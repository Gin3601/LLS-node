import json
import unittest
from unittest import mock

try:
    from .test_repair_helpers import FakeMask, FakeModel, FakeTensor, import_plugin_submodule, load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeMask, FakeModel, FakeTensor, import_plugin_submodule, load_plugin_package


class TestRepairNativeInpaint(unittest.TestCase):
    def test_node_registers_with_expected_schema(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSNativeInpaintConditioning"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSNativeInpaintConditioning"], "LLS Native Inpaint Conditioning")
        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "encode")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("model", "positive", "negative", "latent", "inpaint_info"),
        )
        self.assertEqual(schema["required"]["patch_mode"][0], ["auto", "disabled", "differential_diffusion"])
        self.assertEqual(schema["optional"]["model_info"], ("STRING",))

    def test_node_wraps_inpaint_model_conditioning_without_flux_patch(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSNativeInpaintConditioning"]()
        repair_runtime = import_plugin_submodule(plugin, "repair.runtime")
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
        positive = [["pos", {}]]
        negative = [["neg", {}]]
        latent = {"samples": "latent"}

        with mock.patch.object(
            repair_runtime,
            "encode_inpaint_conditioning",
            return_value=(positive, negative, latent),
        ) as encode_inpaint, mock.patch.object(
            repair_runtime,
            "apply_differential_diffusion",
        ) as patch_model:
            model_out, positive_out, negative_out, latent_out, info = node.encode(
                model=model,
                positive=positive,
                negative=negative,
                vae=object(),
                image=FakeTensor((1, 1024, 1024, 3)),
                mask=FakeMask((1, 1024, 1024), mask_bbox=(64, 64, 256, 256), mask_area_ratio=0.04),
                patch_mode="auto",
                patch_strength=1.0,
                noise_mask=True,
            )

        encode_inpaint.assert_called_once()
        patch_model.assert_not_called()
        self.assertIs(model_out, model)
        self.assertIs(positive_out, positive)
        self.assertIs(negative_out, negative)
        self.assertIs(latent_out, latent)
        payload = json.loads(info)
        self.assertEqual(payload["model_family"], "SDXL")
        self.assertEqual(payload["applied_patch"], "")

    def test_node_auto_applies_flux_differential_diffusion(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSNativeInpaintConditioning"]()
        repair_runtime = import_plugin_submodule(plugin, "repair.runtime")
        model = FakeModel(
            family="FLUX_DEV",
            model_role="fill",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="flux_guided",
            loader_strategy="flux_split_or_bundle",
        )
        patched_model = object()

        with mock.patch.object(
            repair_runtime,
            "apply_differential_diffusion",
            return_value=patched_model,
        ) as patch_model, mock.patch.object(
            repair_runtime,
            "encode_inpaint_conditioning",
            return_value=([["pos", {}]], [["neg", {}]], {"samples": "latent"}),
        ):
            model_out, _positive_out, _negative_out, _latent_out, info = node.encode(
                model=model,
                positive=[["pos", {}]],
                negative=[["neg", {}]],
                vae=object(),
                image=FakeTensor((1, 1024, 1024, 3)),
                mask=FakeMask((1, 1024, 1024), mask_bbox=(64, 64, 256, 256), mask_area_ratio=0.04),
                patch_mode="auto",
                patch_strength=0.75,
                noise_mask=True,
            )

        patch_model.assert_called_once_with(model, strength=0.75)
        self.assertIs(model_out, patched_model)
        payload = json.loads(info)
        self.assertEqual(payload["model_family"], "FLUX_DEV")
        self.assertEqual(payload["applied_patch"], "differential_diffusion")
        self.assertEqual(payload["patch_strength"], 0.75)


if __name__ == "__main__":
    unittest.main()
