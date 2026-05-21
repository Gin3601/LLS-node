import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestProEditCapabilities(unittest.TestCase):
    def test_parse_model_info_infers_sdxl_inpaint_capability_from_name(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "demo-sdxl-inpaint.safetensors",
                "family": "SDXL",
            }
        )

        self.assertEqual(info["model_role"], "inpaint")
        self.assertTrue(info["supports_inpaint_native"])
        self.assertFalse(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "sdxl")

    def test_parse_model_info_infers_flux_edit_capability_from_name(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "demo-flux-fill-dev.safetensors",
                "family": "FLUX_DEV",
            }
        )

        self.assertEqual(info["model_role"], "fill")
        self.assertFalse(info["supports_inpaint_native"])
        self.assertTrue(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "flux")

    def test_parse_model_info_infers_capabilities_from_ckpt_name_alias(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "ckpt_name": "demo-sdxl-inpaint.safetensors",
                "family": "SDXL",
            }
        )

        self.assertEqual(info["checkpoint_name"], "demo-sdxl-inpaint.safetensors")
        self.assertEqual(info["model_name"], "demo-sdxl-inpaint.safetensors")
        self.assertEqual(info["model_role"], "inpaint")
        self.assertTrue(info["supports_inpaint_native"])
        self.assertEqual(info["preferred_edit_backend"], "sdxl")

    def test_parse_model_info_infers_capabilities_from_ckpt_alias(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "ckpt": "demo-flux-fill-dev.safetensors",
                "family": "FLUX_DEV",
            }
        )

        self.assertEqual(info["checkpoint_name"], "demo-flux-fill-dev.safetensors")
        self.assertEqual(info["model_name"], "demo-flux-fill-dev.safetensors")
        self.assertEqual(info["model_role"], "fill")
        self.assertTrue(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "flux")

    def test_explicit_capability_values_override_name_inference(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "model_role": "edit",
                "supports_inpaint_native": True,
                "supports_image_edit_native": True,
                "preferred_edit_backend": "sdxl",
            }
        )

        self.assertEqual(info["model_role"], "edit")
        self.assertTrue(info["supports_inpaint_native"])
        self.assertTrue(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "sdxl")

    def test_parse_model_info_coerces_string_capability_overrides_to_false(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "supports_inpaint_native": "false",
                "supports_image_edit_native": "0",
            }
        )

        self.assertFalse(info["supports_inpaint_native"])
        self.assertFalse(info["supports_image_edit_native"])

    def test_parse_model_info_normalizes_model_family_alias(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "model_family": "SD15",
                "checkpoint_name": "demo.safetensors",
            }
        )

        self.assertEqual(info["family"], "SD1.5")
        self.assertEqual(info["model_family"], "SD1.5")

    def test_resolve_edit_capabilities_reads_model_tags(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            model_name="demo-flux-edit.safetensors",
        )

        capabilities = model_info.resolve_edit_capabilities(model=model, model_info=None)

        self.assertEqual(capabilities["model_family"], "FLUX_DEV")
        self.assertEqual(capabilities["model_role"], "edit")
        self.assertTrue(capabilities["supports_image_edit_native"])
        self.assertEqual(capabilities["preferred_edit_backend"], "flux")

    def test_resolve_edit_capabilities_coerces_string_overrides_to_false(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        capabilities = model_info.resolve_edit_capabilities(
            model=None,
            model_info={
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "supports_inpaint_native": "false",
                "supports_image_edit_native": "0",
            },
        )

        self.assertFalse(capabilities["supports_inpaint_native"])
        self.assertFalse(capabilities["supports_image_edit_native"])

    def test_loader_builds_capability_tags_for_loaded_objects(self):
        plugin = load_plugin_package()
        loader_module = import_plugin_submodule(plugin, "model_loader.nodes")

        tags = loader_module._build_capability_tags("demo-sdxl-inpaint.safetensors", "SDXL")

        self.assertEqual(tags["model_role"], "inpaint")
        self.assertTrue(tags["supports_inpaint_native"])
        self.assertEqual(tags["preferred_edit_backend"], "sdxl")

    def test_loader_tags_loaded_objects_with_capabilities(self):
        plugin = load_plugin_package()
        loader_module = import_plugin_submodule(plugin, "model_loader.nodes")

        loader = loader_module.LLSSimpleCheckpointLoader()
        model = SimpleNamespace()
        clip = SimpleNamespace()
        vae = SimpleNamespace()

        with mock.patch.object(loader_module, "folder_paths", object()), \
             mock.patch.object(loader_module, "comfy_sd", object()), \
             mock.patch.object(loader_module, "_resolve_model_path", return_value=("checkpoints", "/fake/demo-sdxl-inpaint.safetensors")), \
             mock.patch.object(loader_module, "_load_model", return_value=(model, clip, vae)), \
             mock.patch.object(loader_module, "_resolve_text_encoder", return_value=(clip, "embedded", None, None)), \
             mock.patch.object(loader_module, "_resolve_vae", return_value=(vae, "embedded", None)):
            loaded_model, loaded_clip, loaded_vae, text_encoder = loader.load_checkpoint(
                ckpt_name="demo-sdxl-inpaint.safetensors",
                model_family="SDXL",
                load_mode="simple",
                vae_source="auto",
                text_encoder_source="auto",
                external_vae_name=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_1=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_2=loader_module.AUTO_PLACEHOLDER,
            )

        for obj in (loaded_model, loaded_clip, loaded_vae, text_encoder):
            self.assertEqual(obj._lls_model_role, "inpaint")
            self.assertTrue(obj._lls_supports_inpaint_native)
            self.assertFalse(obj._lls_supports_image_edit_native)
            self.assertEqual(obj._lls_preferred_edit_backend, "sdxl")


if __name__ == "__main__":
    unittest.main()
