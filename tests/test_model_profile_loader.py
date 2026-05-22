import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from .test_pro_edit_helpers import import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import import_plugin_submodule, load_plugin_package


class TestModelProfileLoader(unittest.TestCase):
    def test_loader_writes_profile_fields_to_outputs(self):
        plugin = load_plugin_package()
        loader_module = import_plugin_submodule(plugin, "model_loader.nodes")
        loader = loader_module.LLSSimpleCheckpointLoader()
        model = SimpleNamespace()
        clip = SimpleNamespace()
        vae = SimpleNamespace()

        with mock.patch.object(loader_module, "folder_paths", object()), \
             mock.patch.object(loader_module, "comfy_sd", object()), \
             mock.patch.object(loader_module, "_resolve_model_path", return_value=("diffusion_models", "/fake/demo-flux-kontext-dev.safetensors")), \
             mock.patch.object(loader_module, "_load_model", return_value=(model, clip, vae)), \
             mock.patch.object(loader_module, "_resolve_text_encoder", return_value=(clip, "embedded", None, None)), \
             mock.patch.object(loader_module, "_resolve_vae", return_value=(vae, "embedded", None)):
            loaded_model, loaded_clip, loaded_vae, text_encoder = loader.load_checkpoint(
                ckpt_name="demo-flux-kontext-dev.safetensors",
                model_family="FLUX_DEV",
                load_mode="simple",
                vae_source="auto",
                text_encoder_source="auto",
                external_vae_name=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_1=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_2=loader_module.AUTO_PLACEHOLDER,
            )

        for obj in (loaded_model, loaded_clip, loaded_vae, text_encoder):
            self.assertEqual(obj._lls_profile_id, "flux_edit")
            self.assertEqual(obj._lls_backend_type, "flux_edit")
            self.assertEqual(obj._lls_sampler_strategy, "flux_guided")
            self.assertEqual(obj._lls_loader_strategy, "flux_split_or_bundle")
            self.assertEqual(obj._lls_model_role, "edit")


if __name__ == "__main__":
    unittest.main()
