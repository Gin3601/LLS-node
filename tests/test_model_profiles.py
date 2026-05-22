import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestModelProfiles(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "model_profiles.registry")

    def test_resolves_sdxl_base_profile_from_family(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={"checkpoint_name": "plain-sdxl-base.safetensors", "family": "SDXL"},
        )

        self.assertEqual(profile["profile_id"], "sdxl_base")
        self.assertEqual(profile["backend_type"], "none")
        self.assertEqual(profile["sampler_strategy"], "standard_k")
        self.assertEqual(profile["loader_strategy"], "sdxl_checkpoint")

    def test_resolves_flux_kontext_profile_from_name(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={"checkpoint_name": "demo-flux-kontext-dev.safetensors", "family": "FLUX_DEV"},
        )

        self.assertEqual(profile["profile_id"], "flux_edit")
        self.assertEqual(profile["role"], "edit")
        self.assertEqual(profile["backend_type"], "flux_edit")
        self.assertEqual(profile["sampler_strategy"], "flux_guided")
        self.assertEqual(profile["loader_strategy"], "flux_split_or_bundle")

    def test_model_info_override_has_highest_priority(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "profile_id": "sdxl_edit",
                "role": "edit",
                "backend_type": "sdxl_native",
                "sampler_strategy": "standard_k",
                "supports_inpaint_native": True,
                "supports_image_edit_native": True,
                "preferred_edit_backend": "sdxl",
            },
        )

        self.assertEqual(profile["profile_id"], "sdxl_edit")
        self.assertEqual(profile["role"], "edit")
        self.assertEqual(profile["backend_type"], "sdxl_native")
        self.assertTrue(profile["supports_image_edit_native"])

    def test_legacy_tags_upgrade_to_matching_profile(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="",
            backend_type="",
            sampler_strategy="",
            loader_strategy="",
            model_name="legacy-flux-edit.safetensors",
        )

        profile = self.registry.resolve_model_profile(model=model, model_info=None)

        self.assertEqual(profile["profile_id"], "flux_edit")
        self.assertEqual(profile["backend_type"], "flux_edit")
        self.assertEqual(profile["sampler_strategy"], "flux_guided")


if __name__ == "__main__":
    unittest.main()
