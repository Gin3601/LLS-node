import unittest

try:
    from .test_repair_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestRepairRegistry(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "repair.backends.registry")

    def test_native_fill_routes_flux_edit_profile_to_flux_backend(self):
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

        backend, routing = self.registry.resolve_backend("native_fill", model=model)

        self.assertEqual(backend.backend_name, "flux")
        self.assertEqual(routing.backend_name, "flux")
        self.assertEqual(routing.routing_reason, "profile.backend_type")
        self.assertEqual(routing.execution_path, "native_repair")
        self.assertEqual(routing.profile["profile_id"], "flux_edit")
        self.assertEqual(routing.profile["backend_type"], "flux_edit")
        self.assertEqual(routing.capabilities["model_family"], "FLUX_DEV")

    def test_fallback_routes_sdxl_base_profile_to_sdxl_backend(self):
        model = FakeModel(
            family="SDXL",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
            profile_id="sdxl_base",
            backend_type="none",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        backend, routing = self.registry.resolve_backend("vae_inpaint", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "profile.family_fallback")
        self.assertEqual(routing.execution_path, "fallback_repair")
        self.assertEqual(routing.profile["profile_id"], "sdxl_base")

    def test_fallback_routes_sd15_base_profile_to_generic_backend(self):
        model = FakeModel(
            family="SD1.5",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
            profile_id="sd15_base",
            backend_type="none",
            sampler_strategy="standard_k",
            loader_strategy="sd15_checkpoint",
        )

        backend, routing = self.registry.resolve_backend("latent_mask", model=model)

        self.assertEqual(backend.backend_name, "generic")
        self.assertEqual(routing.routing_reason, "profile.family_fallback")
        self.assertEqual(routing.execution_path, "fallback_repair")
        self.assertEqual(routing.profile["profile_id"], "sd15_base")


if __name__ == "__main__":
    unittest.main()
