import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestProEditRegistry(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "pro_edit.backends.registry")

    def test_auto_routes_sdxl_profile_by_backend_type(self):
        model = FakeModel(
            family="SDXL",
            model_role="edit",
            supports_inpaint_native=True,
            supports_image_edit_native=True,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_edit",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "profile.backend_type")
        self.assertEqual(routing.profile["profile_id"], "sdxl_edit")
        self.assertEqual(routing.profile["backend_type"], "sdxl_native")

    def test_auto_routes_flux_profile_by_backend_type(self):
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

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "flux")
        self.assertEqual(routing.routing_reason, "profile.backend_type")
        self.assertEqual(routing.capabilities["model_family"], "FLUX_DEV")
        self.assertEqual(routing.profile["profile_id"], "flux_edit")

    def test_auto_upgrades_legacy_edit_info_to_profile_route(self):
        backend, routing = self.registry.resolve_backend(
            "auto",
            model=None,
            edit_info={
                "backend_name": "sdxl",
                "model_family": "SDXL",
                "model_role": "inpaint",
                "supports_inpaint_native": True,
                "supports_image_edit_native": False,
                "preferred_edit_backend": "sdxl",
            },
        )

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "profile.backend_type")
        self.assertEqual(routing.profile["profile_id"], "sdxl_inpaint")

    def test_manual_flux_override_rejects_sdxl_only_model(self):
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

        with self.assertRaisesRegex(RuntimeError, "backend 'flux' is incompatible with profile"):
            self.registry.resolve_backend("flux", model=model)

    def test_auto_routes_sdxl_base_profile_to_family_fallback(self):
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

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "profile.family_fallback")
        self.assertEqual(routing.execution_path, "fallback_repair")
        self.assertEqual(routing.profile["profile_id"], "sdxl_base")

    def test_auto_routes_sd15_base_profile_to_generic_fallback(self):
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

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "generic")
        self.assertEqual(routing.routing_reason, "profile.family_fallback")
        self.assertEqual(routing.execution_path, "fallback_repair")
        self.assertEqual(routing.profile["profile_id"], "sd15_base")

    def test_manual_sdxl_override_keeps_sdxl_base_on_fallback_path(self):
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

        backend, routing = self.registry.resolve_backend("sdxl", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "manual.family_fallback")
        self.assertEqual(routing.execution_path, "fallback_repair")


if __name__ == "__main__":
    unittest.main()
