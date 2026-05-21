import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestProEditRegistry(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "pro_edit.backends.registry")

    def test_auto_routes_sdxl_inpaint_model(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "model.preferred_edit_backend")

    def test_auto_routes_flux_edit_model(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "flux")
        self.assertEqual(routing.routing_reason, "model.preferred_edit_backend")
        self.assertEqual(routing.capabilities["model_family"], "FLUX_DEV")

    def test_auto_reuses_backend_name_from_edit_info(self):
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
        self.assertEqual(routing.routing_reason, "edit_info.backend_name")

    def test_manual_flux_override_rejects_sdxl_only_model(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )

        with self.assertRaisesRegex(RuntimeError, "backend 'flux' is incompatible"):
            self.registry.resolve_backend("flux", model=model)

    def test_auto_without_matching_backend_raises_clear_error(self):
        model = FakeModel(
            family="SD1.5",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
        )

        with self.assertRaisesRegex(RuntimeError, "No professional edit backend matched"):
            self.registry.resolve_backend("auto", model=model)


if __name__ == "__main__":
    unittest.main()
