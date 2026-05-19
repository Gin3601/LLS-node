import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_node_test",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test"] = module
    spec.loader.exec_module(module)
    return module


class TestLLSUniversalNode(unittest.TestCase):
    def test_plugin_registers_universal_node(self):
        plugin = load_plugin_package()
        self.assertIn("LLSUniversalImageGenerator", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSUniversalImageGenerator"],
            "LLS Universal Image Generator",
        )

    def test_existing_nodes_still_registered(self):
        plugin = load_plugin_package()
        self.assertIn("LLSUpscaleSwitcher", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSSimpleKSampler", plugin.NODE_CLASS_MAPPINGS)

    def test_universal_node_schema_matches_spec(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSUniversalImageGenerator"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Image")
        self.assertEqual(node_cls.FUNCTION, "generate")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(
            tuple(required.keys()),
            (
                "model_family",
                "task_mode",
                "model_name",
                "positive_prompt",
                "negative_prompt",
                "width",
                "height",
                "steps",
                "cfg",
                "seed",
                "sampler_name",
                "scheduler",
                "denoise",
            ),
        )

    def test_dispatcher_returns_backend_classes(self):
        load_plugin_package()
        from lls_node_test.lls_universal.backend_flux import FluxBackend
        from lls_node_test.lls_universal.backend_sd15 import SD15Backend
        from lls_node_test.lls_universal.backend_sdxl import SDXLBackend
        from lls_node_test.lls_universal.dispatcher import get_backend

        self.assertIsInstance(get_backend("SD1.5"), SD15Backend)
        self.assertIsInstance(get_backend("SDXL"), SDXLBackend)
        self.assertIsInstance(get_backend("FLUX"), FluxBackend)


if __name__ == "__main__":
    unittest.main()
