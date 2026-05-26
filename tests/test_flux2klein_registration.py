import unittest

try:
    from .test_flux2klein_helpers import load_plugin_package
except ImportError:
    from test_flux2klein_helpers import load_plugin_package


class TestFlux2KleinRegistration(unittest.TestCase):
    def test_plugin_registers_flux2klein_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSFlux2KleinEditTextEncode", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSFlux2KleinEditTextEncode"],
            "LLS Flux2Klein Edit Text Encode",
        )

    def test_flux2klein_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSFlux2KleinEditTextEncode"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Flux2Klein")
        self.assertEqual(node_cls.FUNCTION, "encode")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("CONDITIONING", "LATENT", "LLS_FLUX2KLEIN_OUTPUT", "IMAGE", "MASK"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("conditioning", "latent", "custom_output", "main_image", "mask"),
        )
        self.assertEqual(required["clip"], ("CLIP",))
        self.assertEqual(required["vae"], ("VAE",))
        self.assertEqual(required["image1"], ("IMAGE",))
        self.assertEqual(required["prompt"][0], "STRING")
        self.assertTrue(required["prompt"][1]["multiline"])
        self.assertEqual(required["ref_longest_edge"][0], "INT")
        self.assertEqual(required["ref_longest_edge"][1]["default"], 1024)
        self.assertEqual(required["ref_longest_edge"][1]["min"], 256)
        self.assertEqual(required["ref_longest_edge"][1]["max"], 4096)
        self.assertEqual(required["ref_longest_edge"][1]["step"], 64)
        self.assertEqual(required["resize_mode"][0], ["longest_edge", "keep_original"])
        self.assertEqual(required["mask_mode"][0], ["none", "use_mask", "invert_mask"])
        self.assertEqual(optional["image2"], ("IMAGE",))
        self.assertEqual(optional["image3"], ("IMAGE",))
        self.assertEqual(optional["mask"], ("MASK",))


if __name__ == "__main__":
    unittest.main()
