import unittest

try:
    from .test_image_composite_helpers import load_plugin_package
except ImportError:
    from test_image_composite_helpers import load_plugin_package


class TestImageCompositeRegistration(unittest.TestCase):
    def test_plugin_registers_image_composite_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleImageComposite", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleImageComposite"],
            "LLS Simple Image Composite",
        )

    def test_image_composite_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleImageComposite"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Image")
        self.assertEqual(node_cls.FUNCTION, "composite")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("output_image",))
        self.assertEqual(required["background_image"], ("IMAGE",))
        self.assertEqual(required["overlay_image"], ("IMAGE",))
        self.assertEqual(required["x_offset"][0], "INT")
        self.assertEqual(required["x_offset"][1]["default"], 0)
        self.assertEqual(required["y_offset"][0], "INT")
        self.assertEqual(required["anchor_mode"][0], ["top_left", "center"])
        self.assertEqual(required["rotation_origin_mode"][0], ["top_left", "center"])
        self.assertEqual(required["opacity"][0], "FLOAT")
        self.assertEqual(required["opacity"][1]["default"], 1.0)
        self.assertEqual(required["blend_mode"][0], ["normal"])
        self.assertEqual(required["scale"][0], "FLOAT")
        self.assertEqual(required["rotation"][0], "FLOAT")
        self.assertEqual(required["keep_aspect"][0], "BOOLEAN")


if __name__ == "__main__":
    unittest.main()
