import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestMaskCreateRegistration(unittest.TestCase):
    def test_plugin_registers_mask_create_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleMaskCreate", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleMaskCreate"],
            "LLS Simple Mask Create",
        )

    def test_mask_create_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskCreate"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Mask")
        self.assertEqual(node_cls.FUNCTION, "create_mask")
        self.assertEqual(node_cls.RETURN_TYPES, ("MASK", "IMAGE", "LLS_MASK_INFO"))
        self.assertEqual(node_cls.RETURN_NAMES, ("mask", "mask_image", "area_info"))

        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(required["image_width"][0], "INT")
        self.assertEqual(required["image_height"][0], "INT")
        self.assertEqual(required["image_width"][1]["default"], 1024)
        self.assertEqual(required["image_height"][1]["default"], 1024)
        self.assertEqual(optional["input_mask"], ("MASK",))
        self.assertEqual(required["shape_type"][0], ["rectangle", "square", "circle", "ellipse"])
        self.assertEqual(required["coordinate_mode"][0], ["pixel", "percent"])
        self.assertEqual(required["shape_type"][1]["default"], "rectangle")
        self.assertEqual(required["coordinate_mode"][1]["default"], "percent")
        self.assertEqual(required["center_x"][0], "FLOAT")
        self.assertEqual(required["center_x"][1]["default"], 0.5)
        self.assertEqual(required["center_y"][1]["default"], 0.5)
        self.assertEqual(required["width"][1]["default"], 0.3)
        self.assertEqual(required["height"][1]["default"], 0.3)
        self.assertEqual(required["radius"][1]["default"], 0.15)
        self.assertEqual(required["feather"][0], "FLOAT")
        self.assertEqual(required["blur"][0], "FLOAT")
        self.assertEqual(required["invert_mask"][0], "BOOLEAN")
        self.assertEqual(required["combine_mode"][0], ["replace", "union", "subtract", "intersect"])
        self.assertEqual(required["combine_mode"][1]["default"], "replace")
        self.assertTrue(callable(getattr(node_cls, "create_mask", None)))

    def test_plugin_registers_mask_preview_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleMaskPreview", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleMaskPreview"],
            "LLS Simple Mask Preview",
        )

    def test_mask_preview_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskPreview"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Mask")
        self.assertEqual(node_cls.FUNCTION, "preview_mask")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("preview_image",))

        required = schema["required"]

        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(required["mask"], ("MASK",))
        self.assertEqual(required["overlay_alpha"][0], "FLOAT")
        self.assertEqual(required["overlay_alpha"][1]["default"], 0.4)
        self.assertEqual(required["overlay_color"][0], ["red", "green", "blue"])
        self.assertTrue(callable(getattr(node_cls, "preview_mask", None)))


if __name__ == "__main__":
    unittest.main()
