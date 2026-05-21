import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestMaskDrawRegistration(unittest.TestCase):
    def test_plugin_registers_mask_draw_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleMaskDraw", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleMaskDraw"],
            "LLS Simple Mask Draw",
        )

    def test_mask_draw_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "draw_mask")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "MASK", "IMAGE"))
        self.assertEqual(node_cls.RETURN_NAMES, ("image", "mask", "preview_image"))

        required = schema["required"]
        optional = schema["optional"]
        hidden = schema["hidden"]

        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(optional["input_mask"], ("MASK",))
        self.assertEqual(required["draw_mode"][0], ["brush", "erase"])
        self.assertEqual(required["brush_size"][0], "INT")
        self.assertEqual(required["brush_size"][1]["max"], 512)
        self.assertEqual(required["brush_softness"][0], "FLOAT")
        self.assertEqual(required["overlay_alpha"][0], "FLOAT")
        self.assertEqual(required["overlay_alpha"][1]["default"], 0.4)
        self.assertEqual(required["invert_mask"][0], "BOOLEAN")
        self.assertEqual(required["mask_state_json"][0], "STRING")
        self.assertEqual(
            required["mask_state_json"][1],
            {"default": "{}", "multiline": False, "advanced": True},
        )
        self.assertEqual(hidden["node_id"], "UNIQUE_ID")
        self.assertTrue(callable(getattr(node_cls, "draw_mask", None)))


if __name__ == "__main__":
    unittest.main()
