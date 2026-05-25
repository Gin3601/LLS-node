import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestConcatByTargetRegistration(unittest.TestCase):
    def test_plugin_registers_concat_by_target_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSConcatByTarget", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSConcatByTarget"],
            "LLS Concat By Target",
        )

    def test_concat_by_target_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSConcatByTarget"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Utils")
        self.assertEqual(node_cls.FUNCTION, "concat")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "MASK", "INT", "INT"))
        self.assertEqual(node_cls.RETURN_NAMES, ("image", "mask", "width", "height"))
        self.assertEqual(required["data_type"][0], ["IMAGE", "MASK"])
        self.assertEqual(required["target"][0], ["A", "B"])
        self.assertEqual(required["position"][0], ["top", "bottom", "left", "right"])
        self.assertEqual(required["resize_mode"][0], ["keep_proportion", "stretch", "none"])
        self.assertEqual(required["align"][0], ["start", "center", "end"])
        self.assertEqual(required["match_target_size"][0], "BOOLEAN")
        self.assertEqual(required["gap"][0], "INT")
        self.assertEqual(required["gap"][1]["default"], 0)
        self.assertEqual(required["background_color"][0], "STRING")
        self.assertEqual(required["background_color"][1]["default"], "#000000")
        self.assertEqual(required["background_value"][0], "FLOAT")
        self.assertEqual(required["background_value"][1]["default"], 0.0)
        self.assertEqual(required["multiple_of"][0], "INT")
        self.assertEqual(required["allow_batch_broadcast"][0], "BOOLEAN")
        self.assertEqual(str(optional["a"][0]), "*")
        self.assertEqual(str(optional["b"][0]), "*")
        self.assertNotIn("image_a", optional)
        self.assertNotIn("image_b", optional)
        self.assertNotIn("mask_a", optional)
        self.assertNotIn("mask_b", optional)


if __name__ == "__main__":
    unittest.main()
