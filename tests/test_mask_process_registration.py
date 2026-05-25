import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestMaskProcessRegistration(unittest.TestCase):
    def test_plugin_registers_mask_processing_nodes(self):
        plugin = load_plugin_package()

        self.assertIn("LLSMaskProcess", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSMaskCombine", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSMaskProcess"],
            "LLS Mask Process",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSMaskCombine"],
            "LLS Mask Combine",
        )

    def test_mask_process_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSMaskProcess"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Mask")
        self.assertEqual(node_cls.FUNCTION, "process")
        self.assertEqual(node_cls.RETURN_TYPES, ("MASK",))
        self.assertEqual(node_cls.RETURN_NAMES, ("mask",))
        self.assertEqual(required["mask"], ("MASK",))
        self.assertEqual(
            required["operation"][0],
            [
                "passthrough",
                "threshold",
                "invert",
                "grow",
                "shrink",
                "blur",
                "feather",
                "fill_holes",
                "remove_small_regions",
                "smooth",
                "clamp",
                "resize_to_image",
            ],
        )
        self.assertEqual(required["value_float"][0], "FLOAT")
        self.assertEqual(required["value_float"][1]["default"], 0.5)
        self.assertEqual(required["value_float"][1]["min"], 0.0)
        self.assertEqual(required["value_float"][1]["max"], 1.0)
        self.assertEqual(required["value_int"][0], "INT")
        self.assertEqual(required["value_int"][1]["default"], 8)
        self.assertEqual(required["value_int"][1]["min"], -512)
        self.assertEqual(required["value_int"][1]["max"], 512)
        self.assertEqual(optional["image"], ("IMAGE",))

    def test_mask_combine_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSMaskCombine"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Mask")
        self.assertEqual(node_cls.FUNCTION, "combine")
        self.assertEqual(node_cls.RETURN_TYPES, ("MASK",))
        self.assertEqual(node_cls.RETURN_NAMES, ("mask",))
        self.assertEqual(required["mask_a"], ("MASK",))
        self.assertEqual(required["mask_b"], ("MASK",))
        self.assertEqual(required["mode"][0], ["add", "subtract", "intersect", "xor", "max", "min"])

    def test_plugin_does_not_register_mask_preview_node(self):
        plugin = load_plugin_package()

        self.assertNotIn("LLSMaskPreview", plugin.NODE_CLASS_MAPPINGS)
        self.assertNotIn("LLSMaskPreview", plugin.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
