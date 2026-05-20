import unittest

from test_repair_helpers import load_plugin_package


class TestRepairRegistration(unittest.TestCase):
    def test_plugin_registers_repair_nodes_with_display_names(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleRepairPrepare", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSSimpleRepairFinish", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleRepairPrepare"],
            "LLS Simple Repair Prepare",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleRepairFinish"],
            "LLS Simple Repair Finish",
        )

    def test_prepare_node_schema(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "prepare")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT"),
        )

        required = schema["required"]
        optional = schema["optional"]

        for field in (
            "image",
            "mask",
            "vae",
            "repair_scope",
            "repair_kernel",
            "task_hint",
            "mask_grow",
            "mask_blur",
            "mask_threshold",
            "invert_mask",
            "crop_context",
            "crop_context_factor",
            "min_size",
            "max_size",
            "resize_mode",
            "expand_left",
            "expand_right",
            "expand_top",
            "expand_bottom",
            "canvas_fill",
            "auto_recommend",
        ):
            self.assertIn(field, required)

        for field in ("model_info", "positive", "negative"):
            self.assertIn(field, optional)

    def test_finish_node_schema(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairFinish"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "finish")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "IMAGE"))

        required = schema["required"]
        optional = schema["optional"]

        for field in (
            "original_image",
            "generated_image",
            "repair_info",
            "feather",
            "color_match",
            "brightness_match",
            "blend_strength",
            "restore_unmasked_area",
            "edge_fix",
            "preview_mode",
        ):
            self.assertIn(field, required)

        for field in ("work_mask", "sample_info"):
            self.assertIn(field, optional)


if __name__ == "__main__":
    unittest.main()
