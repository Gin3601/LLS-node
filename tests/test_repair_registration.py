import unittest

try:
    from .test_repair_helpers import load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
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
            ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT", "CONDITIONING", "CONDITIONING"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise", "positive", "negative"),
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
        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(required["mask"], ("MASK",))
        self.assertEqual(required["vae"], ("VAE",))
        self.assertEqual(optional["model_info"], ("STRING",))
        self.assertEqual(optional["positive"], ("CONDITIONING",))
        self.assertEqual(optional["negative"], ("CONDITIONING",))
        self.assertEqual(
            required["repair_scope"],
            (["auto", "region", "crop", "canvas"], {"default": "auto"}),
        )
        self.assertEqual(
            required["repair_kernel"],
            (["auto", "latent_mask", "vae_inpaint", "native_fill"], {"default": "auto"}),
        )
        self.assertEqual(
            required["task_hint"],
            (
                [
                    "auto",
                    "repair",
                    "remove",
                    "replace",
                    "fill",
                    "appearance",
                    "content",
                    "structure",
                    "dehaze",
                    "deshadow",
                    "recolor",
                ],
                {"default": "auto"},
            ),
        )
        self.assertEqual(
            required["crop_context"],
            ("INT", {"default": 64, "min": 0, "max": 512}),
        )
        self.assertEqual(
            required["auto_recommend"],
            (["enabled", "disabled"], {"default": "enabled"}),
        )
        self.assertEqual(
            required["canvas_fill"],
            (["edge", "blur", "black", "white", "neutral"], {"default": "edge"}),
        )

    def test_finish_node_schema(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairFinish"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "finish")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(node_cls.RETURN_NAMES, ("final_image", "preview_image"))

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
        self.assertEqual(required["original_image"], ("IMAGE",))
        self.assertEqual(required["generated_image"], ("IMAGE",))
        self.assertEqual(required["repair_info"], ("LLS_REPAIR_INFO",))
        self.assertEqual(optional["work_mask"], ("MASK",))
        self.assertEqual(optional["sample_info"], ("STRING",))
        self.assertEqual(
            required["color_match"],
            (["disabled", "mean_std", "histogram_simple"], {"default": "disabled"}),
        )
        self.assertEqual(
            required["brightness_match"],
            (["disabled", "enabled"], {"default": "enabled"}),
        )
        self.assertEqual(
            required["edge_fix"],
            (["none", "soft", "strong"], {"default": "soft"}),
        )
        self.assertEqual(
            required["preview_mode"],
            (["final", "compare", "mask", "before_after"], {"default": "final"}),
        )


if __name__ == "__main__":
    unittest.main()
