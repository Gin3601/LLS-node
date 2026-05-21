import unittest

try:
    from .test_pro_edit_helpers import import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import import_plugin_submodule, load_plugin_package


class TestProEditRegistration(unittest.TestCase):
    def test_plugin_registers_pro_edit_nodes(self):
        plugin = load_plugin_package()

        self.assertIn("LLSProImageEditPrepare", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSProKSamplerBridge", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSProImageEditFinish", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProImageEditPrepare"],
            "LLS Pro Image Edit Prepare",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProKSamplerBridge"],
            "LLS Pro KSampler Bridge",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProImageEditFinish"],
            "LLS Pro Image Edit Finish",
        )

    def test_prepare_node_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(node_cls.FUNCTION, "prepare")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("LATENT", "IMAGE", "MASK", "LLS_EDIT_INFO", "FLOAT", "CONDITIONING", "CONDITIONING"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("latent", "work_image", "work_mask", "edit_info", "recommended_denoise", "positive", "negative"),
        )
        self.assertEqual(
            set(required),
            {
                "image",
                "mask",
                "vae",
                "positive",
                "negative",
                "backend_mode",
                "edit_scope",
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
            },
        )
        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(required["mask"], ("MASK",))
        self.assertEqual(required["vae"], ("VAE",))
        self.assertEqual(required["positive"], ("CONDITIONING",))
        self.assertEqual(required["negative"], ("CONDITIONING",))
        self.assertEqual(required["backend_mode"][0], ["auto", "sdxl", "flux"])
        self.assertEqual(required["edit_scope"][0], ["auto", "region", "crop", "canvas"])
        self.assertEqual(optional["model"], ("MODEL",))
        self.assertEqual(optional["model_info"], ("STRING",))

    def test_bridge_and_finish_schemas_match_contract(self):
        plugin = load_plugin_package()

        bridge_cls = plugin.NODE_CLASS_MAPPINGS["LLSProKSamplerBridge"]
        bridge_schema = bridge_cls.INPUT_TYPES()
        bridge_required = bridge_schema["required"]
        bridge_optional = bridge_schema["optional"]
        self.assertEqual(bridge_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(bridge_cls.FUNCTION, "sample")
        self.assertEqual(bridge_cls.RETURN_TYPES, ("LATENT", "STRING"))
        self.assertEqual(bridge_cls.RETURN_NAMES, ("latent", "sample_info"))
        self.assertEqual(
            set(bridge_required),
            {
                "model",
                "positive",
                "negative",
                "latent_image",
                "backend_mode",
                "quality_preset",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "denoise_mode",
                "flux_guidance",
                "model_family",
            },
        )
        self.assertEqual(bridge_required["backend_mode"][0], ["auto", "sdxl", "flux"])
        self.assertEqual(bridge_required["denoise_mode"][0], ["manual", "auto_from_edit"])
        self.assertEqual(bridge_optional["edit_info"], ("LLS_EDIT_INFO",))
        self.assertEqual(bridge_optional["model_info"], ("STRING",))

        finish_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditFinish"]
        finish_schema = finish_cls.INPUT_TYPES()
        finish_required = finish_schema["required"]
        finish_optional = finish_schema["optional"]
        self.assertEqual(finish_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(finish_cls.FUNCTION, "finish")
        self.assertEqual(finish_cls.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(finish_cls.RETURN_NAMES, ("final_image", "preview_image"))
        self.assertEqual(finish_required["original_image"], ("IMAGE",))
        self.assertEqual(finish_required["generated_image"], ("IMAGE",))
        self.assertEqual(finish_required["edit_info"], ("LLS_EDIT_INFO",))
        self.assertEqual(finish_optional["work_mask"], ("MASK",))
        self.assertEqual(finish_optional["sample_info"], ("STRING",))

    def test_backend_registry_exposes_built_in_scaffolds(self):
        plugin = load_plugin_package()
        registry = import_plugin_submodule(plugin, "pro_edit.backends.registry")

        self.assertEqual(registry.get_backend("sdxl").backend_name, "sdxl")
        self.assertEqual(registry.get_backend("flux").backend_name, "flux")

    def test_normalize_edit_info_accepts_jsonish_string_payloads(self):
        plugin = load_plugin_package()
        utils = import_plugin_submodule(plugin, "pro_edit.pro_edit_utils")

        info = utils.normalize_edit_info('{"backend_name":"sdxl","edit_scope":"crop"}')
        self.assertEqual(info["backend_name"], "sdxl")
        self.assertEqual(info["edit_scope"], "crop")
        self.assertEqual(info["edit_payload_version"], "1.0")

        legacy_info = utils.normalize_edit_info("backend_name=flux|edit_scope=canvas")
        self.assertEqual(legacy_info["backend_name"], "flux")
        self.assertEqual(legacy_info["edit_scope"], "canvas")
        self.assertEqual(legacy_info["edit_payload_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
