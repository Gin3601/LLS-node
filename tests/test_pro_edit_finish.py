import unittest

try:
    from .test_pro_edit_helpers import load_plugin_package, make_torch_image, make_torch_mask, torch
except ImportError:
    from test_pro_edit_helpers import load_plugin_package, make_torch_image, make_torch_mask, torch


class TestProEditFinish(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditFinish"]
        self.node = node_cls()

    def test_region_composite_only_replaces_masked_pixels(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(4, 4, 1.0)
        mask = make_torch_mask(4, 4, (1, 1, 3, 3))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "sdxl",
                "edit_scope": "region",
                "original_size": [4, 4],
                "work_size": [4, 4],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="final",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(float(final_image[0, 0, 0, 0].item()), 0.0)
        self.assertEqual(float(final_image[0, 1, 1, 0].item()), 1.0)
        self.assertEqual(tuple(preview_image.shape), tuple(final_image.shape))

    def test_crop_composite_pastes_generated_crop_back_into_original(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(2, 2, 0.75)
        mask = make_torch_mask(2, 2, (0, 0, 2, 2))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "sdxl",
                "edit_scope": "crop",
                "original_size": [4, 4],
                "work_size": [2, 2],
                "crop_box": [1, 1, 3, 3],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="mask",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(float(final_image[0, 0, 0, 0].item()), 0.0)
        self.assertEqual(float(final_image[0, 1, 1, 0].item()), 0.75)
        self.assertEqual(tuple(preview_image.shape), tuple(original.shape))

    def test_canvas_output_keeps_expanded_canvas_size(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(6, 6, 0.5)
        mask = make_torch_mask(6, 6, (0, 0, 6, 6))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "flux",
                "edit_scope": "canvas",
                "original_size": [4, 4],
                "work_size": [6, 6],
                "original_box_in_canvas": [1, 1, 5, 5],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="compare",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(tuple(final_image.shape[1:3]), (6, 6))
        self.assertEqual(tuple(preview_image.shape[1:3]), (6, 12))


if __name__ == "__main__":
    unittest.main()
