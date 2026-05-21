import unittest
from unittest import mock

try:
    from .test_repair_helpers import FakeMask, FakeTensor, load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeMask, FakeTensor, load_plugin_package


class TestRepairFinish(unittest.TestCase):
    def test_region_finish_dispatches_and_returns_final_preview(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_region_result",
            return_value=FakeTensor((1, 512, 512, 3), "final"),
        ) as compose, mock.patch.object(
            repair_finish,
            "build_preview_image",
            return_value=FakeTensor((1, 512, 512, 3), "preview"),
        ) as preview:
            final_image, preview_image = node.finish(
                original_image=FakeTensor((1, 512, 512, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={"repair_scope": "region", "repair_kernel": "latent_mask", "work_size": [512, 512]},
                feather=8.0,
                color_match="mean_std",
                brightness_match="enabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="soft",
                preview_mode="final",
                work_mask=FakeMask((1, 512, 512), mask_bbox=(64, 64, 256, 256), mask_area_ratio=0.14),
                sample_info=None,
            )

        compose.assert_called_once()
        preview.assert_called_once()
        self.assertEqual(final_image.shape, (1, 512, 512, 3))
        self.assertEqual(preview_image.shape, (1, 512, 512, 3))

    def test_crop_finish_dispatches_with_crop_box(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_crop_result",
            return_value=FakeTensor((1, 1024, 1024, 3), "final"),
        ) as compose:
            final_image, _preview_image = node.finish(
                original_image=FakeTensor((1, 1024, 1024, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={
                    "repair_scope": "crop",
                    "repair_kernel": "vae_inpaint",
                    "crop_box": [100, 100, 356, 356],
                    "work_size": [512, 512],
                },
                feather=8.0,
                color_match="mean_std",
                brightness_match="enabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="none",
                preview_mode="compare",
                work_mask=FakeMask((1, 512, 512), mask_bbox=(0, 0, 256, 256), mask_area_ratio=0.25),
                sample_info=None,
            )

        compose.assert_called_once()
        self.assertEqual(final_image.shape, (1, 1024, 1024, 3))

    def test_canvas_finish_dispatches_with_expanded_output(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_canvas_result",
            return_value=FakeTensor((1, 640, 768, 3), "final"),
        ) as compose:
            final_image, _preview_image = node.finish(
                original_image=FakeTensor((1, 640, 640, 3), "original"),
                generated_image=FakeTensor((1, 640, 768, 3), "generated"),
                repair_info={
                    "repair_scope": "canvas",
                    "repair_kernel": "vae_inpaint",
                    "work_size": [768, 640],
                    "canvas_expand": [128, 0, 0, 0],
                    "original_box_in_canvas": [128, 0, 768, 640],
                },
                feather=8.0,
                color_match="disabled",
                brightness_match="disabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="strong",
                preview_mode="before_after",
                work_mask=FakeMask((1, 640, 768), mask_bbox=(0, 0, 128, 640), mask_area_ratio=0.16),
                sample_info=None,
            )

        compose.assert_called_once()
        self.assertEqual(final_image.shape, (1, 640, 768, 3))

    def test_finish_requires_supported_scope(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_finish import LLSSimpleRepairFinish

        node = LLSSimpleRepairFinish()
        with self.assertRaisesRegex(RuntimeError, "Unsupported repair_scope"):
            node.finish(
                original_image=FakeTensor((1, 512, 512, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={"repair_scope": "mystery"},
                feather=8.0,
                color_match="disabled",
                brightness_match="disabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="none",
                preview_mode="final",
                work_mask=None,
                sample_info=None,
            )


if __name__ == "__main__":
    unittest.main()
