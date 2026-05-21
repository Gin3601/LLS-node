import unittest

try:
    from .test_repair_helpers import FakeMask, FakeVAE, FakeTensor, load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeMask, FakeVAE, FakeTensor, load_plugin_package


class TestRepairPrepare(unittest.TestCase):
    def test_prepare_region_with_latent_mask_attaches_noise_mask(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 1024, 1024, 3), label="region-image")
        mask = FakeMask(
            (1, 1024, 1024),
            mask_bbox=(128, 128, 384, 384),
            mask_area_ratio=0.05,
            label="region-mask",
        )

        latent, work_image, work_mask, repair_info, recommended_denoise, positive, negative = node.prepare(
            image=image,
            mask=mask,
            vae=FakeVAE(),
            repair_scope="region",
            repair_kernel="latent_mask",
            task_hint="repair",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
        )

        self.assertEqual(work_image.shape, (1, 1024, 1024, 3))
        self.assertEqual(work_mask.shape, (1, 1024, 1024))
        self.assertIn("noise_mask", latent)
        self.assertEqual(repair_info["repair_scope"], "region")
        self.assertEqual(repair_info["repair_kernel"], "latent_mask")
        self.assertEqual(recommended_denoise, 0.45)
        self.assertIsNone(positive)
        self.assertIsNone(negative)

    def test_prepare_crop_with_vae_inpaint_populates_crop_metadata(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 768, 1024, 3), label="crop-image")
        mask = FakeMask(
            (1, 768, 1024),
            mask_bbox=(200, 120, 420, 360),
            mask_area_ratio=0.06,
            label="crop-mask",
        )

        latent, work_image, work_mask, repair_info, recommended_denoise, positive, negative = node.prepare(
            image=image,
            mask=mask,
            vae=FakeVAE(),
            repair_scope="crop",
            repair_kernel="vae_inpaint",
            task_hint="content",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
        )

        self.assertEqual(repair_info["repair_scope"], "crop")
        self.assertIsNotNone(repair_info["crop_box"])
        self.assertGreater(repair_info["crop_scale"], 0.0)
        self.assertEqual(repair_info["work_size"], (work_image.shape[2], work_image.shape[1]))
        self.assertEqual(recommended_denoise, 0.65)
        self.assertEqual(latent["source"], "repair_prepare_crop")
        self.assertEqual(work_mask.shape, work_image.shape[:3])
        self.assertIn(":crop[", work_image.label)
        self.assertIn(":crop[", work_mask.label)
        self.assertEqual(work_image.crop_box, tuple(repair_info["crop_box"]))
        self.assertEqual(work_mask.crop_box, tuple(repair_info["crop_box"]))
        self.assertIsNone(positive)
        self.assertIsNone(negative)

    def test_prepare_canvas_with_vae_inpaint_supports_empty_mask(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 640, 640, 3), label="canvas-image")
        mask = FakeMask((1, 640, 640), mask_bbox=None, mask_area_ratio=0.0, label="canvas-mask")

        latent, work_image, work_mask, repair_info, recommended_denoise, positive, negative = node.prepare(
            image=image,
            mask=mask,
            vae=FakeVAE(),
            repair_scope="auto",
            repair_kernel="vae_inpaint",
            task_hint="fill",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=128,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
        )

        self.assertEqual(work_image.shape, (1, 640, 768, 3))
        self.assertEqual(work_mask.shape, (1, 640, 768))
        self.assertEqual(repair_info["repair_scope"], "canvas")
        self.assertEqual(repair_info["canvas_expand"], [128, 0, 0, 0])
        self.assertEqual(recommended_denoise, 0.90)
        self.assertEqual(latent["source"], "repair_prepare_canvas")
        self.assertGreater(work_mask.mask_area_ratio, 0.0)
        self.assertTrue(repair_info["has_mask"])
        self.assertIsNotNone(repair_info["mask_bbox"])
        self.assertEqual(work_image.fill_mode, "edge")
        self.assertIsNone(positive)
        self.assertIsNone(negative)

    def test_prepare_region_applies_mask_preprocessing_before_metrics(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 512, 512, 3), label="preprocess-image")
        mask = FakeMask(
            (1, 512, 512),
            mask_bbox=(200, 200, 280, 280),
            mask_area_ratio=((280 - 200) * (280 - 200)) / float(512 * 512),
            label="preprocess-mask",
        )

        _latent, _work_image, work_mask, repair_info, _recommended_denoise, _positive, _negative = node.prepare(
            image=image,
            mask=mask,
            vae=FakeVAE(),
            repair_scope="region",
            repair_kernel="vae_inpaint",
            task_hint="content",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.8,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
        )

        self.assertIn("normalize", work_mask.label)
        self.assertIn("threshold[0.8]", work_mask.label)
        self.assertIn("grow[24]", work_mask.label)
        self.assertIn("blur[8.0]", work_mask.label)
        self.assertNotEqual(work_mask.mask_bbox, mask.mask_bbox)
        self.assertGreater(work_mask.mask_area_ratio, mask.mask_area_ratio)
        self.assertEqual(repair_info["mask_bbox"], list(work_mask.mask_bbox))
        self.assertEqual(repair_info["mask_area_ratio"], work_mask.mask_area_ratio)

    def test_prepare_passthroughs_positive_and_negative_outputs(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 512, 512, 3), label="passthrough-image")
        mask = FakeMask((1, 512, 512), mask_bbox=(64, 64, 128, 128), mask_area_ratio=0.02)
        positive = [["pos", {"strength": 1.0}]]
        negative = [["neg", {"strength": 1.0}]]

        _latent, _work_image, _work_mask, _repair_info, _recommended_denoise, positive_out, negative_out = node.prepare(
            image=image,
            mask=mask,
            vae=FakeVAE(),
            repair_scope="region",
            repair_kernel="latent_mask",
            task_hint="repair",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
            positive=positive,
            negative=negative,
        )

        self.assertIs(positive_out, positive)
        self.assertIs(negative_out, negative)

    def test_prepare_requires_vae(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

        image = FakeTensor((1, 1024, 1024, 3), label="missing-vae-image")
        mask = FakeMask((1, 1024, 1024), mask_bbox=(0, 0, 64, 64), mask_area_ratio=0.01)

        with self.assertRaisesRegex(RuntimeError, "Missing VAE"):
            node.prepare(
                image=image,
                mask=mask,
                vae=None,
                repair_scope="region",
                repair_kernel="latent_mask",
                task_hint="repair",
                mask_grow=24,
                mask_blur=8.0,
                mask_threshold=0.5,
                invert_mask=False,
                crop_context=64,
                crop_context_factor=1.5,
                min_size=256,
                max_size=1024,
                resize_mode="fit",
                expand_left=0,
                expand_right=0,
                expand_top=0,
                expand_bottom=0,
                canvas_fill="edge",
                auto_recommend="enabled",
                model_info={"model_family": "SDXL", "model_role": "base"},
            )


if __name__ == "__main__":
    unittest.main()
