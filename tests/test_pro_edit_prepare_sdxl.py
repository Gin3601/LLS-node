import unittest

try:
    from .test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )
except ImportError:
    from test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )


class TestProEditPrepareSDXL(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        self.node = node_cls()

    def test_prepare_region_adds_concat_conditioning_for_sdxl(self):
        image = FakeTensor((1, 128, 128, 3), label="image")
        mask = FakeMask((1, 128, 128), mask_bbox=(32, 32, 96, 96), mask_area_ratio=0.25)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_inpaint",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="region",
            mask_grow=8,
            mask_blur=4.0,
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
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["backend_name"], "sdxl")
        self.assertEqual(edit_info["routing_reason"], "profile.backend_type")
        self.assertEqual(edit_info["edit_scope"], "region")
        self.assertEqual(edit_info["profile_id"], "sdxl_inpaint")
        self.assertEqual(edit_info["backend_type"], "sdxl_native")
        self.assertEqual(edit_info["sampler_strategy"], "standard_k")
        self.assertEqual(latent["source"], "pro_edit_prepare_region")
        self.assertIn("concat_latent_image", positive[0][1])
        self.assertIn("concat_mask", positive[0][1])
        self.assertIn("concat_latent_image", negative[0][1])
        self.assertIn("concat_mask", negative[0][1])
        self.assertEqual(work_image.shape, image.shape)
        self.assertEqual(work_mask.shape, mask.shape)
        self.assertGreater(recommended, 0.0)

    def test_prepare_crop_writes_crop_geometry(self):
        image = FakeTensor((1, 1024, 1024, 3), label="image")
        mask = FakeMask((1, 1024, 1024), mask_bbox=(128, 128, 256, 256), mask_area_ratio=0.02)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="edit",
            supports_inpaint_native=True,
            supports_image_edit_native=True,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_edit",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="crop",
            mask_grow=0,
            mask_blur=0.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=32,
            crop_context_factor=1.0,
            min_size=256,
            max_size=512,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["edit_scope"], "crop")
        self.assertIsInstance(edit_info["crop_box"], list)
        self.assertEqual(edit_info["work_size"], [work_image.shape[2], work_image.shape[1]])
        self.assertEqual(edit_info["backend_name"], "sdxl")
        self.assertEqual(edit_info["profile_id"], "sdxl_edit")
        self.assertEqual(edit_info["backend_type"], "sdxl_native")
        self.assertEqual(edit_info["sampler_strategy"], "standard_k")
        self.assertEqual(positive[0][1]["edit_backend"], "sdxl")
        self.assertEqual(negative[0][1]["edit_backend"], "sdxl")
        self.assertGreater(recommended, 0.0)


if __name__ == "__main__":
    unittest.main()
