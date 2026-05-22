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


class TestProEditPrepareFlux(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        self.node = node_cls()

    def test_prepare_region_routes_to_flux_backend(self):
        image = FakeTensor((1, 1024, 1024, 3), label="image")
        mask = FakeMask((1, 1024, 1024), mask_bbox=(300, 300, 724, 724), mask_area_ratio=0.18)
        vae = FakeVAE(latent_channels=16, downscale_ratio=16)
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="flux_guided",
            loader_strategy="flux_split_or_bundle",
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
            min_size=512,
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

        self.assertEqual(edit_info["backend_name"], "flux")
        self.assertEqual(edit_info["routing_reason"], "profile.backend_type")
        self.assertEqual(edit_info["model_family"], "FLUX_DEV")
        self.assertEqual(edit_info["profile_id"], "flux_edit")
        self.assertEqual(edit_info["backend_type"], "flux_edit")
        self.assertEqual(edit_info["sampler_strategy"], "flux_guided")
        self.assertTrue(edit_info["supports_image_edit_native"])
        self.assertIn("concat_latent_image", positive[0][1])
        self.assertIn("concat_mask", positive[0][1])
        self.assertEqual(positive[0][1]["edit_backend"], "flux")
        self.assertEqual(latent["source"], "pro_edit_prepare_region")
        self.assertGreater(recommended, 0.0)

    def test_prepare_canvas_preserves_canvas_geometry_for_flux(self):
        image = FakeTensor((1, 768, 768, 3), label="image")
        mask = FakeMask((1, 768, 768), mask_bbox=(100, 100, 300, 300), mask_area_ratio=0.07)
        vae = FakeVAE(latent_channels=16, downscale_ratio=16)
        model = FakeModel(
            family="FLUX_DEV",
            model_role="fill",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="flux_guided",
            loader_strategy="flux_split_or_bundle",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="canvas",
            mask_grow=0,
            mask_blur=0.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.0,
            min_size=512,
            max_size=1024,
            resize_mode="fit",
            expand_left=64,
            expand_right=64,
            expand_top=32,
            expand_bottom=32,
            canvas_fill="neutral",
            auto_recommend="enabled",
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["edit_scope"], "canvas")
        self.assertEqual(edit_info["original_box_in_canvas"], [64, 32, 832, 800])
        self.assertEqual(edit_info["work_size"], [896, 832])
        self.assertEqual(edit_info["profile_id"], "flux_edit")
        self.assertEqual(edit_info["backend_type"], "flux_edit")
        self.assertEqual(edit_info["sampler_strategy"], "flux_guided")
        self.assertEqual(positive[0][1]["edit_backend"], "flux")
        self.assertGreater(recommended, 0.0)


if __name__ == "__main__":
    unittest.main()
