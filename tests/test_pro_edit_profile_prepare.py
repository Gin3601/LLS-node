import unittest

try:
    from .test_pro_edit_helpers import FakeMask, FakeModel, FakeTensor, FakeVAE, load_plugin_package, make_conditioning
except ImportError:
    from test_pro_edit_helpers import FakeMask, FakeModel, FakeTensor, FakeVAE, load_plugin_package, make_conditioning


class TestProEditProfilePrepare(unittest.TestCase):
    def test_prepare_rejects_base_profile(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]()
        image = FakeTensor((1, 64, 64, 3), label="image")
        mask = FakeMask((1, 64, 64), mask_bbox=(8, 8, 32, 32), mask_area_ratio=0.2)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
            profile_id="sdxl_base",
            backend_type="none",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        with self.assertRaisesRegex(RuntimeError, "Pro image edit is not available for profile 'sdxl_base'"):
            node.prepare(
                image=image,
                mask=mask,
                vae=vae,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                backend_mode="auto",
                edit_scope="region",
                mask_grow=0,
                mask_blur=0.0,
                mask_threshold=0.5,
                invert_mask=False,
                crop_context=32,
                crop_context_factor=1.0,
                min_size=128,
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


if __name__ == "__main__":
    unittest.main()
