import unittest
from unittest import mock

try:
    import torch
except Exception:
    torch = None

try:
    from .test_flux2klein_helpers import (
        BrokenClipStub,
        FakeMask,
        FakeVAE,
        FakeTensor,
        StandardClipStub,
        import_plugin_submodule,
        load_plugin_package,
    )
except ImportError:
    from test_flux2klein_helpers import (
        BrokenClipStub,
        FakeMask,
        FakeVAE,
        FakeTensor,
        StandardClipStub,
        import_plugin_submodule,
        load_plugin_package,
    )


class TestFlux2KleinNode(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSFlux2KleinEditTextEncode"]
        self.node = node_cls()
        self.module = import_plugin_submodule(plugin, "flux2klein.lls_flux2klein_edit_text_encode")

    def test_fallback_path_builds_conditioning_latent_and_custom_output(self):
        conditioning, latent, custom_output, main_image, output_mask = self.node.encode(
            clip=StandardClipStub(),
            vae=FakeVAE(latent_channels=4, downscale_ratio=8),
            image1=FakeTensor((1, 600, 400, 3), label="main"),
            prompt="make it cinematic",
            ref_longest_edge=1024,
            resize_mode="keep_original",
            mask_mode="use_mask",
            image2=FakeTensor((1, 256, 512, 3), label="ref2"),
            image3=None,
            mask=FakeMask((1, 300, 200), mask_bbox=(10, 10, 40, 40), mask_area_ratio=0.1, label="mask"),
        )

        self.assertEqual(
            self.module._build_flux2klein_prompt("make it cinematic", 2),
            "image1: <|vision_start|><|image_pad|><|vision_end|> "
            "image2: <|vision_start|><|image_pad|><|vision_end|> make it cinematic",
        )
        self.assertEqual(
            conditioning[0][0],
            "cond::image1: <|vision_start|><|image_pad|><|vision_end|> "
            "image2: <|vision_start|><|image_pad|><|vision_end|> make it cinematic",
        )
        self.assertEqual(len(conditioning[0][1]["reference_latents"]), 2)
        self.assertEqual(conditioning[0][1]["reference_latents"][0].shape, (1, 4, 75, 50))
        self.assertEqual(conditioning[0][1]["reference_latents"][1].shape, (1, 4, 75, 50))
        self.assertEqual(latent["samples"].shape, (1, 4, 75, 50))
        self.assertEqual(latent["noise_mask"].shape, (1, 75, 50))
        self.assertEqual(main_image.shape, (1, 600, 400, 3))
        self.assertEqual(output_mask.shape, (1, 600, 400))
        self.assertEqual(custom_output["node_name"], "LLS Flux2Klein Edit Text Encode")
        self.assertEqual(custom_output["prompt"], "make it cinematic")
        self.assertEqual(custom_output["main_image"], "image1")
        self.assertEqual(custom_output["reference_images"], ["image2"])
        self.assertEqual(custom_output["num_reference_images"], 1)
        self.assertEqual(custom_output["conditioning_backend"], "flux2klein_multivision_clip")
        self.assertTrue(custom_output["has_mask"])
        self.assertEqual(custom_output["latent_mode"], "image1_reference_latent")
        self.assertEqual(custom_output["latent_has_noise_mask"], True)
        self.assertEqual(custom_output["vision_image_count"], 2)
        self.assertEqual(custom_output["reference_latent_count"], 2)
        self.assertEqual(self.module._count_present_reference_images(main_image, None, None), 1)

    def test_longest_edge_resize_scales_images_and_inverts_mask(self):
        conditioning, latent, custom_output, main_image, output_mask = self.node.encode(
            clip=StandardClipStub(),
            vae=FakeVAE(latent_channels=16, downscale_ratio=16),
            image1=FakeTensor((1, 800, 400, 3), label="main"),
            prompt="change outfit",
            ref_longest_edge=1024,
            resize_mode="longest_edge",
            mask_mode="invert_mask",
            image2=FakeTensor((1, 400, 1600, 3), label="ref2"),
            image3=FakeTensor((1, 600, 300, 3), label="ref3"),
            mask=FakeMask((1, 800, 400), mask_bbox=(20, 20, 180, 380), mask_area_ratio=0.25, label="mask"),
        )

        self.assertEqual(
            conditioning[0][0],
            "cond::image1: <|vision_start|><|image_pad|><|vision_end|> "
            "image2: <|vision_start|><|image_pad|><|vision_end|> "
            "image3: <|vision_start|><|image_pad|><|vision_end|> change outfit",
        )
        self.assertEqual(len(conditioning[0][1]["reference_latents"]), 3)
        self.assertEqual(main_image.shape, (1, 1024, 512, 3))
        self.assertEqual(latent["samples"].shape, (1, 16, 64, 32))
        self.assertEqual(latent["noise_mask"].shape, (1, 64, 32))
        self.assertEqual(output_mask.shape, (1, 1024, 512))
        self.assertAlmostEqual(output_mask.mask_area_ratio, 0.75)
        self.assertEqual(custom_output["reference_images"], ["image2", "image3"])
        self.assertEqual(custom_output["num_reference_images"], 2)
        self.assertEqual(custom_output["resize_mode"], "longest_edge")
        self.assertEqual(custom_output["mask_mode"], "invert_mask")
        self.assertEqual(custom_output["vision_image_count"], 3)
        self.assertEqual(custom_output["reference_latent_count"], 3)
        self.assertEqual(custom_output["latent_has_noise_mask"], True)

    def test_mask_mode_none_outputs_full_mask_but_no_noise_mask(self):
        conditioning, latent, custom_output, main_image, output_mask = self.node.encode(
            clip=StandardClipStub(),
            vae=FakeVAE(latent_channels=16, downscale_ratio=16),
            image1=FakeTensor((1, 512, 512, 3), label="main"),
            prompt="replace background",
            ref_longest_edge=1024,
            resize_mode="keep_original",
            mask_mode="none",
            image2=FakeTensor((1, 512, 512, 3), label="ref2"),
            image3=None,
            mask=FakeMask((1, 512, 512), mask_bbox=(0, 0, 256, 256), mask_area_ratio=0.25, label="mask"),
        )

        self.assertEqual(main_image.shape, (1, 512, 512, 3))
        self.assertEqual(output_mask.shape, (1, 512, 512))
        if hasattr(output_mask, "mask_area_ratio"):
            self.assertEqual(output_mask.mask_area_ratio, 1.0)
        else:
            self.assertTrue(torch is not None)
            self.assertTrue(torch.all(output_mask == 1))
        self.assertNotIn("noise_mask", latent)
        self.assertEqual(custom_output["latent_has_noise_mask"], False)

    def test_missing_clip_backend_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "vision-aware|text encoding backend"):
            self.node.encode(
                clip=BrokenClipStub(),
                vae=FakeVAE(),
                image1=FakeTensor((1, 512, 512, 3), label="main"),
                prompt="replace background",
                ref_longest_edge=1024,
                resize_mode="keep_original",
                mask_mode="none",
                image2=None,
                image3=None,
                mask=None,
            )


if __name__ == "__main__":
    unittest.main()
