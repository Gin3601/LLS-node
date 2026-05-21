import unittest

try:
    from .test_mask_draw_helpers import (
        HAS_MASK_DRAW_RUNTIME_DEPS,
        MASK_DRAW_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
        make_mask,
        make_mask_state_json,
        torch,
    )
except ImportError:
    from test_mask_draw_helpers import (
        HAS_MASK_DRAW_RUNTIME_DEPS,
        MASK_DRAW_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
        make_mask,
        make_mask_state_json,
        torch,
    )


class _FakeVAE:
    def encode(self, image):
        batch, height, width, _channels = tuple(image.shape)
        latent_height = max(1, int(height) // 8)
        latent_width = max(1, int(width) // 8)
        return torch.zeros((batch, 4, latent_height, latent_width), dtype=image.dtype, device=image.device)


@unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
class TestMaskDrawNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.mask_draw_node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]()
        cls.repair_prepare_node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

    def test_draw_mask_uses_saved_mask_when_state_was_touched(self):
        image = make_image(width=6, height=4, color=0.2)
        input_mask = make_mask(width=6, height=4, value=0.0)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        image_out, mask_out, preview_out = self.mask_draw_node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=state_json,
            node_id="node-1",
            input_mask=input_mask,
        )

        self.assertIs(image_out, image)
        self.assertEqual(tuple(mask_out.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask_out == 1.0))
        self.assertEqual(tuple(preview_out.shape), (1, 4, 6, 3))
        self.assertGreater(float(preview_out[..., 0].mean()), float(image[..., 0].mean()))

    def test_draw_mask_falls_back_to_input_mask_when_state_is_untouched(self):
        image = make_image(width=6, height=4, color=0.2)
        input_mask = make_mask(width=3, height=2, value=1.0)

        _image_out, mask_out, preview_out = self.mask_draw_node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json="{}",
            node_id="node-2",
            input_mask=input_mask,
        )

        self.assertEqual(tuple(mask_out.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask_out == 1.0))
        self.assertEqual(tuple(preview_out.shape), (1, 4, 6, 3))

    def test_draw_mask_returns_black_mask_for_clear_state(self):
        image = make_image(width=6, height=4, color=0.2)
        input_mask = make_mask(width=6, height=4, value=1.0)
        state_json = make_mask_state_json(width=6, height=4, value=0.0, touched=True)

        _image_out, mask_out, preview_out = self.mask_draw_node.draw_mask(
            image=image,
            draw_mode="erase",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=state_json,
            node_id="node-3",
            input_mask=input_mask,
        )

        self.assertTrue(torch.all(mask_out == 0.0))
        self.assertTrue(torch.allclose(preview_out, image))

    def test_draw_mask_output_connects_to_repair_prepare_region_path(self):
        image = make_image(width=64, height=64, color=0.25)
        state_json = make_mask_state_json(width=64, height=64, value=1.0, touched=True)

        image_out, mask_out, preview_out = self.mask_draw_node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=state_json,
            node_id="node-4",
            input_mask=None,
        )

        latent, work_image, work_mask, repair_info, recommended_denoise, positive, negative = self.repair_prepare_node.prepare(
            image=image_out,
            mask=mask_out,
            vae=_FakeVAE(),
            repair_scope="region",
            repair_kernel="vae_inpaint",
            task_hint="repair",
            mask_grow=0,
            mask_blur=0.0,
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
            model_info=None,
            positive=None,
            negative=None,
        )

        self.assertEqual(tuple(preview_out.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(work_image.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(work_mask.shape), (1, 64, 64))
        self.assertEqual(latent["source"], "repair_prepare_region")
        self.assertTrue(repair_info["has_mask"])
        self.assertGreater(repair_info["mask_area_ratio"], 0.0)
        self.assertEqual(recommended_denoise, 0.45)
        self.assertIsNone(positive)
        self.assertIsNone(negative)


if __name__ == "__main__":
    unittest.main()
