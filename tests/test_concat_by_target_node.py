import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package, torch
except ImportError:
    from test_mask_draw_helpers import load_plugin_package, torch


HAS_CONCAT_RUNTIME_DEPS = torch is not None
CONCAT_RUNTIME_DEPS_MESSAGE = "concat node tests require torch"


@unittest.skipUnless(HAS_CONCAT_RUNTIME_DEPS, CONCAT_RUNTIME_DEPS_MESSAGE)
class TestConcatByTargetNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.node = plugin.NODE_CLASS_MAPPINGS["LLSConcatByTarget"]()

    def make_image(self, width, height, value, batch=1):
        return torch.full((batch, height, width, 3), float(value), dtype=torch.float32)

    def make_mask(self, width, height, value, batch=1):
        return torch.full((batch, height, width), float(value), dtype=torch.float32)

    def assert_image_output(self, image, mask, width, height, shape):
        self.assertEqual((width, height), (shape[2], shape[1]))
        self.assertEqual(tuple(image.shape), shape)
        self.assertEqual(tuple(mask.shape), (shape[0], shape[1], shape[2]))
        self.assertGreaterEqual(float(image.min().item()), 0.0)
        self.assertLessEqual(float(image.max().item()), 1.0)
        self.assertTrue(torch.allclose(mask, torch.zeros_like(mask)))

    def test_image_mode_target_b_right_places_a_to_the_right_of_b(self):
        image_a = self.make_image(width=2, height=2, value=0.2)
        image_b = self.make_image(width=3, height=2, value=0.8)

        image, mask, width, height = self.node.concat(
            data_type="IMAGE",
            target="B",
            position="right",
            match_target_size=True,
            resize_mode="none",
            align="center",
            gap=1,
            background_color="#000000",
            background_value=0.0,
            multiple_of=0,
            allow_batch_broadcast=True,
            image_a=image_a,
            image_b=image_b,
            mask_a=None,
            mask_b=None,
        )

        self.assert_image_output(image, mask, width, height, (1, 2, 6, 3))
        self.assertTrue(torch.allclose(image[:, :, 0:3, :], torch.full((1, 2, 3, 3), 0.8)))
        self.assertTrue(torch.allclose(image[:, :, 3:4, :], torch.zeros((1, 2, 1, 3))))
        self.assertTrue(torch.allclose(image[:, :, 4:6, :], torch.full((1, 2, 2, 3), 0.2)))

    def test_image_mode_keep_proportion_gap_and_multiple_of(self):
        image_a = self.make_image(width=4, height=4, value=0.25)
        image_b = self.make_image(width=2, height=2, value=0.75)

        image, mask, width, height = self.node.concat(
            data_type="IMAGE",
            target="A",
            position="right",
            match_target_size=True,
            resize_mode="keep_proportion",
            align="center",
            gap=1,
            background_color="#808080",
            background_value=0.0,
            multiple_of=4,
            allow_batch_broadcast=True,
            image_a=image_a,
            image_b=image_b,
            mask_a=None,
            mask_b=None,
        )

        self.assert_image_output(image, mask, width, height, (1, 4, 12, 3))
        self.assertTrue(torch.allclose(image[:, :, 0:4, :], torch.full((1, 4, 4, 3), 0.25)))
        self.assertTrue(torch.allclose(image[:, :, 4:5, :], torch.full((1, 4, 1, 3), 128.0 / 255.0)))
        self.assertTrue(torch.allclose(image[:, :, 5:9, :], torch.full((1, 4, 4, 3), 0.75)))
        self.assertTrue(torch.allclose(image[:, :, 9:12, :], torch.full((1, 4, 3, 3), 128.0 / 255.0)))

    def test_mask_mode_outputs_mask_and_preview_image(self):
        mask_a = self.make_mask(width=2, height=2, value=1.0)
        mask_b = self.make_mask(width=2, height=2, value=0.0)

        image, mask, width, height = self.node.concat(
            data_type="MASK",
            target="A",
            position="bottom",
            match_target_size=True,
            resize_mode="none",
            align="center",
            gap=1,
            background_color="#000000",
            background_value=0.25,
            multiple_of=0,
            allow_batch_broadcast=True,
            image_a=None,
            image_b=None,
            mask_a=mask_a,
            mask_b=mask_b,
        )

        self.assertEqual((width, height), (2, 5))
        self.assertEqual(tuple(mask.shape), (1, 5, 2))
        self.assertEqual(tuple(image.shape), (1, 5, 2, 3))
        self.assertTrue(torch.allclose(mask[:, 0:2, :], torch.full((1, 2, 2), 1.0)))
        self.assertTrue(torch.allclose(mask[:, 2:3, :], torch.full((1, 1, 2), 0.25)))
        self.assertTrue(torch.allclose(mask[:, 3:5, :], torch.zeros((1, 2, 2))))
        self.assertTrue(torch.allclose(image[..., 0], mask))
        self.assertTrue(torch.allclose(image[..., 1], mask))
        self.assertTrue(torch.allclose(image[..., 2], mask))

    def test_batch_broadcast_works_when_enabled(self):
        image_a = self.make_image(width=2, height=2, value=0.1, batch=1)
        image_b = self.make_image(width=2, height=2, value=0.9, batch=2)

        image, mask, width, height = self.node.concat(
            data_type="IMAGE",
            target="A",
            position="left",
            match_target_size=True,
            resize_mode="none",
            align="center",
            gap=0,
            background_color="#000000",
            background_value=0.0,
            multiple_of=0,
            allow_batch_broadcast=True,
            image_a=image_a,
            image_b=image_b,
            mask_a=None,
            mask_b=None,
        )

        self.assert_image_output(image, mask, width, height, (2, 2, 4, 3))
        self.assertTrue(torch.allclose(image[:, :, 0:2, :], torch.full((2, 2, 2, 3), 0.9)))
        self.assertTrue(torch.allclose(image[:, :, 2:4, :], torch.full((2, 2, 2, 3), 0.1)))

    def test_gapless_concat_keeps_output_compact_when_sizes_differ(self):
        image_a = self.make_image(width=2, height=2, value=0.1)
        image_b = self.make_image(width=3, height=4, value=0.9)

        image, mask, width, height = self.node.concat(
            data_type="IMAGE",
            target="A",
            position="right",
            match_target_size=False,
            resize_mode="none",
            align="center",
            gap=0,
            background_color="#000000",
            background_value=0.0,
            multiple_of=0,
            allow_batch_broadcast=True,
            image_a=image_a,
            image_b=image_b,
            mask_a=None,
            mask_b=None,
        )

        self.assert_image_output(image, mask, width, height, (1, 2, 5, 3))
        self.assertTrue(torch.allclose(image[:, :, 0:2, :], torch.full((1, 2, 2, 3), 0.1)))
        self.assertTrue(torch.allclose(image[:, :, 2:5, :], torch.full((1, 2, 3, 3), 0.9)))

    def test_missing_required_inputs_for_current_mode_raise_clear_errors(self):
        with self.assertRaisesRegex(RuntimeError, "image_a and image_b"):
            self.node.concat(
                data_type="IMAGE",
                target="A",
                position="right",
                match_target_size=True,
                resize_mode="none",
                align="center",
                gap=0,
                background_color="#000000",
                background_value=0.0,
                multiple_of=0,
                allow_batch_broadcast=True,
                image_a=None,
                image_b=None,
                mask_a=self.make_mask(1, 1, 0.0),
                mask_b=self.make_mask(1, 1, 0.0),
            )

        with self.assertRaisesRegex(RuntimeError, "mask_a and mask_b"):
            self.node.concat(
                data_type="MASK",
                target="A",
                position="right",
                match_target_size=True,
                resize_mode="none",
                align="center",
                gap=0,
                background_color="#000000",
                background_value=0.0,
                multiple_of=0,
                allow_batch_broadcast=True,
                image_a=self.make_image(1, 1, 0.0),
                image_b=self.make_image(1, 1, 0.0),
                mask_a=None,
                mask_b=None,
            )

    def test_invalid_background_color_and_batch_mismatch_raise_clear_errors(self):
        image_a = self.make_image(width=2, height=2, value=0.1, batch=2)
        image_b = self.make_image(width=2, height=2, value=0.9, batch=3)

        with self.assertRaisesRegex(RuntimeError, "background_color"):
            self.node.concat(
                data_type="IMAGE",
                target="A",
                position="right",
                match_target_size=True,
                resize_mode="none",
                align="center",
                gap=0,
                background_color="bad-color",
                background_value=0.0,
                multiple_of=0,
                allow_batch_broadcast=True,
                image_a=self.make_image(2, 2, 0.1),
                image_b=self.make_image(2, 2, 0.9),
                mask_a=None,
                mask_b=None,
            )

        with self.assertRaisesRegex(RuntimeError, "Batch size mismatch"):
            self.node.concat(
                data_type="IMAGE",
                target="A",
                position="right",
                match_target_size=True,
                resize_mode="none",
                align="center",
                gap=0,
                background_color="#000000",
                background_value=0.0,
                multiple_of=0,
                allow_batch_broadcast=False,
                image_a=image_a,
                image_b=image_b,
                mask_a=None,
                mask_b=None,
            )


if __name__ == "__main__":
    unittest.main()
