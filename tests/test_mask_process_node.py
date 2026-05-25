import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package, torch
except ImportError:
    from test_mask_draw_helpers import load_plugin_package, torch


HAS_MASK_PROCESS_RUNTIME_DEPS = torch is not None
MASK_PROCESS_RUNTIME_DEPS_MESSAGE = "mask processing node tests require torch"


def _require_mask_process_runtime_deps():
    if not HAS_MASK_PROCESS_RUNTIME_DEPS:
        raise unittest.SkipTest(MASK_PROCESS_RUNTIME_DEPS_MESSAGE)


@unittest.skipUnless(HAS_MASK_PROCESS_RUNTIME_DEPS, MASK_PROCESS_RUNTIME_DEPS_MESSAGE)
class TestMaskProcessNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.process_node = plugin.NODE_CLASS_MAPPINGS["LLSMaskProcess"]()
        cls.combine_node = plugin.NODE_CLASS_MAPPINGS["LLSMaskCombine"]()

    def assert_is_normalized_mask(self, mask, shape):
        self.assertEqual(tuple(mask.shape), shape)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertGreaterEqual(float(mask.min().item()), 0.0)
        self.assertLessEqual(float(mask.max().item()), 1.0)

    def test_passthrough_normalizes_supported_mask_shapes(self):
        cases = {
            "hw": torch.tensor([[-1.0, 0.4], [1.2, 0.7]], dtype=torch.float64),
            "bhw": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            "bhw1": torch.tensor([[[[0.0], [1.0]], [[1.0], [0.0]]]], dtype=torch.float32),
            "b1hw": torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32),
        }

        for name, source in cases.items():
            with self.subTest(name=name):
                (mask,) = self.process_node.process(
                    mask=source,
                    operation="passthrough",
                    value_float=0.5,
                    value_int=8,
                )
                self.assert_is_normalized_mask(mask, (1, 2, 2))

    def test_threshold_and_invert_produce_expected_values(self):
        source = torch.tensor([[[0.2, 0.5, 0.7]]], dtype=torch.float32)

        (thresholded,) = self.process_node.process(
            mask=source,
            operation="threshold",
            value_float=0.5,
            value_int=8,
        )
        (inverted,) = self.process_node.process(
            mask=source,
            operation="invert",
            value_float=0.5,
            value_int=8,
        )

        self.assertTrue(
            torch.equal(
                thresholded,
                torch.tensor([[[0.0, 1.0, 1.0]]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                inverted,
                torch.tensor([[[0.8, 0.5, 0.3]]], dtype=torch.float32),
            )
        )

    def test_grow_and_shrink_do_not_error_and_keep_normalized_output(self):
        source = torch.zeros((1, 5, 5), dtype=torch.float32)
        source[0, 2, 2] = 1.0

        (grown,) = self.process_node.process(
            mask=source,
            operation="grow",
            value_float=0.5,
            value_int=1,
        )
        (shrunk,) = self.process_node.process(
            mask=grown,
            operation="shrink",
            value_float=0.5,
            value_int=1,
        )

        self.assert_is_normalized_mask(grown, (1, 5, 5))
        self.assert_is_normalized_mask(shrunk, (1, 5, 5))
        self.assertGreater(float(grown.sum().item()), float(source.sum().item()))
        self.assertLessEqual(float(shrunk.sum().item()), float(grown.sum().item()))

    def test_blur_and_feather_do_not_error_and_create_soft_edges(self):
        source = torch.zeros((1, 7, 7), dtype=torch.float32)
        source[:, 2:5, 2:5] = 1.0

        (blurred,) = self.process_node.process(
            mask=source,
            operation="blur",
            value_float=0.5,
            value_int=1,
        )
        (feathered,) = self.process_node.process(
            mask=source,
            operation="feather",
            value_float=0.5,
            value_int=2,
        )

        self.assert_is_normalized_mask(blurred, (1, 7, 7))
        self.assert_is_normalized_mask(feathered, (1, 7, 7))
        self.assertTrue(bool(((blurred > 0.0) & (blurred < 1.0)).any().item()))
        self.assertTrue(bool(((feathered > 0.0) & (feathered < 1.0)).any().item()))

    def test_fill_holes_remove_small_regions_smooth_and_clamp_are_stable(self):
        with_hole = torch.ones((1, 5, 5), dtype=torch.float32)
        with_hole[0, 2, 2] = 0.0

        (filled,) = self.process_node.process(
            mask=with_hole,
            operation="fill_holes",
            value_float=0.5,
            value_int=8,
        )
        self.assertEqual(float(filled[0, 2, 2].item()), 1.0)

        with_small_region = torch.zeros((1, 6, 6), dtype=torch.float32)
        with_small_region[0, 0, 0] = 1.0
        with_small_region[0, 3:5, 3:5] = 1.0
        (removed,) = self.process_node.process(
            mask=with_small_region,
            operation="remove_small_regions",
            value_float=0.5,
            value_int=2,
        )
        self.assertEqual(float(removed[0, 0, 0].item()), 0.0)
        self.assertEqual(float(removed[0, 3, 3].item()), 1.0)

        jagged = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
        (smoothed,) = self.process_node.process(
            mask=jagged,
            operation="smooth",
            value_float=0.5,
            value_int=1,
        )
        (clamped,) = self.process_node.process(
            mask=torch.tensor([[[-2.0, 2.0]]], dtype=torch.float32),
            operation="clamp",
            value_float=0.5,
            value_int=8,
        )

        self.assert_is_normalized_mask(smoothed, (1, 3, 3))
        self.assert_is_normalized_mask(clamped, (1, 1, 2))

    def test_resize_to_image_uses_optional_image_dimensions_and_batch(self):
        source = torch.ones((1, 2, 3), dtype=torch.float32)
        image = torch.zeros((2, 5, 7, 3), dtype=torch.float32)

        (resized,) = self.process_node.process(
            mask=source,
            operation="resize_to_image",
            value_float=0.5,
            value_int=8,
            image=image,
        )

        self.assert_is_normalized_mask(resized, (2, 5, 7))
        self.assertTrue(torch.allclose(resized, torch.ones((2, 5, 7), dtype=torch.float32)))

    def test_mask_combine_modes_produce_expected_values(self):
        mask_a = torch.tensor([[[1.0, 0.5], [0.0, 1.0]]], dtype=torch.float32)
        mask_b = torch.tensor([[[0.5, 1.0], [0.0, 0.25]]], dtype=torch.float32)
        expectations = {
            "add": torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
            "subtract": torch.tensor([[[0.5, 0.0], [0.0, 0.75]]], dtype=torch.float32),
            "intersect": torch.tensor([[[0.5, 0.5], [0.0, 0.25]]], dtype=torch.float32),
            "xor": torch.tensor([[[0.5, 0.5], [0.0, 0.75]]], dtype=torch.float32),
            "max": torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
            "min": torch.tensor([[[0.5, 0.5], [0.0, 0.25]]], dtype=torch.float32),
        }

        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                (combined,) = self.combine_node.combine(mask_a=mask_a, mask_b=mask_b, mode=mode)
                self.assert_is_normalized_mask(combined, (1, 2, 2))
                self.assertTrue(torch.allclose(combined, expected))

    def test_mask_combine_resizes_and_aligns_batch_to_mask_a(self):
        mask_a = torch.full((2, 4, 6), 0.25, dtype=torch.float32)
        mask_b = torch.ones((1, 2, 3), dtype=torch.float32)

        (combined,) = self.combine_node.combine(mask_a=mask_a, mask_b=mask_b, mode="add")

        self.assert_is_normalized_mask(combined, (2, 4, 6))
        self.assertTrue(torch.allclose(combined, torch.ones((2, 4, 6), dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
