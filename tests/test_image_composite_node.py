import unittest

try:
    from .test_image_composite_helpers import (
        HAS_IMAGE_COMPOSITE_RUNTIME_DEPS,
        IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
    )
except ImportError:
    from test_image_composite_helpers import (
        HAS_IMAGE_COMPOSITE_RUNTIME_DEPS,
        IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
    )


@unittest.skipUnless(HAS_IMAGE_COMPOSITE_RUNTIME_DEPS, IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE)
class TestImageCompositeNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleImageComposite"]()

    def test_top_left_offset_places_overlay_at_requested_position(self):
        background = make_image(width=6, height=6, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertEqual(tuple(output_image.shape), (1, 6, 6, 3))
        self.assertTrue((output_image[0, 1:3, 3:5] > 0.99).all().item())
        self.assertEqual(float(output_image[0, 0, 0, 0].item()), 0.0)

    def test_center_anchor_places_overlay_by_its_center(self):
        background = make_image(width=6, height=6, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(0.0, 1.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=2,
            anchor_mode="center",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image[0, 1:3, 2:4, 1] > 0.99).all().item())

    def test_partial_overflow_is_clipped_to_background_bounds(self):
        background = make_image(width=5, height=5, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=3, height=3, color=(1.0, 0.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=4,
            y_offset=4,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertGreater(float(output_image[0, 4, 4, 0].item()), 0.99)
        self.assertEqual(float(output_image[0, 0, 0, 0].item()), 0.0)

    def test_overlay_fully_outside_returns_background(self):
        background = make_image(width=5, height=5, color=(0.1, 0.1, 0.1))
        overlay = make_image(width=2, height=2, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=8,
            y_offset=8,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image == background).all().item())

    def test_opacity_zero_returns_background(self):
        background = make_image(width=4, height=4, color=(0.2, 0.2, 0.2))
        overlay = make_image(width=2, height=2, color=(1.0, 0.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=0.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image == background).all().item())

    def test_rgba_overlay_uses_alpha_and_opacity(self):
        background = make_image(width=4, height=4, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(1.0, 0.0, 0.0), alpha=0.5)

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=0.5,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertAlmostEqual(float(output_image[0, 1, 1, 0].item()), 0.25, places=2)

    def test_scale_expands_overlay_footprint(self):
        background = make_image(width=5, height=5, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=1, height=1, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=2.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image[0, 1:3, 1:3] > 0.99).all().item())

    def test_rotation_origin_mode_changes_rotated_result(self):
        background = make_image(width=8, height=8, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=3, height=1, color=(0.0, 0.0, 1.0))

        (center_rotated,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=3,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=90.0,
            keep_aspect=True,
        )
        (corner_rotated,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=3,
            anchor_mode="top_left",
            rotation_origin_mode="top_left",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=90.0,
            keep_aspect=True,
        )

        self.assertFalse((center_rotated == corner_rotated).all().item())


if __name__ == "__main__":
    unittest.main()
