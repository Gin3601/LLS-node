import unittest

try:
    from .test_mask_draw_helpers import (
        load_plugin_package,
        torch,
    )
except ImportError:
    from test_mask_draw_helpers import (
        load_plugin_package,
        torch,
    )


HAS_MASK_CREATE_RUNTIME_DEPS = torch is not None
MASK_CREATE_RUNTIME_DEPS_MESSAGE = "mask create node tests require torch"


def _require_mask_create_runtime_deps():
    if not HAS_MASK_CREATE_RUNTIME_DEPS:
        raise unittest.SkipTest(MASK_CREATE_RUNTIME_DEPS_MESSAGE)


def make_image(width=8, height=8, color=0.25):
    _require_mask_create_runtime_deps()
    return torch.full((1, height, width, 3), float(color), dtype=torch.float32)


def make_mask(width=8, height=8, value=0.0):
    _require_mask_create_runtime_deps()
    return torch.full((1, height, width), float(value), dtype=torch.float32)


class _FakeVAE:
    def encode(self, image):
        batch, height, width, _channels = tuple(image.shape)
        latent_height = max(1, int(height) // 8)
        latent_width = max(1, int(width) // 8)
        return torch.zeros((batch, 4, latent_height, latent_width), dtype=image.dtype, device=image.device)


@unittest.skipUnless(HAS_MASK_CREATE_RUNTIME_DEPS, MASK_CREATE_RUNTIME_DEPS_MESSAGE)
class TestMaskCreateNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskCreate"]()
        cls.repair_prepare_node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()

    def _create(self, **overrides):
        params = {
            "image_width": 100,
            "image_height": 80,
            "shape_type": "rectangle",
            "coordinate_mode": "percent",
            "center_x": 0.5,
            "center_y": 0.5,
            "width": 0.3,
            "height": 0.25,
            "radius": 0.15,
            "feather": 0.0,
            "blur": 0.0,
            "invert_mask": False,
            "combine_mode": "replace",
            "input_mask": None,
        }
        params.update(overrides)
        return self.node.create_mask(**params)

    def test_rectangle_percent_mode_creates_expected_bbox_and_area_info(self):
        mask_image, mask, area_info = self._create()

        self.assertEqual(tuple(mask.shape), (1, 80, 100))
        self.assertEqual(tuple(mask_image.shape), (1, 80, 100, 3))
        self.assertEqual(area_info["image_size"], [100, 80])
        self.assertEqual(area_info["shape_type"], "rectangle")
        self.assertEqual(area_info["coordinate_mode"], "percent")
        self.assertEqual(area_info["bbox"], [35, 30, 65, 50])
        self.assertEqual(area_info["width"], 30.0)
        self.assertEqual(area_info["height"], 20.0)
        self.assertIsNone(area_info["radius"])
        self.assertEqual(area_info["binary_area_px"], 600)
        self.assertAlmostEqual(area_info["effective_area_px"], 600.0, places=4)
        self.assertAlmostEqual(area_info["area_ratio"], 600.0 / 8000.0, places=4)
        self.assertEqual(float(mask[0, 40, 50].item()), 1.0)
        self.assertEqual(float(mask[0, 5, 5].item()), 0.0)
        self.assertEqual(float(mask_image[0, 40, 50, 0].item()), 1.0)
        self.assertEqual(float(mask_image[0, 5, 5, 0].item()), 0.0)

    def test_square_pixel_mode_uses_width_as_side(self):
        _mask_image, mask, area_info = self.node.create_mask(
            image_width=64,
            image_height=64,
            shape_type="square",
            coordinate_mode="pixel",
            center_x=20.0,
            center_y=30.0,
            width=10.0,
            height=999.0,
            radius=5.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="replace",
            input_mask=None,
        )

        self.assertEqual(area_info["bbox"], [15, 25, 25, 35])
        self.assertEqual(area_info["binary_area_px"], 100)
        self.assertEqual(area_info["height"], 10.0)
        self.assertEqual(float(mask[0, 30, 20].item()), 1.0)
        self.assertEqual(float(mask[0, 5, 5].item()), 0.0)

    def test_circle_generates_centered_mask(self):
        _mask_image, mask, area_info = self.node.create_mask(
            image_width=101,
            image_height=101,
            shape_type="circle",
            coordinate_mode="pixel",
            center_x=50.0,
            center_y=50.0,
            width=20.0,
            height=20.0,
            radius=10.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="replace",
            input_mask=None,
        )

        self.assertEqual(area_info["radius"], 10.0)
        self.assertGreater(area_info["binary_area_px"], 250)
        self.assertLess(area_info["binary_area_px"], 380)
        self.assertEqual(float(mask[0, 50, 50].item()), 1.0)
        self.assertEqual(float(mask[0, 10, 10].item()), 0.0)

    def test_ellipse_generates_nonempty_mask(self):
        _mask_image, mask, area_info = self.node.create_mask(
            image_width=120,
            image_height=90,
            shape_type="ellipse",
            coordinate_mode="pixel",
            center_x=60.0,
            center_y=45.0,
            width=40.0,
            height=20.0,
            radius=10.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="replace",
            input_mask=None,
        )

        self.assertEqual(area_info["width"], 40.0)
        self.assertEqual(area_info["height"], 20.0)
        self.assertGreater(area_info["binary_area_px"], 500)
        self.assertEqual(float(mask[0, 45, 60].item()), 1.0)
        self.assertEqual(float(mask[0, 5, 5].item()), 0.0)

    def test_union_subtract_and_intersect_with_input_mask(self):
        input_mask = make_mask(width=32, height=32, value=1.0)

        _mask_image, union_mask, union_info = self.node.create_mask(
            image_width=32,
            image_height=32,
            shape_type="rectangle",
            coordinate_mode="pixel",
            center_x=16.0,
            center_y=16.0,
            width=10.0,
            height=10.0,
            radius=5.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="union",
            input_mask=input_mask,
        )
        self.assertEqual(union_info["binary_area_px"], 32 * 32)
        self.assertTrue(torch.all(union_mask == 1.0))

        _mask_image, subtract_mask, subtract_info = self.node.create_mask(
            image_width=32,
            image_height=32,
            shape_type="rectangle",
            coordinate_mode="pixel",
            center_x=16.0,
            center_y=16.0,
            width=10.0,
            height=10.0,
            radius=5.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="subtract",
            input_mask=input_mask,
        )
        self.assertLess(subtract_info["binary_area_px"], 32 * 32)
        self.assertEqual(float(subtract_mask[0, 16, 16].item()), 0.0)

        _mask_image, intersect_mask, intersect_info = self.node.create_mask(
            image_width=32,
            image_height=32,
            shape_type="rectangle",
            coordinate_mode="pixel",
            center_x=16.0,
            center_y=16.0,
            width=10.0,
            height=10.0,
            radius=5.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="intersect",
            input_mask=input_mask,
        )
        self.assertEqual(intersect_info["binary_area_px"], 100)
        self.assertEqual(float(intersect_mask[0, 16, 16].item()), 1.0)
        self.assertEqual(float(intersect_mask[0, 1, 1].item()), 0.0)

    def test_invert_mask_flips_final_mask(self):
        _mask_image, mask, area_info = self._create(invert_mask=True)

        self.assertEqual(area_info["invert_mask"], True)
        self.assertEqual(float(mask[0, 40, 50].item()), 0.0)
        self.assertEqual(float(mask[0, 5, 5].item()), 1.0)

    def test_feather_and_blur_create_soft_mask(self):
        _mask_image, mask, area_info = self._create(
            shape_type="rectangle",
            coordinate_mode="pixel",
            center_x=50.0,
            center_y=40.0,
            width=20.0,
            height=20.0,
            feather=4.0,
            blur=3.0,
        )

        self.assertGreater(area_info["effective_area_px"], 0.0)
        self.assertLess(area_info["effective_area_px"], 8000.0)
        self.assertTrue(bool(((mask > 0.0) & (mask < 1.0)).any().item()))

    def test_shape_outside_image_returns_black_mask_without_crashing(self):
        _mask_image, mask, area_info = self._create(
            shape_type="circle",
            coordinate_mode="percent",
            center_x=2.0,
            center_y=2.0,
            radius=0.1,
        )

        self.assertEqual(area_info["binary_area_px"], 0)
        self.assertEqual(area_info["area_ratio"], 0.0)
        self.assertTrue(torch.all(mask == 0.0))

    def test_input_mask_is_resized_and_output_connects_to_repair_prepare(self):
        image = make_image(width=64, height=64, color=0.25)
        input_mask = make_mask(width=16, height=16, value=1.0)

        _mask_image, mask_out, area_info = self.node.create_mask(
            image_width=64,
            image_height=64,
            shape_type="rectangle",
            coordinate_mode="pixel",
            center_x=32.0,
            center_y=32.0,
            width=10.0,
            height=10.0,
            radius=5.0,
            feather=0.0,
            blur=0.0,
            invert_mask=False,
            combine_mode="union",
            input_mask=input_mask,
        )

        latent, work_image, work_mask, repair_info, recommended_denoise, positive, negative = self.repair_prepare_node.prepare(
            image=image,
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

        self.assertEqual(tuple(mask_out.shape), (1, 64, 64))
        self.assertEqual(tuple(work_image.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(work_mask.shape), (1, 64, 64))
        self.assertEqual(latent["source"], "repair_prepare_region")
        self.assertTrue(repair_info["has_mask"])
        self.assertGreater(area_info["binary_area_px"], 0)
        self.assertEqual(recommended_denoise, 0.45)
        self.assertIsNone(positive)
        self.assertIsNone(negative)


if __name__ == "__main__":
    unittest.main()
