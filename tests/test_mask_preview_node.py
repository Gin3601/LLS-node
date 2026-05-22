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


HAS_MASK_PREVIEW_RUNTIME_DEPS = torch is not None
MASK_PREVIEW_RUNTIME_DEPS_MESSAGE = "mask preview node tests require torch"


def _require_mask_preview_runtime_deps():
    if not HAS_MASK_PREVIEW_RUNTIME_DEPS:
        raise unittest.SkipTest(MASK_PREVIEW_RUNTIME_DEPS_MESSAGE)


def make_image(width=8, height=8, color=0.25):
    _require_mask_preview_runtime_deps()
    return torch.full((1, height, width, 3), float(color), dtype=torch.float32)


def make_mask(width=8, height=8, value=0.0):
    _require_mask_preview_runtime_deps()
    return torch.full((1, height, width), float(value), dtype=torch.float32)


@unittest.skipUnless(HAS_MASK_PREVIEW_RUNTIME_DEPS, MASK_PREVIEW_RUNTIME_DEPS_MESSAGE)
class TestMaskPreviewNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskPreview"]()

    def test_preview_overlays_requested_color(self):
        image = make_image(width=20, height=20, color=0.2)
        mask = make_mask(width=20, height=20, value=1.0)

        (preview_image,) = self.node.preview_mask(
            image=image,
            mask=mask,
            overlay_alpha=0.5,
            overlay_color="green",
        )

        center_pixel = preview_image[0, 10, 10]
        self.assertEqual(tuple(preview_image.shape), (1, 20, 20, 3))
        self.assertGreater(float(center_pixel[1].item()), float(center_pixel[0].item()))
        self.assertGreater(float(center_pixel[1].item()), float(center_pixel[2].item()))

    def test_preview_resizes_mask_to_match_image(self):
        image = make_image(width=20, height=20, color=0.2)
        mask = make_mask(width=10, height=10, value=1.0)

        (preview_image,) = self.node.preview_mask(
            image=image,
            mask=mask,
            overlay_alpha=0.4,
            overlay_color="red",
        )

        self.assertEqual(tuple(preview_image.shape), (1, 20, 20, 3))
        self.assertGreater(float(preview_image[..., 0].mean().item()), float(image[..., 0].mean().item()))


if __name__ == "__main__":
    unittest.main()
