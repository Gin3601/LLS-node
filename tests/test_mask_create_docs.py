import unittest

try:
    from .test_mask_draw_helpers import ROOT
except ImportError:
    from test_mask_draw_helpers import ROOT


class TestMaskCreateDocs(unittest.TestCase):
    def test_readme_documents_mask_create_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Mask Create", readme)
        self.assertNotIn("LLS Simple Mask Preview", readme)
        self.assertIn("LLS Save Image.mask", readme)
        self.assertIn("image_width", readme)
        self.assertIn("image_height", readme)
        self.assertIn("shape_type", readme)
        self.assertIn("coordinate_mode", readme)
        self.assertIn("area_info", readme)
        self.assertIn("LLS Simple Mask Create.mask -> LLS Simple Repair Prepare.mask", readme)
        self.assertIn("LLS Simple Mask Create.mask_image -> Preview Image", readme)
        self.assertIn("LLS Simple Mask Create.mask -> LLS Save Image.mask", readme)
        self.assertIn("LLS Simple Mask Draw.mask -> LLS Save Image.mask", readme)

    def test_readme_lists_supported_shapes_and_combine_modes(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("rectangle", readme)
        self.assertIn("square", readme)
        self.assertIn("circle", readme)
        self.assertIn("ellipse", readme)
        self.assertIn("union", readme)
        self.assertIn("subtract", readme)
        self.assertIn("intersect", readme)


if __name__ == "__main__":
    unittest.main()
