import unittest

try:
    from .test_mask_draw_helpers import ROOT
except ImportError:
    from test_mask_draw_helpers import ROOT


class TestMaskCreateDocs(unittest.TestCase):
    def test_readme_documents_mask_create_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Mask Create", readme)
        self.assertIn("shape_type", readme)
        self.assertIn("coordinate_mode", readme)
        self.assertIn("area_info", readme)
        self.assertIn("Load Image -> LLS Simple Mask Create -> LLS Simple Repair Prepare", readme)
        self.assertIn("Load Image -> LLS Simple Mask Create -> LLS Simple Mask Draw", readme)

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
