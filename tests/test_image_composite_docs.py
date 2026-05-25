import unittest

try:
    from .test_image_composite_helpers import ROOT
except ImportError:
    from test_image_composite_helpers import ROOT


class TestImageCompositeDocs(unittest.TestCase):
    def test_readme_documents_image_composite_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Image Composite", readme)
        self.assertIn("background_image", readme)
        self.assertIn("overlay_image", readme)
        self.assertIn("output_image", readme)
        self.assertIn("x_offset", readme)
        self.assertIn("y_offset", readme)
        self.assertIn("anchor_mode", readme)
        self.assertIn("rotation_origin_mode", readme)
        self.assertIn("opacity", readme)
        self.assertIn("scale", readme)
        self.assertIn("rotation", readme)
        self.assertIn("keep_aspect", readme)


if __name__ == "__main__":
    unittest.main()
