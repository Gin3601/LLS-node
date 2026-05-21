import unittest

try:
    from .test_mask_draw_helpers import ROOT
except ImportError:
    from test_mask_draw_helpers import ROOT


class TestMaskDrawDocs(unittest.TestCase):
    def test_readme_documents_mask_draw_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Mask Draw", readme)
        self.assertIn("Load Image", readme)
        self.assertIn("LLS Simple Repair Prepare", readme)
        self.assertIn("preview_image", readme)
        self.assertIn("brush", readme)
        self.assertIn("erase", readme)

    def test_readme_lists_recommended_use_cases(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("手动指定删除区域", readme)
        self.assertIn("手动指定修复区域", readme)
        self.assertIn("手动指定去阴影区域", readme)
        self.assertIn("手动指定局部增强区域", readme)


if __name__ == "__main__":
    unittest.main()
