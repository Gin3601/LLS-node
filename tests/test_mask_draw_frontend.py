import unittest

try:
    from .test_mask_draw_helpers import ROOT, load_plugin_package
except ImportError:
    from test_mask_draw_helpers import ROOT, load_plugin_package


class TestMaskDrawFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()

        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_mask_draw.js"

        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_mask_draw_extension(self):
        asset = (ROOT / "web" / "js" / "lls_mask_draw.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSSimpleMaskDraw", asset)
        self.assertIn("LLS Simple Mask Draw", asset)
        self.assertIn("mask_state_json", asset)
        self.assertIn("beforeRegisterNodeDef", asset)
        self.assertIn("addDOMWidget", asset)
        self.assertIn("Clear", asset)
        self.assertIn("Undo", asset)
        self.assertIn("Redo", asset)
        self.assertIn("Invert", asset)


if __name__ == "__main__":
    unittest.main()
