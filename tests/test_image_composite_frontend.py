import unittest

try:
    from .test_image_composite_helpers import ROOT, load_plugin_package
except ImportError:
    from test_image_composite_helpers import ROOT, load_plugin_package


class TestImageCompositeFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_image_composite.js"
        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_image_composite_extension(self):
        asset = (ROOT / "web" / "js" / "lls_image_composite.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSSimpleImageComposite", asset)
        self.assertIn("LLS Simple Image Composite", asset)
        self.assertIn("beforeRegisterNodeDef", asset)
        self.assertIn("addDOMWidget", asset)
        self.assertIn("pointerdown", asset)
        self.assertIn("pointermove", asset)
        self.assertIn("rotation_origin_mode", asset)


if __name__ == "__main__":
    unittest.main()
