import unittest

try:
    from .test_mask_draw_helpers import ROOT, load_plugin_package
except ImportError:
    from test_mask_draw_helpers import ROOT, load_plugin_package


class TestConcatByTargetFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_concat_by_target.js"
        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_concat_by_target_extension(self):
        asset = (ROOT / "web" / "js" / "lls_concat_by_target.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSConcatByTarget", asset)
        self.assertIn("LLS Concat By Target", asset)
        self.assertIn("image/mask_A", asset)
        self.assertIn("image/mask_B", asset)
        self.assertIn("beforeRegisterNodeDef", asset)
        self.assertIn("onNodeCreated", asset)
        self.assertIn("onGraphConfigured", asset)
        self.assertIn("onConnectionsChange", asset)
        self.assertIn("input.label", asset)


if __name__ == "__main__":
    unittest.main()
