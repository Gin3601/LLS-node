import unittest

try:
    from .test_mask_draw_helpers import ROOT
except ImportError:
    from test_mask_draw_helpers import ROOT


class TestSamplingFrontendCompat(unittest.TestCase):
    def test_ksampler_compat_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_ksampler_compat.js"

        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_ksampler_compat_asset_targets_sampler_node(self):
        asset = (ROOT / "web" / "js" / "lls_ksampler_compat.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSSimpleKSampler", asset)
        self.assertIn("LLS Simple KSampler", asset)
        self.assertIn("widgets_values", asset)
        self.assertIn("configure", asset)


if __name__ == "__main__":
    unittest.main()
