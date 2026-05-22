import pathlib
import unittest


README_PATH = pathlib.Path(__file__).resolve().parents[1] / "README.md"


class TestProEditDocs(unittest.TestCase):
    def test_readme_documents_pro_edit_nodes_and_workflow(self):
        text = README_PATH.read_text(encoding="utf-8")

        for needle in (
            "LLS Pro Image Edit Prepare",
            "LLS Pro KSampler Bridge",
            "LLS Pro Image Edit Finish",
            "Simple = lightweight masked latent resampling",
            "Pro = true image edit / inpaint pipeline",
            "backend_mode = auto | sdxl | flux",
            "profile-driven routing",
            "profile_id",
            "backend_type",
            "sampler_strategy",
            "execution_path",
            "LLS Simple Checkpoint Loader writes the resolved model profile",
            "base profiles automatically fall back to a generic local repaint path",
            "Load Image -> Load Mask or LLS Simple Mask Draw -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Pro Image Edit Prepare -> LLS Pro KSampler Bridge -> VAE Decode -> LLS Pro Image Edit Finish -> Preview Image",
            "Adding new professional edit models",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
