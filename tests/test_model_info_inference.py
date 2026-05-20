import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_INFO_PATH = ROOT / "utils" / "model_info.py"


def load_model_info_module():
    spec = importlib.util.spec_from_file_location("lls_model_info_test", MODEL_INFO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FluxClipStub:
    def tokenize(self, _text):
        return {"text": "", "t5xxl": ["token"], "l": ["token"]}


class SdxlClipStub:
    def tokenize(self, _text):
        return {"text": "", "g": ["token"], "l": ["token"]}


class Sd15ClipStub:
    def tokenize(self, _text):
        return {"text": ""}


class BrokenClipStub:
    def tokenize(self, _text):
        raise RuntimeError("tokenizer unavailable")


class ModelWithFamily:
    _lls_family = "FLUX"


class ModelWithType:
    model_type = "flux-dev"


class LatentFormatStub:
    def __init__(self, latent_channels):
        self.latent_channels = latent_channels


class ModelWithLatentFormat:
    def __init__(self, latent_channels):
        self.latent_channels = latent_channels

    def get_model_object(self, name):
        if name != "latent_format":
            raise AssertionError(f"unexpected model object: {name}")
        return LatentFormatStub(self.latent_channels)


class PlainModel:
    pass


class TestModelInfoInference(unittest.TestCase):
    def test_infer_family_from_clip_uses_tokenizer_shape(self):
        model_info = load_model_info_module()

        self.assertEqual(model_info.infer_family_from_clip(FluxClipStub()), "FLUX_DEV")
        self.assertEqual(model_info.infer_family_from_clip(SdxlClipStub()), "SDXL")
        self.assertEqual(model_info.infer_family_from_clip(Sd15ClipStub()), "SD1.5")
        self.assertEqual(model_info.infer_family_from_clip(BrokenClipStub()), "SD1.5")

    def test_infer_family_from_model_prioritizes_lls_family_marker(self):
        model_info = load_model_info_module()

        self.assertEqual(model_info.infer_family_from_model(ModelWithFamily()), "FLUX_DEV")

    def test_infer_family_from_model_falls_back_to_model_type_and_latent_format(self):
        model_info = load_model_info_module()

        self.assertEqual(model_info.infer_family_from_model(ModelWithType()), "FLUX_DEV")
        self.assertEqual(model_info.infer_family_from_model(ModelWithLatentFormat(128)), "FLUX_DEV")
        self.assertEqual(model_info.infer_family_from_model(ModelWithLatentFormat(4)), "SD1.5")
        self.assertEqual(model_info.infer_family_from_model(PlainModel()), "SD1.5")

    def test_infer_task_mode_from_latent_uses_source_field(self):
        model_info = load_model_info_module()

        self.assertEqual(model_info.infer_task_mode_from_latent({"source": "image_encode"}), "img2img")
        self.assertEqual(model_info.infer_task_mode_from_latent({"source": "empty_latent"}), "txt2img")
        self.assertEqual(model_info.infer_task_mode_from_latent({}), "txt2img")
        self.assertEqual(model_info.infer_task_mode_from_latent(None), "txt2img")


if __name__ == "__main__":
    unittest.main()
