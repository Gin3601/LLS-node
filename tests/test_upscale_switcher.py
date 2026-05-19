import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_node_test_upscale",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test_upscale"] = module
    spec.loader.exec_module(module)
    return module


class RecordingUpscaleSwitcher:
    def __init__(self, cls):
        self._impl = cls()
        self.calls = []

    def _upscale_with_pytorch(self, image, scale, interpolation):
        self.calls.append(("pytorch", image, scale, interpolation))
        return ("PYTORCH_RESULT",)

    def _upscale_with_model(self, image, model_name, tile, overlap):
        self.calls.append(("model", image, model_name, tile, overlap))
        return ("MODEL_RESULT",)


class TestUpscaleSwitcher(unittest.TestCase):
    def test_schema_defaults_to_pytorch_when_no_upscale_models_exist(self):
        load_plugin_package()
        from lls_node_test_upscale.upscale import nodes as upscale_nodes

        with mock.patch.object(
            upscale_nodes,
            "_get_upscale_model_names",
            return_value=[upscale_nodes.NO_UPSCALE_MODEL_PLACEHOLDER],
        ):
            schema = upscale_nodes.LLSUpscaleSwitcher.INPUT_TYPES()

        self.assertEqual(schema["required"]["mode"][1]["default"], "pytorch")

    def test_placeholder_model_name_falls_back_to_pytorch(self):
        load_plugin_package()
        from lls_node_test_upscale.upscale import nodes as upscale_nodes

        recorder = RecordingUpscaleSwitcher(upscale_nodes.LLSUpscaleSwitcher)
        with mock.patch.object(
            upscale_nodes.LLSUpscaleSwitcher,
            "_upscale_with_pytorch",
            new=RecordingUpscaleSwitcher._upscale_with_pytorch,
        ), mock.patch.object(
            upscale_nodes.LLSUpscaleSwitcher,
            "_upscale_with_model",
            new=RecordingUpscaleSwitcher._upscale_with_model,
        ):
            result = upscale_nodes.LLSUpscaleSwitcher.upscale(
                recorder,
                image="image",
                mode="upscale_model",
                scale=2.0,
                interpolation="bilinear",
                model_name=upscale_nodes.NO_UPSCALE_MODEL_PLACEHOLDER,
                tile=512,
                overlap=32,
            )

        self.assertEqual(result, ("PYTORCH_RESULT",))
        self.assertEqual(
            recorder.calls,
            [("pytorch", "image", 2.0, "bilinear")],
        )


if __name__ == "__main__":
    unittest.main()
