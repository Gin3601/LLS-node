import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_node_test_compat",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test_compat"] = module
    spec.loader.exec_module(module)
    return module


class LegacyClipStub:
    def __init__(self):
        self.layer = None

    def clone(self):
        cloned = LegacyClipStub()
        cloned.layer = self.layer
        return cloned

    def clip_layer(self, layer):
        self.layer = layer

    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens(self, tokens, return_pooled=False):
        if not return_pooled:
            raise AssertionError("legacy path should request pooled output")
        cond = f"cond::{tokens['text']}"
        pooled = f"pooled::{tokens['text']}"
        return cond, pooled


class UniversalClipStub:
    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens(self, tokens, return_pooled=False):
        if not return_pooled:
            raise AssertionError("legacy path should request pooled output")
        return f"cond::{tokens['text']}", f"pooled::{tokens['text']}"


class TestConditioningCompatibility(unittest.TestCase):
    def test_simple_prompt_encode_schema_accepts_legacy_none_clip_skip(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        clip_skip_type, clip_skip_config = node_cls.INPUT_TYPES()["required"]["clip_skip"]

        self.assertIsInstance(clip_skip_type, list)
        self.assertIn(None, clip_skip_type)
        self.assertEqual(clip_skip_config["default"], -1)

    def test_simple_prompt_encode_falls_back_for_legacy_clip(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()

        positive, negative, prompt_info = node.encode(
            LegacyClipStub(),
            "a cat",
            "low quality",
            -2,
        )

        self.assertEqual(
            positive,
            [["cond::a cat", {"pooled_output": "pooled::a cat"}]],
        )
        self.assertEqual(
            negative,
            [["cond::low quality", {"pooled_output": "pooled::low quality"}]],
        )
        self.assertEqual(json.loads(prompt_info)["clip_skip"], -2)

    def test_simple_prompt_encode_normalizes_none_clip_skip(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()

        positive, negative, prompt_info = node.encode(
            LegacyClipStub(),
            "a cat",
            "low quality",
            None,
        )

        self.assertEqual(
            positive,
            [["cond::a cat", {"pooled_output": "pooled::a cat"}]],
        )
        self.assertEqual(
            negative,
            [["cond::low quality", {"pooled_output": "pooled::low quality"}]],
        )
        self.assertEqual(json.loads(prompt_info)["clip_skip"], -1)

    def test_universal_backend_falls_back_for_legacy_clip(self):
        load_plugin_package()
        from lls_node_test_compat.lls_universal.backend_sd15 import SD15Backend

        backend = SD15Backend()
        positive = backend._encode_standard_prompt(UniversalClipStub(), "hello")

        self.assertEqual(
            positive,
            [["cond::hello", {"pooled_output": "pooled::hello"}]],
        )


if __name__ == "__main__":
    unittest.main()
