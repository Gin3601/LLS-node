import unittest
from unittest import mock

try:
    from .test_pro_edit_helpers import import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import import_plugin_submodule, load_plugin_package


class _Expr:
    def __init__(self, text):
        self.text = text

    def clamp(self, _min_value, _max_value):
        return self

    def __mul__(self, other):
        other_text = getattr(other, "text", repr(other))
        return _Expr(f"({self.text}*{other_text})")

    def __rmul__(self, other):
        other_text = getattr(other, "text", repr(other))
        return _Expr(f"({other_text}*{self.text})")

    def __add__(self, other):
        other_text = getattr(other, "text", repr(other))
        return _Expr(f"({self.text}+{other_text})")

    def __rsub__(self, other):
        other_text = getattr(other, "text", repr(other))
        return _Expr(f"({other_text}-{self.text})")


class _FakeTorchTensor:
    def __init__(self, label):
        self.label = label
        self.text = label

    def masked_fill(self, _mask, _fill_value):
        raise AssertionError("torch tensor branch should not call masked_fill directly")

    def __mul__(self, other):
        other_text = getattr(other, "text", repr(other))
        return _Expr(f"({self.label}*{other_text})")


class _FakeMask:
    def __init__(self, label):
        self.label = label

    def unsqueeze(self, dim):
        return _Expr(f"{self.label}.unsqueeze({dim})")


class _FakeTorchModule:
    Tensor = _FakeTorchTensor


class _FallbackTensor:
    def __init__(self):
        self.called = False

    def masked_fill(self, _mask, _fill_value):
        self.called = True
        return "fallback-masked"


class TestProEditUtils(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        self.utils = import_plugin_submodule(plugin, "pro_edit.pro_edit_utils")

    def test_build_masked_pixel_image_prefers_torch_tensor_broadcast_path(self):
        image = _FakeTorchTensor("image")
        mask = _FakeMask("mask")

        with mock.patch.object(self.utils, "torch", _FakeTorchModule()):
            result = self.utils.build_masked_pixel_image(image, mask, fill_value=0.0)

        self.assertEqual(result.text, "((image*(1.0-mask.unsqueeze(-1)))+(0.0*mask.unsqueeze(-1)))")

    def test_build_masked_pixel_image_falls_back_to_masked_fill_for_non_torch_objects(self):
        image = _FallbackTensor()
        mask = object()

        with mock.patch.object(self.utils, "torch", None):
            result = self.utils.build_masked_pixel_image(image, mask, fill_value=0.5)

        self.assertEqual(result, "fallback-masked")
        self.assertTrue(image.called)


if __name__ == "__main__":
    unittest.main()
