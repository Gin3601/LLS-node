import importlib
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_image_composite"


def _import_optional(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


torch = _import_optional("torch")
np = _import_optional("numpy")
_pil_image_module = _import_optional("PIL.Image")
Image = _pil_image_module
HAS_IMAGE_COMPOSITE_RUNTIME_DEPS = torch is not None and np is not None and Image is not None
IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE = "image composite tests require torch, numpy, and Pillow"


def load_plugin_package():
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(f"{MODULE_NAME}."):
            sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def require_image_composite_runtime_deps():
    if not HAS_IMAGE_COMPOSITE_RUNTIME_DEPS:
        raise unittest.SkipTest(IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE)


def make_image(width=8, height=8, color=(0.0, 0.0, 0.0), alpha=None):
    require_image_composite_runtime_deps()
    rgb = tuple(float(channel) for channel in color)
    channels = 4 if alpha is not None else 3
    tensor = torch.zeros((1, height, width, channels), dtype=torch.float32)
    tensor[..., 0] = rgb[0]
    tensor[..., 1] = rgb[1]
    tensor[..., 2] = rgb[2]
    if alpha is not None:
        tensor[..., 3] = float(alpha)
    return tensor
