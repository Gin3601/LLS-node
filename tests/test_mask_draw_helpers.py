import base64
import importlib.util
import pathlib
import sys
import importlib
import io
import json
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_mask_draw"


def _import_optional(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


torch = _import_optional("torch")
np = _import_optional("numpy")
_pil_image_module = _import_optional("PIL.Image")
Image = _pil_image_module
HAS_MASK_DRAW_RUNTIME_DEPS = torch is not None and np is not None and Image is not None
MASK_DRAW_RUNTIME_DEPS_MESSAGE = "mask draw helper tests require torch, numpy, and Pillow"


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


def load_mask_draw_utils():
    load_plugin_package()
    return importlib.import_module(f"{MODULE_NAME}.mask_draw.utils")


def make_image(width=8, height=8, color=0.25):
    require_mask_draw_runtime_deps()
    return torch.full((1, height, width, 3), float(color), dtype=torch.float32)


def make_mask(width=8, height=8, value=0.0):
    require_mask_draw_runtime_deps()
    return torch.full((1, height, width), float(value), dtype=torch.float32)


def make_mask_state_json(width=8, height=8, value=1.0, touched=True):
    require_mask_draw_runtime_deps()
    pixels = Image.new("L", (width, height), color=int(round(float(value) * 255.0)))
    buffer = io.BytesIO()
    pixels.save(buffer, format="PNG")
    payload = {
        "version": 1,
        "mask_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "touched": bool(touched),
        "editor": {
            "draw_mode": "brush",
            "brush_size": 32,
            "brush_softness": 0.5,
            "overlay_alpha": 0.4,
        },
    }
    return json.dumps(payload)


def require_mask_draw_runtime_deps():
    if not HAS_MASK_DRAW_RUNTIME_DEPS:
        raise unittest.SkipTest(MASK_DRAW_RUNTIME_DEPS_MESSAGE)
