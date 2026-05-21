# LLS Simple Mask Draw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single interactive `LLS Simple Mask Draw` node that lets the user paint, erase, clear, preview, persist, and output a repair mask directly over an input image for downstream local repair workflows.

**Architecture:** Add a dedicated `mask_draw/` backend package plus one `WEB_DIRECTORY` frontend extension in `web/js/lls_mask_draw.js`. Keep the backend responsible for mask-state decoding, output mask resolution, and preview-image generation; keep the frontend responsible for interactive drawing, session-only undo/redo, file-backed source-image recovery, and workflow-persisted mask raster state.

**Tech Stack:** Python, unittest, JSON/base64, Pillow, torch, ComfyUI node definitions, ComfyUI frontend JavaScript extension hooks, HTML Canvas, `node --check`

---

## File Structure

- Create: `mask_draw/__init__.py`
  - Expose the new node registration maps.
- Create: `mask_draw/node.py`
  - Define `LLSSimpleMaskDraw`, its schema, and runtime orchestration.
- Create: `mask_draw/utils.py`
  - Decode persisted state, resolve final mask precedence, resize masks, and generate `preview_image`.
- Create: `web/js/lls_mask_draw.js`
  - Register the frontend extension, hide the persistence widget, render the editor, manage strokes, and persist `mask_state_json`.
- Create: `tests/test_mask_draw_helpers.py`
  - Shared plugin loader, tiny-image builders, and state-payload helpers.
- Create: `tests/test_mask_draw_registration.py`
  - Verify plugin registration, node category, and node schema.
- Create: `tests/test_mask_draw_utils.py`
  - Verify persisted-state parsing, fallback precedence, inversion, resize, and preview-image shape.
- Create: `tests/test_mask_draw_node.py`
  - Verify node orchestration, clear semantics, and repair-prepare compatibility.
- Create: `tests/test_mask_draw_frontend.py`
  - Verify `WEB_DIRECTORY` export and presence of the JS extension entrypoints.
- Create: `tests/test_mask_draw_docs.py`
  - Verify the README documents the node and connection pattern.
- Modify: `__init__.py`
  - Append `mask_draw` to `_SUBPACKAGES` and export `WEB_DIRECTORY = "./web"`.
- Modify: `README.md`
  - Document the new node, usage, wiring, and limitations.

### Task 1: Add Package Scaffolding and Registration Contract

**Files:**
- Create: `tests/test_mask_draw_helpers.py`
- Create: `tests/test_mask_draw_registration.py`
- Create: `mask_draw/__init__.py`
- Create: `mask_draw/node.py`
- Create: `mask_draw/utils.py`
- Modify: `__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_draw_helpers.py
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_mask_draw"


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
```

```python
# tests/test_mask_draw_registration.py
import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestMaskDrawRegistration(unittest.TestCase):
    def test_plugin_registers_mask_draw_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleMaskDraw", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleMaskDraw"],
            "LLS Simple Mask Draw",
        )

    def test_mask_draw_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "draw_mask")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "MASK", "IMAGE"))
        self.assertEqual(node_cls.RETURN_NAMES, ("image", "mask", "preview_image"))

        required = schema["required"]
        optional = schema["optional"]
        hidden = schema["hidden"]

        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(optional["input_mask"], ("MASK",))
        self.assertEqual(required["draw_mode"][0], ["brush", "erase"])
        self.assertEqual(required["brush_size"][0], "INT")
        self.assertEqual(required["brush_softness"][0], "FLOAT")
        self.assertEqual(required["overlay_alpha"][0], "FLOAT")
        self.assertEqual(required["invert_mask"][0], "BOOLEAN")
        self.assertEqual(required["mask_state_json"][0], "STRING")
        self.assertEqual(hidden["node_id"], "UNIQUE_ID")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_registration.py' -v`
Expected: FAIL because `mask_draw` is not registered and the node class does not exist yet.

- [ ] **Step 3: Implement the minimal scaffolding**

```python
# __init__.py
from __future__ import annotations

import importlib
import types


WEB_DIRECTORY = "./web"

_SUBPACKAGES: list[str] = [
    "model_loader",
    "conditioning",
    "sampling",
    "qwen",
    "latent",
    "image",
    "repair",
    "upscale",
    "mask",
    "mask_draw",
    "controlnet",
    "lora",
    "video",
    "audio",
    "utils",
]


def _merge_subpackage(pkg_name: str, classes: dict, display_names: dict) -> None:
    full_name = f"{__name__}.{pkg_name}"
    try:
        mod: types.ModuleType = importlib.import_module(full_name)
    except Exception as exc:  # pragma: no cover
        print(f"[LLS] WARNING: Failed to import subpackage '{pkg_name}': {exc}")
        return

    sub_classes: dict = getattr(mod, "NODE_CLASS_MAPPINGS", {})
    sub_names: dict = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for key, cls in sub_classes.items():
        if key in classes:
            continue
        classes[key] = cls
        if key in sub_names:
            display_names[key] = sub_names[key]


def _merge_root_nodes(classes: dict, display_names: dict) -> None:
    try:
        mod: types.ModuleType = importlib.import_module(f"{__name__}.nodes")
    except Exception as exc:  # pragma: no cover
        print(f"[LLS] WARNING: Failed to import root nodes module: {exc}")
        return

    root_classes: dict = getattr(mod, "NODE_CLASS_MAPPINGS", {})
    root_names: dict = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for key, cls in root_classes.items():
        if key in classes:
            continue
        classes[key] = cls
        if key in root_names:
            display_names[key] = root_names[key]


NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

_merge_root_nodes(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

for _pkg in _SUBPACKAGES:
    _merge_subpackage(_pkg, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
```

```python
# mask_draw/__init__.py
from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

```python
# mask_draw/utils.py
def parse_mask_state(mask_state_json):
    return {
        "version": 1,
        "mask_png_base64": "",
        "touched": False,
        "editor": {},
    }
```

```python
# mask_draw/node.py
class LLSSimpleMaskDraw:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "draw_mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "preview_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "draw_mode": (["brush", "erase"], {"default": "brush"}),
                "brush_size": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "brush_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_state_json": ("STRING", {"default": "{}", "multiline": False, "advanced": True}),
            },
            "optional": {
                "input_mask": ("MASK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    def draw_mask(
        self,
        image,
        draw_mode,
        brush_size,
        brush_softness,
        overlay_alpha,
        invert_mask,
        mask_state_json,
        input_mask=None,
        node_id=None,
    ):
        raise RuntimeError("[LLS] LLS Simple Mask Draw is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskDraw": LLSSimpleMaskDraw}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskDraw": "LLS Simple Mask Draw"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_registration.py' -v`
Expected: PASS with the registration and schema contract green.

- [ ] **Step 5: Commit**

```bash
git add __init__.py mask_draw/__init__.py mask_draw/node.py mask_draw/utils.py tests/test_mask_draw_helpers.py tests/test_mask_draw_registration.py
git commit -m "feat: scaffold LLS simple mask draw node"
```

### Task 2: Implement Backend State Decoding and Mask Resolution Helpers

**Files:**
- Modify: `tests/test_mask_draw_helpers.py`
- Create: `tests/test_mask_draw_utils.py`
- Modify: `mask_draw/utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_draw_helpers.py
import base64
import io
import json

from PIL import Image
import torch


def make_image(width=8, height=8, color=0.25):
    return torch.full((1, height, width, 3), float(color), dtype=torch.float32)


def make_mask(width=8, height=8, value=0.0):
    return torch.full((1, height, width), float(value), dtype=torch.float32)


def make_mask_state_json(width=8, height=8, value=1.0, touched=True):
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
```

```python
# tests/test_mask_draw_utils.py
import unittest

import torch

try:
    from .test_mask_draw_helpers import make_image, make_mask, make_mask_state_json
    from ..mask_draw.utils import build_preview_image, parse_mask_state, resolve_output_mask
except ImportError:
    from test_mask_draw_helpers import make_image, make_mask, make_mask_state_json
    from lls_node_test_mask_draw.mask_draw.utils import build_preview_image, parse_mask_state, resolve_output_mask


class TestMaskDrawUtils(unittest.TestCase):
    def test_parse_mask_state_defaults_invalid_json(self):
        state = parse_mask_state("{invalid")

        self.assertFalse(state["touched"])
        self.assertEqual(state["mask_png_base64"], "")
        self.assertEqual(state["version"], 1)

    def test_resolve_output_mask_prefers_saved_mask_when_touched(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=0.0)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        mask = resolve_output_mask(image=image, input_mask=input_mask, mask_state_json=state_json, invert_mask=False)

        self.assertEqual(tuple(mask.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    def test_resolve_output_mask_uses_input_mask_when_untouched(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=3, height=2, value=1.0)

        mask = resolve_output_mask(image=image, input_mask=input_mask, mask_state_json="{}", invert_mask=False)

        self.assertEqual(tuple(mask.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    def test_resolve_output_mask_returns_black_after_clear_state(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=1.0)
        state_json = make_mask_state_json(width=6, height=4, value=0.0, touched=True)

        mask = resolve_output_mask(image=image, input_mask=input_mask, mask_state_json=state_json, invert_mask=False)

        self.assertTrue(torch.all(mask == 0.0))

    def test_resolve_output_mask_applies_invert(self):
        image = make_image(width=6, height=4)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        mask = resolve_output_mask(image=image, input_mask=None, mask_state_json=state_json, invert_mask=True)

        self.assertTrue(torch.all(mask == 0.0))

    def test_build_preview_image_preserves_shape(self):
        image = make_image(width=6, height=4, color=0.2)
        mask = make_mask(width=6, height=4, value=1.0)

        preview = build_preview_image(image=image, mask=mask, overlay_alpha=0.4)

        self.assertEqual(tuple(preview.shape), (1, 4, 6, 3))
        self.assertGreater(float(preview[..., 0].mean()), float(image[..., 0].mean()))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_utils.py' -v`
Expected: FAIL because `resolve_output_mask` and `build_preview_image` do not exist yet and `parse_mask_state` is incomplete.

- [ ] **Step 3: Implement the minimal helper logic**

```python
# mask_draw/utils.py
from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    F = None
    _TORCH_ERR = exc
else:  # pragma: no cover
    _TORCH_ERR = None


def get_image_size(image) -> tuple[int, int]:
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError("[LLS] image must have shape [batch, height, width, channels].")
    return int(shape[2]), int(shape[1])


def parse_mask_state(mask_state_json):
    default = {
        "version": 1,
        "mask_png_base64": "",
        "touched": False,
        "editor": {},
    }
    if not mask_state_json:
        return default
    try:
        raw = json.loads(mask_state_json)
    except Exception:
        return default
    if not isinstance(raw, dict):
        return default
    return {
        "version": int(raw.get("version", 1)),
        "mask_png_base64": str(raw.get("mask_png_base64") or ""),
        "touched": bool(raw.get("touched", False)),
        "editor": dict(raw.get("editor") or {}),
    }


def make_black_mask(image):
    if torch is None:
        raise RuntimeError("[LLS] torch is required for mask generation.") from _TORCH_ERR
    width, height = get_image_size(image)
    return torch.zeros((image.shape[0], height, width), dtype=image.dtype, device=image.device)


def resize_mask_to_image(mask, image):
    width, height = get_image_size(image)
    if mask is None:
        return None
    if tuple(mask.shape[1:]) == (height, width):
        return mask.clamp(0.0, 1.0)
    resized = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(1).clamp(0.0, 1.0)


def decode_mask_png(mask_png_base64, image):
    if not mask_png_base64:
        return None
    width, height = get_image_size(image)
    try:
        payload = base64.b64decode(mask_png_base64)
        loaded = Image.open(io.BytesIO(payload)).convert("L")
    except Exception:
        return None
    if loaded.size != (width, height):
        loaded = loaded.resize((width, height), Image.BILINEAR)
    array = np.asarray(loaded).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).to(device=image.device, dtype=image.dtype)
    return tensor.unsqueeze(0).clamp(0.0, 1.0)


def resolve_output_mask(image, input_mask, mask_state_json, invert_mask):
    state = parse_mask_state(mask_state_json)
    resolved = None
    if state["touched"] and state["mask_png_base64"]:
        resolved = decode_mask_png(state["mask_png_base64"], image)
    if resolved is None and input_mask is not None:
        resolved = resize_mask_to_image(input_mask, image)
    if resolved is None:
        resolved = make_black_mask(image)
    if invert_mask:
        resolved = 1.0 - resolved
    return resolved.clamp(0.0, 1.0)


def build_preview_image(image, mask, overlay_alpha):
    alpha = max(0.0, min(1.0, float(overlay_alpha)))
    expanded = mask.unsqueeze(-1)
    overlay = torch.zeros_like(image)
    overlay[..., 0] = 1.0
    return (image * (1.0 - expanded * alpha) + overlay * (expanded * alpha)).clamp(0.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_utils.py' -v`
Expected: PASS with green coverage for saved-state precedence, fallback, invert, resize, and preview shape.

- [ ] **Step 5: Commit**

```bash
git add mask_draw/utils.py tests/test_mask_draw_helpers.py tests/test_mask_draw_utils.py
git commit -m "feat: add mask draw backend resolution helpers"
```

### Task 3: Implement Node Orchestration and Repair-Prepare Compatibility

**Files:**
- Create: `tests/test_mask_draw_node.py`
- Modify: `mask_draw/node.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_draw_node.py
import unittest

import torch

try:
    from .test_mask_draw_helpers import load_plugin_package, make_image, make_mask, make_mask_state_json
except ImportError:
    from test_mask_draw_helpers import load_plugin_package, make_image, make_mask, make_mask_state_json


class FakeVAE:
    def encode(self, image):
        batch, height, width, _channels = image.shape
        return torch.zeros((batch, 4, max(1, height // 8), max(1, width // 8)), dtype=image.dtype, device=image.device)


class TestMaskDrawNode(unittest.TestCase):
    def test_node_uses_saved_mask_when_touched(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]()
        image = make_image(width=6, height=4, color=0.2)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        image_out, mask_out, preview_out = node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=state_json,
            input_mask=None,
            node_id="123",
        )

        self.assertIs(image_out, image)
        self.assertEqual(tuple(mask_out.shape), (1, 4, 6))
        self.assertEqual(tuple(preview_out.shape), (1, 4, 6, 3))
        self.assertTrue(torch.all(mask_out == 1.0))

    def test_node_falls_back_to_input_mask_when_untouched(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]()
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=3, height=2, value=1.0)

        _image_out, mask_out, _preview_out = node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json="{}",
            input_mask=input_mask,
            node_id="123",
        )

        self.assertEqual(tuple(mask_out.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask_out == 1.0))

    def test_node_returns_black_after_clear_state(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]()
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=1.0)
        cleared_state = make_mask_state_json(width=6, height=4, value=0.0, touched=True)

        _image_out, mask_out, _preview_out = node.draw_mask(
            image=image,
            draw_mode="erase",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=cleared_state,
            input_mask=input_mask,
            node_id="123",
        )

        self.assertTrue(torch.all(mask_out == 0.0))

    def test_node_output_connects_to_repair_prepare(self):
        plugin = load_plugin_package()
        draw_node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleMaskDraw"]()
        prepare_node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]()
        image = make_image(width=64, height=64, color=0.1)
        state_json = make_mask_state_json(width=64, height=64, value=1.0, touched=True)

        image_out, mask_out, _preview_out = draw_node.draw_mask(
            image=image,
            draw_mode="brush",
            brush_size=32,
            brush_softness=0.5,
            overlay_alpha=0.4,
            invert_mask=False,
            mask_state_json=state_json,
            input_mask=None,
            node_id="123",
        )

        latent, work_image, work_mask, repair_info, recommended, *_rest = prepare_node.prepare(
            image=image_out,
            mask=mask_out,
            vae=FakeVAE(),
            repair_scope="region",
            repair_kernel="latent_mask",
            task_hint="repair",
            mask_grow=24,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=256,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model_info={"model_family": "SDXL", "model_role": "base"},
        )

        self.assertEqual(tuple(work_image.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(work_mask.shape), (1, 64, 64))
        self.assertEqual(repair_info["repair_scope"], "region")
        self.assertIn("noise_mask", latent)
        self.assertEqual(recommended, 0.45)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_node.py' -v`
Expected: FAIL because the node method still raises `RuntimeError`.

- [ ] **Step 3: Implement the node orchestration**

```python
# mask_draw/node.py
from __future__ import annotations

from .utils import build_preview_image, resolve_output_mask


class LLSSimpleMaskDraw:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "draw_mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "preview_image")
    DESCRIPTION = "Interactively draw and refine a repair mask directly on an input image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "draw_mode": (["brush", "erase"], {"default": "brush"}),
                "brush_size": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "brush_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_state_json": ("STRING", {"default": "{}", "multiline": False, "advanced": True}),
            },
            "optional": {
                "input_mask": ("MASK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    def draw_mask(
        self,
        image,
        draw_mode,
        brush_size,
        brush_softness,
        overlay_alpha,
        invert_mask,
        mask_state_json,
        input_mask=None,
        node_id=None,
    ):
        if image is None:
            raise RuntimeError("[LLS] image input is required for LLS Simple Mask Draw.")

        mask = resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json=mask_state_json,
            invert_mask=bool(invert_mask),
        )
        preview_image = build_preview_image(image=image, mask=mask, overlay_alpha=overlay_alpha)
        return (image, mask, preview_image)


NODE_CLASS_MAPPINGS = {"LLSSimpleMaskDraw": LLSSimpleMaskDraw}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleMaskDraw": "LLS Simple Mask Draw"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_node.py' -v`
Expected: PASS with green coverage for touched-state precedence, untouched fallback, clear semantics, and repair-prepare compatibility.

- [ ] **Step 5: Commit**

```bash
git add mask_draw/node.py tests/test_mask_draw_node.py
git commit -m "feat: implement LLS simple mask draw backend node"
```

### Task 4: Add the Frontend Extension and JS Asset Registration

**Files:**
- Create: `tests/test_mask_draw_frontend.py`
- Create: `web/js/lls_mask_draw.js`
- Modify: `__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_draw_frontend.py
import pathlib
import unittest

try:
    from .test_mask_draw_helpers import ROOT, load_plugin_package
except ImportError:
    from test_mask_draw_helpers import ROOT, load_plugin_package


class TestMaskDrawFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()

        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_mask_draw.js"
        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_target_extension(self):
        asset = (ROOT / "web" / "js" / "lls_mask_draw.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLS Simple Mask Draw", asset)
        self.assertIn("mask_state_json", asset)
        self.assertIn("beforeRegisterNodeDef", asset)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_frontend.py' -v`
Expected: FAIL because the `web/js/lls_mask_draw.js` asset does not exist yet.

- [ ] **Step 3: Implement the frontend extension**

```javascript
// web/js/lls_mask_draw.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_NAME = "LLS Simple Mask Draw";
const HISTORY_LIMIT = 20;

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function parseState(value) {
  try {
    const parsed = JSON.parse(value || "{}");
    if (!parsed || typeof parsed !== "object") return {};
    return parsed;
  } catch {
    return {};
  }
}

function serializeState(stateWidget, state) {
  stateWidget.value = JSON.stringify({
    version: 1,
    mask_png_base64: state.maskPngBase64 || "",
    touched: !!state.touched,
    editor: {
      draw_mode: state.drawMode || "brush",
      brush_size: state.brushSize || 32,
      brush_softness: state.brushSoftness ?? 0.5,
      overlay_alpha: state.overlayAlpha ?? 0.4,
    },
  });
  app.graph.setDirtyCanvas(true, true);
}

function dataUrlToBase64(dataUrl) {
  const commaIndex = dataUrl.indexOf(",");
  return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : "";
}

function getNodeInputLink(node, inputName) {
  const input = node.inputs?.find((item) => item.name === inputName);
  if (!input || input.link == null) return null;
  return node.graph?.links?.[input.link] ?? null;
}

function getUpstreamNode(node) {
  const link = getNodeInputLink(node, "image");
  if (!link) return null;
  return node.graph?.getNodeById?.(link.origin_id) ?? null;
}

function getUpstreamNodeForInput(node, inputName) {
  const link = getNodeInputLink(node, inputName);
  if (!link) return null;
  return node.graph?.getNodeById?.(link.origin_id) ?? null;
}

function resolveViewUrl(widgetValue, folderType) {
  if (!widgetValue) return null;
  const params = new URLSearchParams();
  params.set("filename", widgetValue);
  if (!String(widgetValue).startsWith("blake3:") && folderType) {
    params.set("type", folderType);
  }
  return `/view?${params.toString()}`;
}

async function tryLoadFileBackedSource(node) {
  const upstream = getUpstreamNode(node);
  if (!upstream) return null;

  const widget = upstream.widgets?.find((item) => item.name === "image");
  if (!widget || !widget.value) return null;

  const typeName = String(upstream.type || upstream.comfyClass || upstream.title || "");
  if (typeName.includes("LoadImageOutput")) {
    return resolveViewUrl(widget.value, "output");
  }
  if (typeName.includes("LoadImage")) {
    return resolveViewUrl(widget.value, "input");
  }
  return null;
}

async function tryLoadFileBackedMask(node) {
  const upstream = getUpstreamNodeForInput(node, "input_mask");
  if (!upstream) return null;

  const imageWidget = upstream.widgets?.find((item) => item.name === "image");
  if (!imageWidget || !imageWidget.value) return null;

  const typeName = String(upstream.type || upstream.comfyClass || upstream.title || "");
  if (typeName.includes("LoadImageMask")) {
    const params = new URLSearchParams();
    params.set("filename", imageWidget.value);
    params.set("type", "input");
    params.set("channel", "a");
    return `/view?${params.toString()}`;
  }
  if (typeName.includes("LoadImage")) {
    const params = new URLSearchParams();
    params.set("filename", imageWidget.value);
    params.set("type", "input");
    params.set("channel", "a");
    return `/view?${params.toString()}`;
  }
  return null;
}

function pushHistory(history, dataUrl) {
  history.undo.push(dataUrl);
  if (history.undo.length > HISTORY_LIMIT) {
    history.undo.shift();
  }
  history.redo.length = 0;
}

function drawOverlay(displayCtx, maskCanvas, overlayAlpha, invertMask) {
  const width = displayCtx.canvas.width;
  const height = displayCtx.canvas.height;
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const scratchCtx = scratch.getContext("2d");
  scratchCtx.drawImage(maskCanvas, 0, 0, width, height);
  const imageData = scratchCtx.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const value = invertMask ? 255 - data[i] : data[i];
    data[i] = 255;
    data[i + 1] = 0;
    data[i + 2] = 0;
    data[i + 3] = Math.round((value / 255) * overlayAlpha * 255);
  }
  scratchCtx.putImageData(imageData, 0, 0);
  displayCtx.drawImage(scratch, 0, 0);
}

function buildEditor(node) {
  const stateWidget = findWidget(node, "mask_state_json");
  const drawModeWidget = findWidget(node, "draw_mode");
  const brushSizeWidget = findWidget(node, "brush_size");
  const brushSoftnessWidget = findWidget(node, "brush_softness");
  const overlayAlphaWidget = findWidget(node, "overlay_alpha");
  const invertWidget = findWidget(node, "invert_mask");

  hideWidget(stateWidget);

  const container = document.createElement("div");
  container.className = "lls-mask-draw";
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "8px";
  container.style.minWidth = "320px";

  const toolbar = document.createElement("div");
  toolbar.style.display = "flex";
  toolbar.style.gap = "6px";

  const stage = document.createElement("canvas");
  stage.width = 512;
  stage.height = 512;
  stage.style.width = "100%";
  stage.style.border = "1px solid #555";
  stage.style.borderRadius = "8px";
  stage.style.background = "#111";

  const maskCanvas = document.createElement("canvas");
  const history = { undo: [], redo: [] };
  const stageCtx = stage.getContext("2d");
  const maskCtx = maskCanvas.getContext("2d");
  const state = {
    touched: false,
    drawMode: drawModeWidget?.value || "brush",
    brushSize: brushSizeWidget?.value || 32,
    brushSoftness: brushSoftnessWidget?.value ?? 0.5,
    overlayAlpha: overlayAlphaWidget?.value ?? 0.4,
    baseImage: null,
    drawing: false,
    maskPngBase64: "",
  };

  function snapshotMask() {
    return maskCanvas.toDataURL("image/png");
  }

  function restoreMask(dataUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
        resolve();
      };
      img.src = dataUrl;
    });
  }

  function renderStage() {
    stageCtx.clearRect(0, 0, stage.width, stage.height);
    if (state.baseImage) {
      stageCtx.drawImage(state.baseImage, 0, 0, stage.width, stage.height);
    } else {
      stageCtx.fillStyle = "#1a1a1a";
      stageCtx.fillRect(0, 0, stage.width, stage.height);
      stageCtx.fillStyle = "#888";
      stageCtx.fillText("Run once or connect Load Image to initialize preview", 16, 24);
    }
    drawOverlay(stageCtx, maskCanvas, state.overlayAlpha, !!invertWidget?.value);
  }

  function persistMask() {
    const pngData = snapshotMask();
    state.maskPngBase64 = dataUrlToBase64(pngData);
    serializeState(stateWidget, state);
    renderStage();
  }

  function makeButton(label, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.onclick = handler;
    return button;
  }

  async function loadInitialImage() {
    const url = await tryLoadFileBackedSource(node);
    if (!url) return;
    const image = new Image();
    image.onload = () => {
      state.baseImage = image;
      if (!maskCanvas.width || !maskCanvas.height) {
        maskCanvas.width = image.naturalWidth;
        maskCanvas.height = image.naturalHeight;
        maskCtx.fillStyle = "black";
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      }
      renderStage();

      if (!state.maskPngBase64) {
        tryLoadFileBackedMask(node).then((maskUrl) => {
          if (!maskUrl) return;
          const maskImage = new Image();
          maskImage.onload = () => {
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            maskCtx.drawImage(maskImage, 0, 0, maskCanvas.width, maskCanvas.height);
            renderStage();
          };
          maskImage.src = maskUrl;
        });
      }
    };
    image.src = url;
  }

  function syncWidgetState() {
    state.drawMode = drawModeWidget?.value || "brush";
    state.brushSize = brushSizeWidget?.value || 32;
    state.brushSoftness = brushSoftnessWidget?.value ?? 0.5;
    state.overlayAlpha = overlayAlphaWidget?.value ?? 0.4;
    renderStage();
  }

  const savedState = parseState(stateWidget?.value);
  if (savedState.editor) {
    state.drawMode = savedState.editor.draw_mode || state.drawMode;
    state.brushSize = savedState.editor.brush_size || state.brushSize;
    state.brushSoftness = savedState.editor.brush_softness ?? state.brushSoftness;
    state.overlayAlpha = savedState.editor.overlay_alpha ?? state.overlayAlpha;
  }
  state.touched = !!savedState.touched;
  state.maskPngBase64 = savedState.mask_png_base64 || "";

  toolbar.appendChild(
    makeButton("Clear", () => {
      if (!maskCanvas.width || !maskCanvas.height) return;
      pushHistory(history, snapshotMask());
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.fillStyle = "black";
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      state.touched = true;
      persistMask();
    }),
  );
  toolbar.appendChild(
    makeButton("Undo", async () => {
      if (!history.undo.length) return;
      history.redo.push(snapshotMask());
      const previous = history.undo.pop();
      await restoreMask(previous);
      state.touched = true;
      persistMask();
    }),
  );
  toolbar.appendChild(
    makeButton("Redo", async () => {
      if (!history.redo.length) return;
      history.undo.push(snapshotMask());
      const next = history.redo.pop();
      await restoreMask(next);
      state.touched = true;
      persistMask();
    }),
  );
  toolbar.appendChild(
    makeButton("Invert", () => {
      if (invertWidget) {
        invertWidget.value = !invertWidget.value;
        invertWidget.callback?.(invertWidget.value);
      }
      renderStage();
    }),
  );

  stage.addEventListener("pointerdown", (event) => {
    if (!maskCanvas.width || !maskCanvas.height) return;
    state.drawing = true;
    pushHistory(history, snapshotMask());
    const rect = stage.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const radius = Math.max(1, Number(state.brushSize || 32)) / 2;
    const gradient = maskCtx.createRadialGradient(x, y, 0, x, y, radius);
    const softness = Math.max(0.0, Math.min(1.0, Number(state.brushSoftness ?? 0.5)));
    const edge = 1.0 - softness;
    const color = state.drawMode === "erase" ? "0,0,0" : "255,255,255";
    gradient.addColorStop(0.0, `rgba(${color},1)`);
    gradient.addColorStop(edge, `rgba(${color},1)`);
    gradient.addColorStop(1.0, `rgba(${color},0)`);
    maskCtx.fillStyle = gradient;
    maskCtx.beginPath();
    maskCtx.arc(x, y, radius, 0, Math.PI * 2);
    maskCtx.fill();
    state.touched = true;
    persistMask();
  });

  stage.addEventListener("pointermove", (event) => {
    if (!state.drawing || !maskCanvas.width || !maskCanvas.height) return;
    const rect = stage.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const radius = Math.max(1, Number(state.brushSize || 32)) / 2;
    const gradient = maskCtx.createRadialGradient(x, y, 0, x, y, radius);
    const softness = Math.max(0.0, Math.min(1.0, Number(state.brushSoftness ?? 0.5)));
    const edge = 1.0 - softness;
    const color = state.drawMode === "erase" ? "0,0,0" : "255,255,255";
    gradient.addColorStop(0.0, `rgba(${color},1)`);
    gradient.addColorStop(edge, `rgba(${color},1)`);
    gradient.addColorStop(1.0, `rgba(${color},0)`);
    maskCtx.fillStyle = gradient;
    maskCtx.beginPath();
    maskCtx.arc(x, y, radius, 0, Math.PI * 2);
    maskCtx.fill();
    state.touched = true;
    persistMask();
  });

  stage.addEventListener("pointerup", () => {
    state.drawing = false;
  });
  stage.addEventListener("pointerleave", () => {
    state.drawing = false;
  });

  container.appendChild(toolbar);
  container.appendChild(stage);

  node.addDOMWidget("lls_mask_editor", "lls_mask_editor", container, {
    serialize: false,
    hideOnZoom: false,
    getValue: () => stateWidget?.value,
    setValue: (value) => {
      if (stateWidget) {
        stateWidget.value = value;
      }
    },
  });

  if (state.maskPngBase64) {
    const restored = new Image();
    restored.onload = () => {
      if (!maskCanvas.width || !maskCanvas.height) {
        maskCanvas.width = restored.naturalWidth;
        maskCanvas.height = restored.naturalHeight;
      }
      maskCtx.drawImage(restored, 0, 0, maskCanvas.width, maskCanvas.height);
      renderStage();
    };
    restored.src = `data:image/png;base64,${state.maskPngBase64}`;
  }

  drawModeWidget && (drawModeWidget.callback = syncWidgetState);
  brushSizeWidget && (brushSizeWidget.callback = syncWidgetState);
  brushSoftnessWidget && (brushSoftnessWidget.callback = syncWidgetState);
  overlayAlphaWidget && (overlayAlphaWidget.callback = syncWidgetState);
  invertWidget && (invertWidget.callback = syncWidgetState);

  loadInitialImage();
  syncWidgetState();
}

app.registerExtension({
  name: "LLS.MaskDraw",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_NAME) {
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onMaskDrawNodeCreated() {
      const result = onNodeCreated?.apply(this, arguments);
      if (!this.__llsMaskDrawEditorAttached) {
        this.__llsMaskDrawEditorAttached = true;
        buildEditor(this);
      }
      return result;
    };
  },
});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_frontend.py' -v`
Expected: PASS with green coverage for `WEB_DIRECTORY` export and asset presence.

- [ ] **Step 5: Run a syntax check for the asset**

Run: `node --check web/js/lls_mask_draw.js`
Expected: PASS with no output.

- [ ] **Step 6: Commit**

```bash
git add __init__.py web/js/lls_mask_draw.js tests/test_mask_draw_frontend.py
git commit -m "feat: add LLS simple mask draw frontend editor"
```

### Task 5: Update README and Run Full Verification

**Files:**
- Create: `tests/test_mask_draw_docs.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing README contract test**

```python
# tests/test_mask_draw_docs.py
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestMaskDrawDocs(unittest.TestCase):
    def test_readme_documents_mask_draw_node_and_repair_wiring(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Mask Draw", readme)
        self.assertIn("Load Image", readme)
        self.assertIn("LLS Simple Repair Prepare", readme)
        self.assertIn("preview_image", readme)
        self.assertIn("手动指定删除区域", readme)
        self.assertIn("brush", readme)
        self.assertIn("erase", readme)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the README test to verify it fails**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_docs.py' -v`
Expected: FAIL because the README does not document the new node yet.

- [ ] **Step 3: Update the README**

```markdown
## Image Repair

`LLS-node` now includes a direct manual mask-drawing helper for local repair workflows:

- `LLS Simple Mask Draw`
- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

### `LLS Simple Mask Draw`

Draw or erase a repair mask directly on the input image inside the ComfyUI node.

**Inputs**

- `image: IMAGE`
- `input_mask: MASK` (optional)

**Outputs**

- `image: IMAGE`
- `mask: MASK`
- `preview_image: IMAGE`

**Supported first-version interactions**

- `brush`
- `erase`
- `Clear`
- `Undo`
- `Redo`
- overlay alpha preview
- optional invert mask

### Minimal manual workflow

`Load Image -> LLS Simple Mask Draw -> Preview Image`

### Repair workflow wiring

- `Load Image.image -> LLS Simple Mask Draw.image`
- `LLS Simple Mask Draw.image -> LLS Simple Repair Prepare.image`
- `LLS Simple Mask Draw.mask -> LLS Simple Repair Prepare.mask`

### Typical uses

- 手动指定删除区域
- 手动指定修复区域
- 手动指定去阴影区域
- 手动指定局部增强区域

### First-version limits

- no polygon tool
- no rectangle tool
- no ellipse tool
- no magic wand
- no automatic segmentation
- some non-file upstream images may need one execution before the editor can show the base image
```

- [ ] **Step 4: Run the README test to verify it passes**

Run: `python3 -m unittest discover -s tests -p 'test_mask_draw_docs.py' -v`
Expected: PASS with the new node documented.

- [ ] **Step 5: Run the focused automated verification suite**

Run: `python3 -m unittest tests.test_mask_draw_registration tests.test_mask_draw_utils tests.test_mask_draw_node tests.test_mask_draw_frontend tests.test_mask_draw_docs -v`
Expected: PASS with all new mask-draw tests green.

- [ ] **Step 6: Run regression checks for adjacent repair behavior**

Run: `python3 -m unittest tests.test_repair_registration tests.test_repair_prepare -v`
Expected: PASS with no regression in repair registration or prepare behavior.

- [ ] **Step 7: Run a lightweight compile check**

Run: `python3 -m compileall __init__.py mask_draw tests`
Expected: PASS with no syntax errors.

- [ ] **Step 8: Perform the manual frontend verification checklist**

Run these checks in ComfyUI:

```text
1. Add `LLS Simple Mask Draw` to the graph and confirm it appears under `LLS/Image Repair`.
2. Connect `Load Image -> LLS Simple Mask Draw -> Preview Image`.
3. Confirm the image preview appears immediately for a file-backed `Load Image`.
4. Paint with `brush` and verify `preview_image` shows a red overlay.
5. Switch to `erase` and verify mask removal.
6. Click `Clear` and verify the output mask becomes black.
7. Use `Undo` and `Redo` repeatedly and verify no crash or stale state.
8. Save the workflow, close it, reopen it, and verify the current mask raster is restored.
9. Connect `Load Image Mask` into `LLS Simple Mask Draw.input_mask` and verify the loaded mask appears as the editable starting mask when the source is file-backed.
10. Connect `LLS Simple Mask Draw.image` and `.mask` into `LLS Simple Repair Prepare`.
11. Queue the repair-prepare path and verify it accepts the mask output without errors.
```

Expected: All manual checks succeed, with only the documented first-version limitation for non-file upstream image previews.

- [ ] **Step 9: Commit**

```bash
git add README.md tests/test_mask_draw_docs.py
git commit -m "docs: document LLS simple mask draw"
```
