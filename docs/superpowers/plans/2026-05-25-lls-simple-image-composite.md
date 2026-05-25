# LLS Simple Image Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `LLS Simple Image Composite` node that composites an overlay image onto a background image with translation, scale, rotation, opacity, independent anchor and rotation-origin modes, plus a ComfyUI node-local realtime preview UI.

**Architecture:** Keep the backend compositing logic in focused `image/` modules and merge registration in `image/__init__.py` rather than growing `image/nodes.py` further. Mirror the existing `LLS Simple Mask Draw` pattern by pairing a Python node with a `web/js` extension that renders a node-local preview and synchronizes drag operations back into widgets.

**Tech Stack:** Python, unittest, torch, numpy, Pillow, ComfyUI node contracts, DOM canvas frontend extension

---

### Task 1: Create Test Helpers And Lock Registration / Frontend / Docs Contracts

**Files:**
- Create: `tests/test_image_composite_helpers.py`
- Create: `tests/test_image_composite_registration.py`
- Create: `tests/test_image_composite_frontend.py`
- Create: `tests/test_image_composite_docs.py`

- [ ] **Step 1: Create the shared test helper file**

```python
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
```

- [ ] **Step 2: Create failing registration, frontend, and docs tests**

```python
# tests/test_image_composite_registration.py
import unittest

try:
    from .test_image_composite_helpers import load_plugin_package
except ImportError:
    from test_image_composite_helpers import load_plugin_package


class TestImageCompositeRegistration(unittest.TestCase):
    def test_plugin_registers_image_composite_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleImageComposite", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleImageComposite"],
            "LLS Simple Image Composite",
        )

    def test_image_composite_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleImageComposite"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Image")
        self.assertEqual(node_cls.FUNCTION, "composite")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("output_image",))
        self.assertEqual(required["background_image"], ("IMAGE",))
        self.assertEqual(required["overlay_image"], ("IMAGE",))
        self.assertEqual(required["x_offset"][0], "INT")
        self.assertEqual(required["x_offset"][1]["default"], 0)
        self.assertEqual(required["y_offset"][0], "INT")
        self.assertEqual(required["anchor_mode"][0], ["top_left", "center"])
        self.assertEqual(required["rotation_origin_mode"][0], ["top_left", "center"])
        self.assertEqual(required["opacity"][0], "FLOAT")
        self.assertEqual(required["opacity"][1]["default"], 1.0)
        self.assertEqual(required["blend_mode"][0], ["normal"])
        self.assertEqual(required["scale"][0], "FLOAT")
        self.assertEqual(required["rotation"][0], "FLOAT")
        self.assertEqual(required["keep_aspect"][0], "BOOLEAN")
```

```python
# tests/test_image_composite_frontend.py
import unittest

try:
    from .test_image_composite_helpers import ROOT, load_plugin_package
except ImportError:
    from test_image_composite_helpers import ROOT, load_plugin_package


class TestImageCompositeFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_image_composite.js"
        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_image_composite_extension(self):
        asset = (ROOT / "web" / "js" / "lls_image_composite.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSSimpleImageComposite", asset)
        self.assertIn("LLS Simple Image Composite", asset)
        self.assertIn("beforeRegisterNodeDef", asset)
        self.assertIn("addDOMWidget", asset)
        self.assertIn("pointerdown", asset)
        self.assertIn("pointermove", asset)
        self.assertIn("rotation_origin_mode", asset)
```

```python
# tests/test_image_composite_docs.py
import unittest

try:
    from .test_image_composite_helpers import ROOT
except ImportError:
    from test_image_composite_helpers import ROOT


class TestImageCompositeDocs(unittest.TestCase):
    def test_readme_documents_image_composite_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Image Composite", readme)
        self.assertIn("background_image", readme)
        self.assertIn("overlay_image", readme)
        self.assertIn("output_image", readme)
        self.assertIn("x_offset", readme)
        self.assertIn("y_offset", readme)
        self.assertIn("anchor_mode", readme)
        self.assertIn("rotation_origin_mode", readme)
        self.assertIn("opacity", readme)
        self.assertIn("scale", readme)
        self.assertIn("rotation", readme)
        self.assertIn("keep_aspect", readme)
```

- [ ] **Step 3: Run the contract tests to verify they fail**

Run: `python3 -m unittest tests.test_image_composite_registration tests.test_image_composite_frontend tests.test_image_composite_docs`
Expected: FAIL because the node, frontend asset, and README section do not exist yet.


### Task 2: Write Failing Runtime Tests For Backend Compositing

**Files:**
- Create: `tests/test_image_composite_node.py`
- Test: `tests/test_image_composite_node.py`

- [ ] **Step 1: Create failing node behavior tests**

```python
import unittest

try:
    from .test_image_composite_helpers import (
        HAS_IMAGE_COMPOSITE_RUNTIME_DEPS,
        IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
    )
except ImportError:
    from test_image_composite_helpers import (
        HAS_IMAGE_COMPOSITE_RUNTIME_DEPS,
        IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE,
        load_plugin_package,
        make_image,
    )


@unittest.skipUnless(HAS_IMAGE_COMPOSITE_RUNTIME_DEPS, IMAGE_COMPOSITE_RUNTIME_DEPS_MESSAGE)
class TestImageCompositeNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin = load_plugin_package()
        cls.node = plugin.NODE_CLASS_MAPPINGS["LLSSimpleImageComposite"]()

    def test_top_left_offset_places_overlay_at_requested_position(self):
        background = make_image(width=6, height=6, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertEqual(tuple(output_image.shape), (1, 6, 6, 3))
        self.assertTrue((output_image[0, 1:3, 3:5] > 0.99).all().item())
        self.assertEqual(float(output_image[0, 0, 0, 0].item()), 0.0)

    def test_center_anchor_places_overlay_by_its_center(self):
        background = make_image(width=6, height=6, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(0.0, 1.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=2,
            anchor_mode="center",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image[0, 1:3, 2:4, 1] > 0.99).all().item())

    def test_partial_overflow_is_clipped_to_background_bounds(self):
        background = make_image(width=5, height=5, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=3, height=3, color=(1.0, 0.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=4,
            y_offset=4,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertGreater(float(output_image[0, 4, 4, 0].item()), 0.99)
        self.assertEqual(float(output_image[0, 0, 0, 0].item()), 0.0)

    def test_overlay_fully_outside_returns_background(self):
        background = make_image(width=5, height=5, color=(0.1, 0.1, 0.1))
        overlay = make_image(width=2, height=2, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=8,
            y_offset=8,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image == background).all().item())

    def test_opacity_zero_returns_background(self):
        background = make_image(width=4, height=4, color=(0.2, 0.2, 0.2))
        overlay = make_image(width=2, height=2, color=(1.0, 0.0, 0.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=0.0,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image == background).all().item())

    def test_rgba_overlay_uses_alpha_and_opacity(self):
        background = make_image(width=4, height=4, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=2, height=2, color=(1.0, 0.0, 0.0), alpha=0.5)

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=0.5,
            blend_mode="normal",
            scale=1.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertAlmostEqual(float(output_image[0, 1, 1, 0].item()), 0.25, places=2)

    def test_scale_expands_overlay_footprint(self):
        background = make_image(width=5, height=5, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=1, height=1, color=(1.0, 1.0, 1.0))

        (output_image,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=1,
            y_offset=1,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=2.0,
            rotation=0.0,
            keep_aspect=True,
        )

        self.assertTrue((output_image[0, 1:3, 1:3] > 0.99).all().item())

    def test_rotation_origin_mode_changes_rotated_result(self):
        background = make_image(width=8, height=8, color=(0.0, 0.0, 0.0))
        overlay = make_image(width=3, height=1, color=(0.0, 0.0, 1.0))

        (center_rotated,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=3,
            anchor_mode="top_left",
            rotation_origin_mode="center",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=90.0,
            keep_aspect=True,
        )
        (corner_rotated,) = self.node.composite(
            background_image=background,
            overlay_image=overlay,
            x_offset=3,
            y_offset=3,
            anchor_mode="top_left",
            rotation_origin_mode="top_left",
            opacity=1.0,
            blend_mode="normal",
            scale=1.0,
            rotation=90.0,
            keep_aspect=True,
        )

        self.assertFalse((center_rotated == corner_rotated).all().item())
```

- [ ] **Step 2: Run the backend tests to verify they fail**

Run: `python3 -m unittest tests.test_image_composite_node`
Expected: FAIL because `LLSSimpleImageComposite` and its backend logic do not exist yet.


### Task 3: Implement Backend Compositing Utilities

**Files:**
- Create: `image/composite_utils.py`
- Test: `tests/test_image_composite_node.py`

- [ ] **Step 1: Create the backend utility module with dependency guards and tensor validation**

```python
from __future__ import annotations

import io

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None
    _NUMPY_ERR = exc
else:  # pragma: no cover
    _NUMPY_ERR = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_ERR = exc
else:  # pragma: no cover
    _PIL_ERR = None

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_ERR = exc
else:  # pragma: no cover
    _TORCH_ERR = None


ANCHOR_MODE_CHOICES = ["top_left", "center"]
ROTATION_ORIGIN_MODE_CHOICES = ["top_left", "center"]
BLEND_MODE_CHOICES = ["normal"]


def _require_runtime_deps():
    if torch is None:
        raise RuntimeError("[LLS] torch is required for image compositing.") from _TORCH_ERR
    if np is None:
        raise RuntimeError("[LLS] numpy is required for image compositing.") from _NUMPY_ERR
    if Image is None:
        raise RuntimeError("[LLS] Pillow is required for image compositing.") from _PIL_ERR


def _require_image_tensor(name, image):
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError(f"[LLS] {name} must have shape [batch, height, width, channels].")
    if int(shape[0]) <= 0 or int(shape[1]) <= 0 or int(shape[2]) <= 0:
        raise RuntimeError(f"[LLS] {name} must have positive batch, width, and height.")
    channels = int(shape[3])
    if channels not in {3, 4}:
        raise RuntimeError(f"[LLS] {name} must have 3 or 4 channels; got {channels}.")
```

- [ ] **Step 2: Implement the single-image compositing helpers**

```python
def _tensor_sample_to_rgba_pil(sample):
    rgb = sample.detach().cpu().clamp(0.0, 1.0).numpy()
    if rgb.shape[-1] == 3:
        alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        rgb = np.concatenate([rgb, alpha], axis=-1)
    rgba = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _pil_to_tensor(image, *, device, dtype):
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[-1] != 4:
        raise RuntimeError("[LLS] compositing helper expected an RGBA PIL image.")
    rgb = array[..., :3]
    return torch.from_numpy(rgb).to(device=device, dtype=dtype)


def _scaled_overlay_rgba(overlay_rgba, scale):
    width, height = overlay_rgba.size
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    if (scaled_width, scaled_height) == (width, height):
        return overlay_rgba
    return overlay_rgba.resize((scaled_width, scaled_height), Image.BICUBIC)


def _resolve_top_left(width, height, x_offset, y_offset, anchor_mode):
    if anchor_mode == "top_left":
        return int(x_offset), int(y_offset)
    if anchor_mode == "center":
        return int(x_offset - (width / 2.0)), int(y_offset - (height / 2.0))
    raise RuntimeError(f"[LLS] Unsupported anchor_mode '{anchor_mode}'.")


def _resolve_rotation_center(top_left_x, top_left_y, width, height, rotation_origin_mode):
    if rotation_origin_mode == "top_left":
        return float(top_left_x), float(top_left_y)
    if rotation_origin_mode == "center":
        return float(top_left_x) + (width / 2.0), float(top_left_y) + (height / 2.0)
    raise RuntimeError(f"[LLS] Unsupported rotation_origin_mode '{rotation_origin_mode}'.")
```

- [ ] **Step 3: Implement the batch-aware public compositing function**

```python
def composite_images(
    background_image,
    overlay_image,
    *,
    x_offset,
    y_offset,
    anchor_mode,
    rotation_origin_mode,
    opacity,
    blend_mode,
    scale,
    rotation,
    keep_aspect,
):
    del keep_aspect

    _require_runtime_deps()
    if background_image is None:
        raise RuntimeError("[LLS] background_image is required for image compositing.")
    if overlay_image is None:
        raise RuntimeError("[LLS] overlay_image is required for image compositing.")
    if blend_mode != "normal":
        raise RuntimeError(f"[LLS] Unsupported blend_mode '{blend_mode}'.")

    _require_image_tensor("background_image", background_image)
    _require_image_tensor("overlay_image", overlay_image)
    clamped_scale = max(0.01, float(scale))
    clamped_opacity = max(0.0, min(1.0, float(opacity)))
    if clamped_opacity == 0.0:
        return background_image[..., :3].clone()

    background_batch = int(background_image.shape[0])
    overlay_batch = int(overlay_image.shape[0])
    if overlay_batch not in {1, background_batch}:
        raise RuntimeError(
            f"[LLS] overlay_image batch must be 1 or match background_image batch; got {overlay_batch} and {background_batch}."
        )

    results = []
    for index in range(background_batch):
        background_sample = background_image[index]
        overlay_sample = overlay_image[0 if overlay_batch == 1 else index]
        background_rgba = _tensor_sample_to_rgba_pil(background_sample)
        overlay_rgba = _scaled_overlay_rgba(_tensor_sample_to_rgba_pil(overlay_sample), clamped_scale)
        top_left_x, top_left_y = _resolve_top_left(
            overlay_rgba.size[0],
            overlay_rgba.size[1],
            int(x_offset),
            int(y_offset),
            anchor_mode,
        )

        overlay_canvas = Image.new("RGBA", background_rgba.size, (0, 0, 0, 0))
        overlay_canvas.paste(overlay_rgba, (top_left_x, top_left_y), overlay_rgba)

        if float(rotation) != 0.0:
            rotation_center = _resolve_rotation_center(
                top_left_x,
                top_left_y,
                overlay_rgba.size[0],
                overlay_rgba.size[1],
                rotation_origin_mode,
            )
            overlay_canvas = overlay_canvas.rotate(
                float(rotation),
                resample=Image.BICUBIC,
                center=rotation_center,
                expand=False,
            )

        if clamped_opacity < 1.0:
            overlay_alpha = overlay_canvas.getchannel("A")
            overlay_alpha = overlay_alpha.point(lambda value: int(round(value * clamped_opacity)))
            overlay_canvas.putalpha(overlay_alpha)

        composed = Image.alpha_composite(background_rgba, overlay_canvas)
        results.append(
            _pil_to_tensor(
                composed,
                device=background_image.device,
                dtype=background_image.dtype,
            )
        )

    return torch.stack(results, dim=0)
```

- [ ] **Step 4: Run the backend tests to verify they pass**

Run: `python3 -m unittest tests.test_image_composite_node`
Expected: PASS


### Task 4: Implement The Node Class And Merge Registration In `image/__init__.py`

**Files:**
- Create: `image/composite.py`
- Modify: `image/__init__.py`
- Test: `tests/test_image_composite_registration.py`

- [ ] **Step 1: Create the node class file**

```python
from __future__ import annotations

from .composite_utils import (
    ANCHOR_MODE_CHOICES,
    BLEND_MODE_CHOICES,
    ROTATION_ORIGIN_MODE_CHOICES,
    composite_images,
)


class LLSSimpleImageComposite:
    CATEGORY = "LLS/Image"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    DESCRIPTION = "Composite an overlay image onto a background image with translation, scale, rotation, and opacity."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "anchor_mode": (ANCHOR_MODE_CHOICES, {"default": "top_left"}),
                "rotation_origin_mode": (ROTATION_ORIGIN_MODE_CHOICES, {"default": "center"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (BLEND_MODE_CHOICES, {"default": "normal"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 32.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "keep_aspect": ("BOOLEAN", {"default": True}),
            }
        }

    def composite(
        self,
        background_image,
        overlay_image,
        x_offset,
        y_offset,
        anchor_mode,
        rotation_origin_mode,
        opacity,
        blend_mode,
        scale,
        rotation,
        keep_aspect,
    ):
        output_image = composite_images(
            background_image,
            overlay_image,
            x_offset=x_offset,
            y_offset=y_offset,
            anchor_mode=anchor_mode,
            rotation_origin_mode=rotation_origin_mode,
            opacity=opacity,
            blend_mode=blend_mode,
            scale=scale,
            rotation=rotation,
            keep_aspect=keep_aspect,
        )
        return (output_image,)


NODE_CLASS_MAPPINGS = {"LLSSimpleImageComposite": LLSSimpleImageComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleImageComposite": "LLS Simple Image Composite"}
```

- [ ] **Step 2: Merge the new node registration in `image/__init__.py`**

```python
"""
node.image
==========
功能域：图像处理与后处理（对应功能分类总览第 5 节）
"""
from .composite import (
    NODE_CLASS_MAPPINGS as COMPOSITE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes import (
    NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CORE_NODE_DISPLAY_NAME_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(CORE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(COMPOSITE_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(CORE_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 3: Run the registration tests to verify they pass**

Run: `python3 -m unittest tests.test_image_composite_registration`
Expected: PASS

- [ ] **Step 4: Commit the backend node work**

```bash
git add image/composite.py image/composite_utils.py image/__init__.py \
  tests/test_image_composite_helpers.py tests/test_image_composite_registration.py tests/test_image_composite_node.py
git commit -m "Add LLS Simple Image Composite backend"
```


### Task 5: Implement The ComfyUI Frontend Preview Extension

**Files:**
- Create: `web/js/lls_image_composite.js`
- Test: `tests/test_image_composite_frontend.py`

- [ ] **Step 1: Create the frontend extension skeleton and source-resolution helpers**

```javascript
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "LLS.ImageComposite";
const TARGET_NODE_CLASS = "LLSSimpleImageComposite";
const TARGET_NODE_DISPLAY_NAME = "LLS Simple Image Composite";

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function getInputLink(node, inputName) {
  const input = node.inputs?.find((item) => item.name === inputName);
  if (!input || input.link == null) {
    return null;
  }
  return node.graph?.links?.[input.link] ?? null;
}

function getUpstreamNode(node, inputName) {
  const link = getInputLink(node, inputName);
  if (!link) {
    return null;
  }
  return node.graph?.getNodeById?.(link.origin_id) ?? null;
}

function buildViewUrl(filename, type = "input") {
  if (!filename) {
    return null;
  }
  const relativePath = `/view?filename=${encodeURIComponent(String(filename))}&type=${encodeURIComponent(type)}`;
  return typeof api?.apiURL === "function" ? api.apiURL(relativePath) : relativePath;
}

function resolveImageSource(node, inputName) {
  const upstream = getUpstreamNode(node, inputName);
  const imageWidget = upstream?.widgets?.find((widget) => widget.name === "image");
  if (!imageWidget?.value) {
    return null;
  }
  const className = String(upstream?.comfyClass || upstream?.type || "");
  if (className === "LoadImage") {
    return buildViewUrl(imageWidget.value, "input");
  }
  if (className === "LoadImageOutput") {
    return buildViewUrl(imageWidget.value, "output");
  }
  return null;
}
```

- [ ] **Step 2: Implement preview rendering and drag-to-offset synchronization**

```javascript
function renderCompositePreview(context, state) {
  const { backgroundImage, overlayImage, xOffset, yOffset, anchorMode, rotationOriginMode, opacity, scale, rotation } = state;
  if (!backgroundImage || !overlayImage) {
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.fillStyle = "#1b1b1b";
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    context.fillStyle = "#9a9a9a";
    context.fillText("Connect Load Image / Load Image Output for preview.", 14, 24);
    return;
  }

  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  context.drawImage(backgroundImage, 0, 0, context.canvas.width, context.canvas.height);

  const scaledWidth = overlayImage.width * scale;
  const scaledHeight = overlayImage.height * scale;
  const topLeftX = anchorMode === "center" ? xOffset - (scaledWidth / 2) : xOffset;
  const topLeftY = anchorMode === "center" ? yOffset - (scaledHeight / 2) : yOffset;
  const rotationCenterX = rotationOriginMode === "center" ? (topLeftX + scaledWidth / 2) : topLeftX;
  const rotationCenterY = rotationOriginMode === "center" ? (topLeftY + scaledHeight / 2) : topLeftY;

  context.save();
  context.globalAlpha = opacity;
  context.translate(rotationCenterX, rotationCenterY);
  context.rotate((rotation * Math.PI) / 180);
  context.translate(-rotationCenterX, -rotationCenterY);
  context.drawImage(overlayImage, topLeftX, topLeftY, scaledWidth, scaledHeight);
  context.restore();
}
```

```javascript
app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedPatched() {
      onNodeCreated?.apply(this, arguments);

      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 220;
      const context = canvas.getContext("2d");

      let dragging = false;
      let dragStart = null;

      canvas.addEventListener("pointerdown", (event) => {
        dragging = true;
        dragStart = { x: event.clientX, y: event.clientY };
        canvas.setPointerCapture?.(event.pointerId);
      });

      canvas.addEventListener("pointermove", (event) => {
        if (!dragging || !dragStart) {
          return;
        }
        const xWidget = findWidget(this, "x_offset");
        const yWidget = findWidget(this, "y_offset");
        const deltaX = Math.round(event.clientX - dragStart.x);
        const deltaY = Math.round(event.clientY - dragStart.y);
        xWidget.value = Number(xWidget.value || 0) + deltaX;
        yWidget.value = Number(yWidget.value || 0) + deltaY;
        dragStart = { x: event.clientX, y: event.clientY };
        this.setDirtyCanvas?.(true, true);
      });

      canvas.addEventListener("pointerup", () => {
        dragging = false;
        dragStart = null;
      });

      this.addDOMWidget("preview", "preview", canvas, {
        serialize: false,
        hideOnZoom: false,
      });
    };
  },
});
```

- [ ] **Step 3: Run the frontend tests to verify they pass**

Run: `python3 -m unittest tests.test_image_composite_frontend`
Expected: PASS

- [ ] **Step 4: Commit the frontend work**

```bash
git add web/js/lls_image_composite.js tests/test_image_composite_frontend.py
git commit -m "Add LLS image composite preview extension"
```


### Task 6: Update README And Verify Documentation Coverage

**Files:**
- Modify: `README.md`
- Test: `tests/test_image_composite_docs.py`

- [ ] **Step 1: Add the new README section**

```markdown
### `LLS Simple Image Composite`

`LLS Simple Image Composite` 用于把一张前景图叠加到背景图上，并输出最终合成图像。背景图尺寸始终作为最终输出尺寸，前景图支持平移、缩放、旋转和透明度控制。

**输入：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `background_image` | `IMAGE` | 背景图 |
| `overlay_image` | `IMAGE` | 前景图 |
| `x_offset` | `INT` | 前景图水平偏移 |
| `y_offset` | `INT` | 前景图垂直偏移 |
| `anchor_mode` | `top_left / center` | 前景图定位方式 |
| `rotation_origin_mode` | `top_left / center` | 前景图旋转原点 |
| `opacity` | `FLOAT` | 前景图透明度 |
| `scale` | `FLOAT` | 前景图缩放倍率 |
| `rotation` | `FLOAT` | 前景图旋转角度 |
| `blend_mode` | `normal` | 当前只支持 normal |
| `keep_aspect` | `BOOLEAN` | 当前保持等比缩放 |

**输出：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `output_image` | `IMAGE` | 合成后的最终图像 |

**典型用途：**

- `Load Image(background) + Load Image(overlay) -> LLS Simple Image Composite -> Preview Image`
- 把 logo 贴到背景图指定位置
- 拖动前景图实时预览位置、缩放和旋转

**说明：**

- 前景图超出背景范围时会自动裁剪
- 如果前景图带 alpha 通道，会优先按 alpha 合成
- 节点内实时预览优先支持 `Load Image` / `Load Image Output` 作为上游文件来源
```

- [ ] **Step 2: Run the docs tests to verify they pass**

Run: `python3 -m unittest tests.test_image_composite_docs`
Expected: PASS


### Task 7: Run Full Verification And Final Commit

**Files:**
- Modify: `docs/superpowers/plans/2026-05-25-lls-simple-image-composite.md`
- Test: `tests/test_image_composite_registration.py`
- Test: `tests/test_image_composite_node.py`
- Test: `tests/test_image_composite_frontend.py`
- Test: `tests/test_image_composite_docs.py`

- [ ] **Step 1: Run the focused image composite test suite**

Run: `python3 -m unittest tests.test_image_composite_registration tests.test_image_composite_node tests.test_image_composite_frontend tests.test_image_composite_docs`
Expected: PASS

- [ ] **Step 2: Run the full repository test suite**

Run: `python3 -m unittest discover -s tests`
Expected: PASS with `0 failed`

- [ ] **Step 3: Commit the completed feature**

```bash
git add README.md image/__init__.py image/composite.py image/composite_utils.py \
  web/js/lls_image_composite.js \
  tests/test_image_composite_helpers.py tests/test_image_composite_registration.py \
  tests/test_image_composite_node.py tests/test_image_composite_frontend.py tests/test_image_composite_docs.py \
  docs/superpowers/plans/2026-05-25-lls-simple-image-composite.md
git commit -m "Add LLS Simple Image Composite node"
```
