# LLS Pro Image Edit / Inpaint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new professional local image edit / inpaint workflow to `LLS-node` with three new nodes, backend-aware `SDXL` and `FLUX` conditioning, auto or manual backend selection, and real masked finish compositing, while leaving the existing `LLSSimple*` repair chain unchanged.

**Architecture:** Add a dedicated `pro_edit/` package that owns the new `Prepare`, `KSampler Bridge`, and `Finish` nodes plus a backend registry for `sdxl` and `flux`. Extend `utils/model_info.py` and `model_loader/nodes.py` so backend routing uses explicit capability metadata, then reuse existing low-level sampling and geometry helpers only where semantics match while keeping professional edit behavior separate from the simplified repair path.

**Tech Stack:** Python, unittest, ComfyUI node definitions, existing JSON/model-info helpers, fake tensor and mask test doubles, optional torch tensor blending for finish tests, `python3 -m compileall`

---

## File Structure

- Create: `pro_edit/__init__.py`
  - Export professional edit node registration maps.
- Create: `pro_edit/pro_edit_prepare.py`
  - Define `LLSProImageEditPrepare`.
- Create: `pro_edit/pro_edit_bridge.py`
  - Define `LLSProKSamplerBridge`.
- Create: `pro_edit/pro_edit_finish.py`
  - Define `LLSProImageEditFinish`.
- Create: `pro_edit/pro_edit_utils.py`
  - Normalize `edit_info`, prepare work areas, apply backend-native concat conditioning payloads, and composite final preview images.
- Create: `pro_edit/backends/__init__.py`
  - Import and register built-in professional edit backends.
- Create: `pro_edit/backends/base.py`
  - Define the backend contract and routing record helpers.
- Create: `pro_edit/backends/registry.py`
  - Resolve `auto | sdxl | flux` backend routing and validate capability mismatches.
- Create: `pro_edit/backends/sdxl.py`
  - Implement SDXL-native edit preparation and bridge conditioning.
- Create: `pro_edit/backends/flux.py`
  - Implement FLUX-native edit preparation and bridge conditioning.
- Create: `tests/test_pro_edit_helpers.py`
  - Shared plugin loader, fake image and mask objects, fake model and VAE objects, and small conditioning builders.
- Create: `tests/test_pro_edit_registration.py`
  - Verify plugin registration and professional node schemas.
- Create: `tests/test_pro_edit_capabilities.py`
  - Verify capability inference, normalization, and loader-side tagging payloads.
- Create: `tests/test_pro_edit_registry.py`
  - Verify auto routing, manual override, and mismatch failures.
- Create: `tests/test_pro_edit_prepare_sdxl.py`
  - Verify SDXL prepare behavior and native concat conditioning payloads.
- Create: `tests/test_pro_edit_prepare_flux.py`
  - Verify FLUX prepare behavior and edit-capable routing.
- Create: `tests/test_pro_edit_bridge.py`
  - Verify bridge denoise behavior, backend validation, and sample-info output.
- Create: `tests/test_pro_edit_finish.py`
  - Verify region, crop, and canvas compositing plus preview modes.
- Create: `tests/test_pro_edit_docs.py`
  - Verify README covers the new professional edit workflow.
- Modify: `__init__.py`
  - Append `pro_edit` to `_SUBPACKAGES`.
- Modify: `utils/model_info.py`
  - Add professional edit capability inference and resolution helpers.
- Modify: `model_loader/nodes.py`
  - Tag loaded objects with edit capability metadata.
- Modify: `README.md`
  - Document the new professional workflow, auto or manual backend routing, capability requirements, and model-onboarding guidance.

### Task 1: Scaffold the `pro_edit` Package and Registration Contract

**Files:**
- Create: `tests/test_pro_edit_helpers.py`
- Create: `tests/test_pro_edit_registration.py`
- Create: `pro_edit/__init__.py`
- Create: `pro_edit/pro_edit_prepare.py`
- Create: `pro_edit/pro_edit_bridge.py`
- Create: `pro_edit/pro_edit_finish.py`
- Create: `pro_edit/pro_edit_utils.py`
- Create: `pro_edit/backends/__init__.py`
- Create: `pro_edit/backends/base.py`
- Create: `pro_edit/backends/registry.py`
- Create: `pro_edit/backends/sdxl.py`
- Create: `pro_edit/backends/flux.py`
- Modify: `__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_helpers.py
import importlib
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_pro_edit"
_UNSET = object()

try:
    import torch
except Exception:
    torch = None


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


def import_plugin_submodule(plugin, dotted_name: str):
    return importlib.import_module(f"{plugin.__name__}.{dotted_name}")


class FakeTensor:
    def __init__(self, shape, label="image", crop_box=None, fill_mode=None, original_box=None):
        self.shape = tuple(shape)
        self.label = label
        self.crop_box = crop_box
        self.fill_mode = fill_mode
        self.original_box = original_box

    def cropped(self, x1, y1, x2, y2):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, max(0, y2 - y1), max(0, x2 - x1), channels),
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=(x1, y1, x2, y2),
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )

    def resized(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:resized[{width}x{height}]",
            crop_box=self.crop_box,
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )

    def canvas_expanded(self, width, height, fill_mode="edge", original_box=None):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:canvas[{fill_mode}]",
            crop_box=self.crop_box,
            fill_mode=fill_mode,
            original_box=original_box,
        )

    def masked_fill(self, mask, fill_value):
        del mask
        return FakeTensor(
            self.shape,
            label=f"{self.label}:masked[{fill_value}]",
            crop_box=self.crop_box,
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )


class FakeMask:
    def __init__(self, shape, mask_bbox=None, mask_area_ratio=0.0, label="mask", crop_box=None):
        self.shape = tuple(shape)
        self.mask_bbox = mask_bbox
        self.mask_area_ratio = float(mask_area_ratio)
        self.label = label
        self.crop_box = crop_box

    def _clone(self, shape=None, mask_bbox=_UNSET, mask_area_ratio=None, label=None, crop_box=None):
        return FakeMask(
            shape or self.shape,
            self.mask_bbox if mask_bbox is _UNSET else mask_bbox,
            self.mask_area_ratio if mask_area_ratio is None else mask_area_ratio,
            self.label if label is None else label,
            self.crop_box if crop_box is None else crop_box,
        )

    def _current_size(self):
        return self.shape[2], self.shape[1]

    @staticmethod
    def _clamp_box(box, width, height):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(x1, min(int(x2), width))
        y2 = max(y1, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_area_ratio(box, width, height):
        if box is None or width <= 0 or height <= 0:
            return 0.0
        return ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)

    def cropped(self, x1, y1, x2, y2):
        batch, _, _ = self.shape
        new_width = max(0, x2 - x1)
        new_height = max(0, y2 - y1)
        new_bbox = None
        if self.mask_bbox is not None:
            new_bbox = self._clamp_box(
                (
                    self.mask_bbox[0] - x1,
                    self.mask_bbox[1] - y1,
                    self.mask_bbox[2] - x1,
                    self.mask_bbox[3] - y1,
                ),
                new_width,
                new_height,
            )
        return FakeMask(
            (batch, new_height, new_width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, new_width, new_height),
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=(x1, y1, x2, y2),
        )

    def resized(self, width, height):
        batch, _, _ = self.shape
        source_width, source_height = self._current_size()
        new_bbox = None
        if self.mask_bbox is not None and source_width > 0 and source_height > 0:
            scale_x = width / float(source_width)
            scale_y = height / float(source_height)
            new_bbox = self._clamp_box(
                (
                    round(self.mask_bbox[0] * scale_x),
                    round(self.mask_bbox[1] * scale_y),
                    round(self.mask_bbox[2] * scale_x),
                    round(self.mask_bbox[3] * scale_y),
                ),
                width,
                height,
            )
        return FakeMask(
            (batch, height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:resized[{width}x{height}]",
            crop_box=self.crop_box,
        )

    def canvas_expanded(self, width, height, original_box=None):
        batch, _, _ = self.shape
        if original_box is None:
            original_box = (0, 0, self.shape[2], self.shape[1])
        offset_x, offset_y = original_box[0], original_box[1]
        new_bbox = None
        if self.mask_bbox is not None:
            new_bbox = self._clamp_box(
                (
                    self.mask_bbox[0] + offset_x,
                    self.mask_bbox[1] + offset_y,
                    self.mask_bbox[2] + offset_x,
                    self.mask_bbox[3] + offset_y,
                ),
                width,
                height,
            )
        return FakeMask(
            (batch, height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:canvas",
            crop_box=self.crop_box,
        )

    def normalized(self):
        return self._clone(label=f"{self.label}:normalize")

    def inverted(self, image_size):
        width, height = image_size
        inverted_ratio = max(0.0, min(1.0, 1.0 - self.mask_area_ratio))
        inverted_bbox = None if inverted_ratio <= 0.0 else (0, 0, width, height)
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=inverted_bbox,
            mask_area_ratio=inverted_ratio,
            label=f"{self.label}:invert",
        )

    def thresholded(self, threshold):
        width, height = self._current_size()
        if self.mask_bbox is None:
            return self._clone(label=f"{self.label}:threshold[{threshold}]")
        shrink_ratio = max(0.0, min(1.0, 1.0 - max(0.0, float(threshold) - 0.5)))
        if shrink_ratio <= 0.0:
            return self._clone(mask_bbox=None, mask_area_ratio=0.0, label=f"{self.label}:threshold[{threshold}]")
        x1, y1, x2, y2 = self.mask_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_width = max(1.0, (x2 - x1) * shrink_ratio / 2.0)
        half_height = max(1.0, (y2 - y1) * shrink_ratio / 2.0)
        new_bbox = self._clamp_box(
            (
                round(cx - half_width),
                round(cy - half_height),
                round(cx + half_width),
                round(cy + half_height),
            ),
            width,
            height,
        )
        return self._clone(
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:threshold[{threshold}]",
        )

    def grown(self, amount, image_size):
        width, height = image_size
        if self.mask_bbox is None:
            return self._clone(shape=(self.shape[0], height, width), label=f"{self.label}:grow[{amount}]")
        x1, y1, x2, y2 = self.mask_bbox
        new_bbox = self._clamp_box((x1 - amount, y1 - amount, x2 + amount, y2 + amount), width, height)
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:grow[{amount}]",
        )

    def blurred(self, radius, image_size):
        blur_expand = max(1, int(round(float(radius) / 4.0)))
        return self.grown(blur_expand, image_size)._clone(label=f"{self.label}:blur[{radius}]")

    def merged_with(self, other, image_size):
        width, height = image_size
        if self.mask_bbox is None:
            return other._clone(shape=(other.shape[0], height, width), label=f"{self.label}:merge:{other.label}")
        if other.mask_bbox is None:
            return self._clone(shape=(self.shape[0], height, width), label=f"{self.label}:merge:{other.label}")
        new_bbox = self._clamp_box(
            (
                min(self.mask_bbox[0], other.mask_bbox[0]),
                min(self.mask_bbox[1], other.mask_bbox[1]),
                max(self.mask_bbox[2], other.mask_bbox[2]),
                max(self.mask_bbox[3], other.mask_bbox[3]),
            ),
            width,
            height,
        )
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:merge:{other.label}",
        )

    @classmethod
    def canvas_region(cls, shape, original_box, label="canvas-region"):
        batch, height, width = shape
        original_width = max(0, original_box[2] - original_box[0])
        original_height = max(0, original_box[3] - original_box[1])
        total_area = max(1, width * height)
        original_area = max(0, original_width * original_height)
        area_ratio = max(0.0, min(1.0, (total_area - original_area) / float(total_area)))
        bbox = None if area_ratio <= 0.0 else (0, 0, width, height)
        return cls(shape, mask_bbox=bbox, mask_area_ratio=area_ratio, label=label)


class FakeLatentTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVAE:
    def __init__(self, latent_channels=4, downscale_ratio=8):
        self.latent_channels = int(latent_channels)
        self.downscale_ratio = int(downscale_ratio)

    def encode(self, image):
        batch, height, width, _ = image.shape
        latent_height = max(1, height // self.downscale_ratio)
        latent_width = max(1, width // self.downscale_ratio)
        return FakeLatentTensor((batch, self.latent_channels, latent_height, latent_width))


class FakeModel:
    def __init__(
        self,
        family="SDXL",
        model_role="base",
        supports_inpaint_native=False,
        supports_image_edit_native=False,
        preferred_edit_backend=None,
        model_name="demo-model.safetensors",
    ):
        self._lls_family = family
        self._lls_model_role = model_role
        self._lls_supports_inpaint_native = supports_inpaint_native
        self._lls_supports_image_edit_native = supports_image_edit_native
        self._lls_preferred_edit_backend = preferred_edit_backend
        self._lls_model_name = model_name


def make_conditioning(label):
    return [[label, {"origin": label}]]


def make_torch_image(width: int, height: int, value: float):
    if torch is None:
        raise RuntimeError("torch is required for this helper")
    return torch.full((1, height, width, 3), float(value), dtype=torch.float32)


def make_torch_mask(width: int, height: int, box):
    if torch is None:
        raise RuntimeError("torch is required for this helper")
    x1, y1, x2, y2 = box
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[:, y1:y2, x1:x2] = 1.0
    return mask
```

```python
# tests/test_pro_edit_registration.py
import unittest

try:
    from .test_pro_edit_helpers import import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import import_plugin_submodule, load_plugin_package


class TestProEditRegistration(unittest.TestCase):
    def test_plugin_registers_pro_edit_nodes(self):
        plugin = load_plugin_package()

        self.assertIn("LLSProImageEditPrepare", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSProKSamplerBridge", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSProImageEditFinish", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProImageEditPrepare"],
            "LLS Pro Image Edit Prepare",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProKSamplerBridge"],
            "LLS Pro KSampler Bridge",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSProImageEditFinish"],
            "LLS Pro Image Edit Finish",
        )

    def test_prepare_node_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(node_cls.FUNCTION, "prepare")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("LATENT", "IMAGE", "MASK", "LLS_EDIT_INFO", "FLOAT", "CONDITIONING", "CONDITIONING"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("latent", "work_image", "work_mask", "edit_info", "recommended_denoise", "positive", "negative"),
        )

        required = schema["required"]
        optional = schema["optional"]

        for field in (
            "image",
            "mask",
            "vae",
            "positive",
            "negative",
            "backend_mode",
            "edit_scope",
            "mask_grow",
            "mask_blur",
            "mask_threshold",
            "invert_mask",
            "crop_context",
            "crop_context_factor",
            "min_size",
            "max_size",
            "resize_mode",
            "expand_left",
            "expand_right",
            "expand_top",
            "expand_bottom",
            "canvas_fill",
            "auto_recommend",
        ):
            self.assertIn(field, required)

        for field in ("model", "model_info"):
            self.assertIn(field, optional)

        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(required["mask"], ("MASK",))
        self.assertEqual(required["vae"], ("VAE",))
        self.assertEqual(required["positive"], ("CONDITIONING",))
        self.assertEqual(required["negative"], ("CONDITIONING",))
        self.assertEqual(schema["required"]["backend_mode"][0], ["auto", "sdxl", "flux"])
        self.assertEqual(schema["required"]["edit_scope"][0], ["auto", "region", "crop", "canvas"])
        self.assertEqual(optional["model"], ("MODEL",))
        self.assertEqual(optional["model_info"], ("STRING",))

    def test_bridge_and_finish_schemas_match_contract(self):
        plugin = load_plugin_package()

        bridge_cls = plugin.NODE_CLASS_MAPPINGS["LLSProKSamplerBridge"]
        bridge_schema = bridge_cls.INPUT_TYPES()
        self.assertEqual(bridge_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(bridge_cls.FUNCTION, "sample")
        self.assertEqual(bridge_cls.RETURN_TYPES, ("LATENT", "STRING"))
        self.assertEqual(bridge_cls.RETURN_NAMES, ("latent", "sample_info"))

        bridge_required = bridge_schema["required"]
        bridge_optional = bridge_schema["optional"]

        for field in (
            "model",
            "positive",
            "negative",
            "latent_image",
            "backend_mode",
            "quality_preset",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "denoise_mode",
            "flux_guidance",
            "model_family",
        ):
            self.assertIn(field, bridge_required)

        self.assertEqual(bridge_required["backend_mode"][0], ["auto", "sdxl", "flux"])
        self.assertEqual(bridge_required["denoise_mode"][0], ["manual", "auto_from_edit"])
        self.assertEqual(bridge_optional["edit_info"], ("LLS_EDIT_INFO",))
        self.assertEqual(bridge_optional["model_info"], ("STRING",))

        finish_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditFinish"]
        finish_schema = finish_cls.INPUT_TYPES()
        self.assertEqual(finish_cls.CATEGORY, "LLS/Image Edit")
        self.assertEqual(finish_cls.FUNCTION, "finish")
        self.assertEqual(finish_cls.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(finish_cls.RETURN_NAMES, ("final_image", "preview_image"))

        finish_required = finish_schema["required"]
        finish_optional = finish_schema["optional"]

        for field in (
            "original_image",
            "generated_image",
            "edit_info",
            "feather",
            "color_match",
            "brightness_match",
            "blend_strength",
            "restore_unmasked_area",
            "edge_fix",
            "preview_mode",
        ):
            self.assertIn(field, finish_required)

        self.assertEqual(finish_required["original_image"], ("IMAGE",))
        self.assertEqual(finish_required["generated_image"], ("IMAGE",))
        self.assertEqual(finish_required["edit_info"], ("LLS_EDIT_INFO",))
        self.assertEqual(finish_optional["work_mask"], ("MASK",))
        self.assertEqual(finish_optional["sample_info"], ("STRING",))

    def test_backend_registry_loads_builtin_backends(self):
        plugin = load_plugin_package()
        registry = import_plugin_submodule(plugin, "pro_edit.backends.registry")

        self.assertEqual(registry.get_backend("sdxl").backend_name, "sdxl")
        self.assertEqual(registry.get_backend("flux").backend_name, "flux")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_registration.py' -v`

Expected: FAIL because the `pro_edit` package, registrations, and node classes do not exist yet.

- [ ] **Step 3: Implement the minimal scaffolding**

```python
# __init__.py
_SUBPACKAGES: list[str] = [
    "model_loader",
    "conditioning",
    "sampling",
    "qwen",
    "latent",
    "image",
    "repair",
    "mask_draw",
    "pro_edit",
    "upscale",
    "mask",
    "controlnet",
    "lora",
    "video",
    "audio",
    "utils",
]
```

```python
# pro_edit/__init__.py
from .pro_edit_bridge import (
    NODE_CLASS_MAPPINGS as BRIDGE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BRIDGE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .pro_edit_finish import (
    NODE_CLASS_MAPPINGS as FINISH_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)
from .pro_edit_prepare import (
    NODE_CLASS_MAPPINGS as PREPARE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
)


def _merge_registry_maps(*maps):
    merged = {}
    for mapping in maps:
        overlap = set(merged).intersection(mapping)
        if overlap:
            duplicate_keys = ", ".join(sorted(overlap))
            raise RuntimeError(f"[LLS] Duplicate pro_edit node registration keys: {duplicate_keys}")
        merged.update(mapping)
    return merged


NODE_CLASS_MAPPINGS = _merge_registry_maps(
    PREPARE_NODE_CLASS_MAPPINGS,
    BRIDGE_NODE_CLASS_MAPPINGS,
    FINISH_NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_registry_maps(
    PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
    BRIDGE_NODE_DISPLAY_NAME_MAPPINGS,
    FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

```python
# pro_edit/pro_edit_utils.py
from ..utils.model_info import parse_jsonish_info


EDIT_INFO_TYPE = "LLS_EDIT_INFO"
BACKEND_MODE_CHOICES = ["auto", "sdxl", "flux"]
EDIT_SCOPE_CHOICES = ["auto", "region", "crop", "canvas"]
PREVIEW_MODE_CHOICES = ["final", "compare", "mask", "before_after"]
AUTO_RECOMMEND_CHOICES = ["enabled", "disabled"]
CANVAS_FILL_CHOICES = ["edge", "blur", "black", "white", "neutral"]
COLOR_MATCH_CHOICES = ["disabled", "mean_std", "histogram_simple"]
BRIGHTNESS_MATCH_CHOICES = ["disabled", "enabled"]
EDGE_FIX_CHOICES = ["none", "soft", "strong"]


def normalize_edit_info(edit_info):
    info = parse_jsonish_info(edit_info)
    info.setdefault("backend_name", "")
    info.setdefault("routing_reason", "")
    info.setdefault("edit_scope", "region")
    info.setdefault("edit_payload_version", "1.0")
    return info
```

```python
# pro_edit/backends/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]


class ProEditBackend(Protocol):
    backend_name: str

    def supports(self, capabilities: dict[str, Any]) -> bool:
        raise NotImplementedError

    def prepare(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_bridge(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError
```

```python
# pro_edit/backends/registry.py
from __future__ import annotations

from .base import RoutingResult


_BACKENDS = {}


def register_backend(backend):
    _BACKENDS[backend.backend_name] = backend
    return backend


def get_backend(name: str):
    backend = _BACKENDS.get(str(name or "").strip())
    if backend is None:
        raise RuntimeError(f"[LLS] Unknown pro edit backend '{name}'.")
    return backend


def resolve_backend(backend_mode: str, **_kwargs):
    mode = str(backend_mode or "auto")
    if mode == "auto":
        raise RuntimeError("[LLS] Pro edit backend auto routing is not implemented yet.")
    backend = get_backend(mode)
    return backend, RoutingResult(backend.backend_name, "manual", {})


from . import flux, sdxl  # noqa: F401,E402
```

```python
# pro_edit/backends/sdxl.py
from .registry import register_backend


class SDXLProEditBackend:
    backend_name = "sdxl"

    def supports(self, capabilities):
        return str(capabilities.get("model_family") or "").startswith("SDXL")

    def prepare(self, **kwargs):
        raise RuntimeError("[LLS] SDXL pro edit backend is not implemented yet.")

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(SDXLProEditBackend())
```

```python
# pro_edit/backends/flux.py
from .registry import register_backend


class FluxProEditBackend:
    backend_name = "flux"

    def supports(self, capabilities):
        return str(capabilities.get("model_family") or "").startswith("FLUX")

    def prepare(self, **kwargs):
        raise RuntimeError("[LLS] FLUX pro edit backend is not implemented yet.")

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(FluxProEditBackend())
```

```python
# pro_edit/backends/__init__.py
from . import flux, sdxl

__all__ = ["flux", "sdxl"]
```

```python
# pro_edit/pro_edit_prepare.py
from .pro_edit_utils import AUTO_RECOMMEND_CHOICES, BACKEND_MODE_CHOICES, CANVAS_FILL_CHOICES, EDIT_SCOPE_CHOICES


class LLSProImageEditPrepare:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_EDIT_INFO", "FLOAT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "edit_info", "recommended_denoise", "positive", "negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "edit_scope": (EDIT_SCOPE_CHOICES, {"default": "auto"}),
                "mask_grow": ("INT", {"default": 24, "min": 0, "max": 2048}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("INT", {"default": 64, "min": 0, "max": 512}),
                "crop_context_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "min_size": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (["fit", "pad", "stretch"], {"default": "fit"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "canvas_fill": (CANVAS_FILL_CHOICES, {"default": "edge"}),
                "auto_recommend": (AUTO_RECOMMEND_CHOICES, {"default": "enabled"}),
            },
            "optional": {
                "model": ("MODEL",),
                "model_info": ("STRING",),
            },
        }

    def prepare(self, **_kwargs):
        raise RuntimeError("[LLS] LLS Pro Image Edit Prepare is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSProImageEditPrepare": LLSProImageEditPrepare}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProImageEditPrepare": "LLS Pro Image Edit Prepare"}
```

```python
# pro_edit/pro_edit_bridge.py
from .pro_edit_utils import BACKEND_MODE_CHOICES
from ..sampling.nodes import _QUALITY_PRESETS, _get_samplers, _get_schedulers
from ..utils.model_info import MODEL_FAMILY_CHOICES


class LLSProKSamplerBridge:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sample_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "quality_preset": (_QUALITY_PRESETS, {"default": "Family Default"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_samplers(), {"default": "euler"}),
                "scheduler": (_get_schedulers(), {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (["manual", "auto_from_edit"], {"default": "manual"}),
                "flux_guidance": ("STRING,FLOAT,INT", {"default": 3.5, "widgetType": "FLOAT"}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "edit_info": ("LLS_EDIT_INFO",),
                "model_info": ("STRING",),
            },
        }

    def sample(self, **_kwargs):
        raise RuntimeError("[LLS] LLS Pro KSampler Bridge is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSProKSamplerBridge": LLSProKSamplerBridge}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProKSamplerBridge": "LLS Pro KSampler Bridge"}
```

```python
# pro_edit/pro_edit_finish.py
from .pro_edit_utils import (
    BRIGHTNESS_MATCH_CHOICES,
    COLOR_MATCH_CHOICES,
    EDGE_FIX_CHOICES,
    PREVIEW_MODE_CHOICES,
)


class LLSProImageEditFinish:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "edit_info": ("LLS_EDIT_INFO",),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "color_match": (COLOR_MATCH_CHOICES, {"default": "disabled"}),
                "brightness_match": (BRIGHTNESS_MATCH_CHOICES, {"default": "enabled"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (EDGE_FIX_CHOICES, {"default": "soft"}),
                "preview_mode": (PREVIEW_MODE_CHOICES, {"default": "final"}),
            },
            "optional": {
                "work_mask": ("MASK",),
                "sample_info": ("STRING",),
            },
        }

    def finish(self, **_kwargs):
        raise RuntimeError("[LLS] LLS Pro Image Edit Finish is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSProImageEditFinish": LLSProImageEditFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSProImageEditFinish": "LLS Pro Image Edit Finish"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_registration.py' -v`

Expected: PASS with three tests.

- [ ] **Step 5: Commit**

```bash
git add __init__.py pro_edit tests/test_pro_edit_helpers.py tests/test_pro_edit_registration.py
git commit -m "feat: scaffold pro image edit nodes"
```

### Task 2: Add Edit Capability Metadata Inference and Loader Tagging

**Files:**
- Create: `tests/test_pro_edit_capabilities.py`
- Modify: `utils/model_info.py`
- Modify: `model_loader/nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_capabilities.py
import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestProEditCapabilities(unittest.TestCase):
    def test_parse_model_info_infers_sdxl_inpaint_capability_from_name(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "demo-sdxl-inpaint.safetensors",
                "family": "SDXL",
            }
        )

        self.assertEqual(info["model_role"], "inpaint")
        self.assertTrue(info["supports_inpaint_native"])
        self.assertFalse(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "sdxl")

    def test_parse_model_info_infers_flux_edit_capability_from_name(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "demo-flux-fill-dev.safetensors",
                "family": "FLUX_DEV",
            }
        )

        self.assertEqual(info["model_role"], "fill")
        self.assertFalse(info["supports_inpaint_native"])
        self.assertTrue(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "flux")

    def test_explicit_capability_values_override_name_inference(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "model_role": "edit",
                "supports_inpaint_native": True,
                "supports_image_edit_native": True,
                "preferred_edit_backend": "sdxl",
            }
        )

        self.assertEqual(info["model_role"], "edit")
        self.assertTrue(info["supports_inpaint_native"])
        self.assertTrue(info["supports_image_edit_native"])
        self.assertEqual(info["preferred_edit_backend"], "sdxl")

    def test_resolve_edit_capabilities_reads_model_tags(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            model_name="demo-flux-edit.safetensors",
        )

        capabilities = model_info.resolve_edit_capabilities(model=model, model_info=None)

        self.assertEqual(capabilities["model_family"], "FLUX_DEV")
        self.assertEqual(capabilities["model_role"], "edit")
        self.assertTrue(capabilities["supports_image_edit_native"])
        self.assertEqual(capabilities["preferred_edit_backend"], "flux")

    def test_loader_builds_capability_tags_for_loaded_objects(self):
        plugin = load_plugin_package()
        loader_module = import_plugin_submodule(plugin, "model_loader.nodes")

        tags = loader_module._build_capability_tags("demo-sdxl-inpaint.safetensors", "SDXL")

        self.assertEqual(tags["model_role"], "inpaint")
        self.assertTrue(tags["supports_inpaint_native"])
        self.assertEqual(tags["preferred_edit_backend"], "sdxl")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_capabilities.py' -v`

Expected: FAIL because capability inference helpers and loader capability tags are not implemented.

- [ ] **Step 3: Implement capability inference and loader tagging**

```python
# utils/model_info.py
_ROLE_KEYWORDS = (
    ("inpaint", "inpaint"),
    ("img2img", "edit"),
    ("imageedit", "edit"),
    ("image-edit", "edit"),
    ("edit", "edit"),
    ("fill", "fill"),
    ("refiner", "refiner"),
)


def infer_model_role_from_name(model_name: str | None, family: str | None = None) -> str:
    del family
    name = str(model_name or "").lower()
    for needle, role in _ROLE_KEYWORDS:
        if needle in name:
            return role
    return "base"


def infer_edit_capabilities(model_name: str | None, family: str | None = None) -> dict[str, Any]:
    resolved_family = canonicalize_family(family or infer_family_from_name(model_name, "SD1.5"))
    role = infer_model_role_from_name(model_name, resolved_family)

    supports_inpaint_native = False
    supports_image_edit_native = False
    preferred_edit_backend = None

    if is_sdxl_family(resolved_family):
        preferred_edit_backend = "sdxl" if role in {"inpaint", "edit", "fill"} else None
        supports_inpaint_native = role in {"inpaint", "edit", "fill"}
        supports_image_edit_native = role in {"edit", "fill"}
    elif is_flux_family(resolved_family):
        preferred_edit_backend = "flux" if role in {"inpaint", "edit", "fill"} else None
        supports_inpaint_native = role == "inpaint"
        supports_image_edit_native = role in {"inpaint", "edit", "fill"}

    return {
        "model_role": role,
        "supports_inpaint_native": supports_inpaint_native,
        "supports_image_edit_native": supports_image_edit_native,
        "preferred_edit_backend": preferred_edit_backend,
    }


def resolve_edit_capabilities(model=None, model_info: dict[str, Any] | str | None = None) -> dict[str, Any]:
    info = parse_model_info(model_info)
    model_name = str(
        info.get("checkpoint_name")
        or info.get("model_name")
        or get_lls_attr(model, "model_name", "")
        or ""
    )
    family = canonicalize_family(
        info.get("family")
        or info.get("model_family")
        or get_lls_attr(model, "family", None)
        or infer_family_from_model(model)
    )
    inferred = infer_edit_capabilities(model_name, family)
    return {
        "model_family": family,
        "model_name": model_name,
        "model_role": str(
            info.get("model_role")
            or get_lls_attr(model, "model_role", None)
            or inferred["model_role"]
        ),
        "supports_inpaint_native": bool(
            info.get(
                "supports_inpaint_native",
                get_lls_attr(model, "supports_inpaint_native", inferred["supports_inpaint_native"]),
            )
        ),
        "supports_image_edit_native": bool(
            info.get(
                "supports_image_edit_native",
                get_lls_attr(model, "supports_image_edit_native", inferred["supports_image_edit_native"]),
            )
        ),
        "preferred_edit_backend": (
            info.get("preferred_edit_backend")
            or get_lls_attr(model, "preferred_edit_backend", inferred["preferred_edit_backend"])
        ),
    }
```

```python
# utils/model_info.py
def parse_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    family = canonicalize_family(
        raw.get("family")
        or raw.get("model_family")
        or infer_family_from_name(
            raw.get("checkpoint_name") or raw.get("model_name") or raw.get("ckpt_name") or raw.get("ckpt"),
            "SD1.5",
        )
    )
    defaults = get_family_defaults(family)
    capability_defaults = infer_edit_capabilities(
        raw.get("checkpoint_name") or raw.get("model_name"),
        family,
    )

    info: dict[str, Any] = {**defaults}
    info.update(raw)
    info["family"] = family
    info["model_role"] = str(raw.get("model_role", capability_defaults["model_role"]))
    info["supports_inpaint_native"] = bool(
        raw.get("supports_inpaint_native", capability_defaults["supports_inpaint_native"])
    )
    info["supports_image_edit_native"] = bool(
        raw.get("supports_image_edit_native", capability_defaults["supports_image_edit_native"])
    )
    info["preferred_edit_backend"] = raw.get(
        "preferred_edit_backend",
        capability_defaults["preferred_edit_backend"],
    )
    info.setdefault("model_family", family)
    info.setdefault("checkpoint_name", raw.get("checkpoint_name") or raw.get("model_name") or "")
    info.setdefault("model_name", info["checkpoint_name"])
    info.setdefault("vae_source", raw.get("vae_source", "auto"))
    info.setdefault("text_encoder_source", raw.get("text_encoder_source", "auto"))
    info.setdefault("load_mode", raw.get("load_mode", "simple"))
    return info
```

```python
# model_loader/nodes.py
from ..utils.model_info import (
    MODEL_FAMILY_CHOICES,
    build_model_info,
    canonicalize_family,
    get_family_defaults,
    infer_family_from_name,
    is_flux_family,
    is_sdxl_family,
    tag_lls_object,
)


def _build_capability_tags(model_name: str, family: str) -> dict[str, object]:
    info = build_model_info(
        checkpoint_name=model_name,
        model_name=model_name,
        family=family,
    )
    return {
        "model_role": info["model_role"],
        "supports_inpaint_native": info["supports_inpaint_native"],
        "supports_image_edit_native": info["supports_image_edit_native"],
        "preferred_edit_backend": info["preferred_edit_backend"],
    }
```

```python
# model_loader/nodes.py
capability_tags = _build_capability_tags(ckpt_name, family)

tag_lls_object(
    model,
    family=family,
    model_name=ckpt_name,
    checkpoint_name=ckpt_name,
    load_mode=load_mode,
    **capability_tags,
)
tag_lls_object(
    text_encoder,
    family=family,
    model_name=ckpt_name,
    checkpoint_name=ckpt_name,
    text_encoder_type=defaults["text_encoder_type"],
    text_encoder_source=resolved_text_encoder_source,
    text_encoder_name=resolved_text_encoder_name,
    text_encoder_name_1=resolved_text_encoder_name_1,
    text_encoder_name_2=resolved_text_encoder_name_2,
    **capability_tags,
)
tag_lls_object(
    vae,
    family=family,
    model_name=ckpt_name,
    checkpoint_name=ckpt_name,
    vae_name=resolved_vae_label,
    vae_source=resolved_vae_source,
    **capability_tags,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_capabilities.py' -v`

Expected: PASS with five tests.

- [ ] **Step 5: Commit**

```bash
git add utils/model_info.py model_loader/nodes.py tests/test_pro_edit_capabilities.py
git commit -m "feat: add pro edit capability metadata"
```

### Task 3: Build Backend Registry and Routing Validation

**Files:**
- Create: `tests/test_pro_edit_registry.py`
- Modify: `pro_edit/backends/base.py`
- Modify: `pro_edit/backends/registry.py`
- Modify: `pro_edit/backends/sdxl.py`
- Modify: `pro_edit/backends/flux.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_registry.py
import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestProEditRegistry(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "pro_edit.backends.registry")

    def test_auto_routes_sdxl_inpaint_model(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "model.preferred_edit_backend")

    def test_auto_routes_flux_edit_model(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "flux")
        self.assertEqual(routing.routing_reason, "model.preferred_edit_backend")
        self.assertEqual(routing.capabilities["model_family"], "FLUX_DEV")

    def test_auto_reuses_backend_name_from_edit_info(self):
        backend, routing = self.registry.resolve_backend(
            "auto",
            model=None,
            edit_info={
                "backend_name": "sdxl",
                "model_family": "SDXL",
                "model_role": "inpaint",
                "supports_inpaint_native": True,
                "supports_image_edit_native": False,
                "preferred_edit_backend": "sdxl",
            },
        )

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.routing_reason, "edit_info.backend_name")

    def test_manual_flux_override_rejects_sdxl_only_model(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )

        with self.assertRaisesRegex(RuntimeError, "backend 'flux' is incompatible"):
            self.registry.resolve_backend("flux", model=model)

    def test_auto_without_matching_backend_raises_clear_error(self):
        model = FakeModel(
            family="SD1.5",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
        )

        with self.assertRaisesRegex(RuntimeError, "No professional edit backend matched"):
            self.registry.resolve_backend("auto", model=model)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_registry.py' -v`

Expected: FAIL because `resolve_backend` does not yet understand capability records or auto routing.

- [ ] **Step 3: Implement the backend contract and routing rules**

```python
# pro_edit/backends/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]


class ProEditBackend(Protocol):
    backend_name: str

    def supports(self, capabilities: dict[str, Any]) -> bool:
        raise NotImplementedError

    def prepare(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_bridge(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


def validate_manual_backend(backend_name: str, backend: ProEditBackend, capabilities: dict[str, Any]) -> None:
    if not backend.supports(capabilities):
        family = capabilities.get("model_family") or "UNKNOWN"
        raise RuntimeError(
            f"[LLS] Pro edit backend '{backend_name}' is incompatible with model family '{family}'."
        )
```

```python
# pro_edit/backends/registry.py
from __future__ import annotations

from .base import RoutingResult, validate_manual_backend
from ...utils.model_info import resolve_edit_capabilities


_BACKENDS = {}


def register_backend(backend):
    _BACKENDS[backend.backend_name] = backend
    return backend


def get_backend(name: str):
    backend = _BACKENDS.get(str(name or "").strip())
    if backend is None:
        raise RuntimeError(f"[LLS] Unknown pro edit backend '{name}'.")
    return backend


def _normalize_capabilities(model=None, model_info=None, edit_info=None):
    resolved = resolve_edit_capabilities(model=model, model_info=model_info)
    raw_edit_info = dict(edit_info or {})
    if raw_edit_info.get("model_family"):
        resolved["model_family"] = str(raw_edit_info["model_family"])
    if raw_edit_info.get("model_role"):
        resolved["model_role"] = str(raw_edit_info["model_role"])
    if "supports_inpaint_native" in raw_edit_info:
        resolved["supports_inpaint_native"] = bool(raw_edit_info["supports_inpaint_native"])
    if "supports_image_edit_native" in raw_edit_info:
        resolved["supports_image_edit_native"] = bool(raw_edit_info["supports_image_edit_native"])
    if raw_edit_info.get("preferred_edit_backend"):
        resolved["preferred_edit_backend"] = raw_edit_info["preferred_edit_backend"]
    return resolved


def resolve_backend(backend_mode: str, *, model=None, model_info=None, edit_info=None):
    mode = str(backend_mode or "auto")
    capabilities = _normalize_capabilities(model=model, model_info=model_info, edit_info=edit_info)
    raw_edit_info = dict(edit_info or {})

    if mode in {"sdxl", "flux"}:
        backend = get_backend(mode)
        validate_manual_backend(mode, backend, capabilities)
        return backend, RoutingResult(mode, "manual.override", capabilities)

    if raw_edit_info.get("backend_name"):
        backend_name = str(raw_edit_info["backend_name"])
        backend = get_backend(backend_name)
        validate_manual_backend(backend_name, backend, capabilities)
        return backend, RoutingResult(backend_name, "edit_info.backend_name", capabilities)

    preferred = str(capabilities.get("preferred_edit_backend") or "").strip()
    if preferred:
        backend = get_backend(preferred)
        validate_manual_backend(preferred, backend, capabilities)
        return backend, RoutingResult(preferred, "model.preferred_edit_backend", capabilities)

    for backend_name in ("sdxl", "flux"):
        backend = get_backend(backend_name)
        if backend.supports(capabilities):
            return backend, RoutingResult(backend_name, "model.capability_match", capabilities)

    raise RuntimeError(
        "[LLS] No professional edit backend matched the current model capability. "
        "Use an edit-capable SDXL or FLUX model, or provide explicit capability metadata."
    )
```

```python
# pro_edit/backends/sdxl.py
from .registry import register_backend


class SDXLProEditBackend:
    backend_name = "sdxl"

    def supports(self, capabilities):
        family = str(capabilities.get("model_family") or "")
        role = str(capabilities.get("model_role") or "unknown")
        return family.startswith("SDXL") and (
            role in {"inpaint", "edit", "fill"}
            or bool(capabilities.get("supports_inpaint_native"))
            or str(capabilities.get("preferred_edit_backend") or "") == "sdxl"
        )

    def prepare(self, **kwargs):
        raise RuntimeError("[LLS] SDXL pro edit backend is not implemented yet.")

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(SDXLProEditBackend())
```

```python
# pro_edit/backends/flux.py
from .registry import register_backend


class FluxProEditBackend:
    backend_name = "flux"

    def supports(self, capabilities):
        family = str(capabilities.get("model_family") or "")
        role = str(capabilities.get("model_role") or "unknown")
        return family.startswith("FLUX") and (
            role in {"inpaint", "edit", "fill"}
            or bool(capabilities.get("supports_image_edit_native"))
            or str(capabilities.get("preferred_edit_backend") or "") == "flux"
        )

    def prepare(self, **kwargs):
        raise RuntimeError("[LLS] FLUX pro edit backend is not implemented yet.")

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(FluxProEditBackend())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_registry.py' -v`

Expected: PASS with five tests.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/backends tests/test_pro_edit_registry.py
git commit -m "feat: add pro edit backend registry"
```

### Task 4: Implement SDXL Prepare Path and Native Concat Conditioning

**Files:**
- Create: `tests/test_pro_edit_prepare_sdxl.py`
- Modify: `pro_edit/pro_edit_utils.py`
- Modify: `pro_edit/pro_edit_prepare.py`
- Modify: `pro_edit/backends/sdxl.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_prepare_sdxl.py
import unittest

try:
    from .test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )
except ImportError:
    from test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )


class TestProEditPrepareSDXL(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        self.node = node_cls()

    def test_prepare_region_adds_concat_conditioning_for_sdxl(self):
        image = FakeTensor((1, 128, 128, 3), label="image")
        mask = FakeMask((1, 128, 128), mask_bbox=(32, 32, 96, 96), mask_area_ratio=0.25)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="region",
            mask_grow=8,
            mask_blur=4.0,
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
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["backend_name"], "sdxl")
        self.assertEqual(edit_info["routing_reason"], "model.preferred_edit_backend")
        self.assertEqual(edit_info["edit_scope"], "region")
        self.assertEqual(latent["source"], "pro_edit_prepare_region")
        self.assertIn("concat_latent_image", positive[0][1])
        self.assertIn("concat_mask", positive[0][1])
        self.assertIn("concat_latent_image", negative[0][1])
        self.assertIn("concat_mask", negative[0][1])
        self.assertEqual(work_image.shape, image.shape)
        self.assertEqual(work_mask.shape, mask.shape)
        self.assertGreater(recommended, 0.0)

    def test_prepare_crop_writes_crop_geometry(self):
        image = FakeTensor((1, 1024, 1024, 3), label="image")
        mask = FakeMask((1, 1024, 1024), mask_bbox=(128, 128, 256, 256), mask_area_ratio=0.02)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="edit",
            supports_inpaint_native=True,
            supports_image_edit_native=True,
            preferred_edit_backend="sdxl",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="crop",
            mask_grow=0,
            mask_blur=0.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=32,
            crop_context_factor=1.0,
            min_size=256,
            max_size=512,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["edit_scope"], "crop")
        self.assertIsInstance(edit_info["crop_box"], list)
        self.assertEqual(edit_info["work_size"], [work_image.shape[2], work_image.shape[1]])
        self.assertEqual(edit_info["backend_name"], "sdxl")
        self.assertEqual(positive[0][1]["edit_backend"], "sdxl")
        self.assertEqual(negative[0][1]["edit_backend"], "sdxl")
        self.assertGreater(recommended, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_sdxl.py' -v`

Expected: FAIL because the prepare node does not yet build work areas or native SDXL conditioning payloads.

- [ ] **Step 3: Implement shared prepare helpers and SDXL backend behavior**

```python
# pro_edit/pro_edit_utils.py
from __future__ import annotations

from typing import Any

from ..repair.repair_utils import (
    build_canvas_info,
    build_canvas_repair_mask,
    compute_crop_box,
    crop_image_to,
    crop_mask_to,
    expand_canvas_image,
    expand_canvas_mask,
    get_image_size,
    get_mask_metrics,
    make_noise_mask,
    merge_masks,
    preprocess_mask,
    recommend_denoise,
    resize_image_to,
    resize_mask_to,
    resolve_repair_scope,
    resolve_work_size,
)

try:
    import node_helpers
except Exception:
    node_helpers = None

try:
    import torch
except Exception:
    torch = None


def set_conditioning_values(conditioning, values: dict[str, Any]):
    if node_helpers is not None:
        try:
            return node_helpers.conditioning_set_values(conditioning, values)
        except Exception:
            pass

    updated = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            updated.append(entry)
            continue
        token, meta = entry
        merged = dict(meta or {})
        merged.update(values)
        updated.append([token, merged])
    return updated


def build_masked_pixel_image(image, mask, *, fill_value: float):
    if hasattr(image, "masked_fill"):
        return image.masked_fill(mask, fill_value)
    if torch is not None and isinstance(image, torch.Tensor):
        mask_4d = mask.unsqueeze(-1).clamp(0.0, 1.0)
        return (image * (1.0 - mask_4d)) + (fill_value * mask_4d)
    raise RuntimeError("[LLS] image object does not support masked pixel preprocessing.")


def build_native_conditioning_payload(vae, work_image, work_mask, *, latent_source: str, masked_fill_value: float):
    latent_samples = vae.encode(work_image)
    masked_pixels = build_masked_pixel_image(work_image, work_mask, fill_value=masked_fill_value)
    concat_latent_image = vae.encode(masked_pixels)
    concat_mask = make_noise_mask(work_mask, latent_samples)
    latent = {
        "samples": latent_samples,
        "noise_mask": concat_mask,
        "source": latent_source,
    }
    return latent, concat_latent_image, concat_mask


def build_workspace(
    image,
    mask,
    *,
    edit_scope,
    mask_grow,
    mask_blur,
    mask_threshold,
    invert_mask,
    crop_context,
    crop_context_factor,
    min_size,
    max_size,
    resize_mode,
    expand_left,
    expand_right,
    expand_top,
    expand_bottom,
    canvas_fill,
    auto_recommend,
):
    original_width, original_height = get_image_size(image)
    original_size = (original_width, original_height)
    processed_mask = preprocess_mask(
        mask,
        original_size,
        invert_mask=bool(invert_mask),
        mask_threshold=float(mask_threshold),
        mask_grow=int(mask_grow),
        mask_blur=float(mask_blur),
    )
    mask_bbox, mask_area_ratio = get_mask_metrics(processed_mask, original_size)
    canvas_expand = (
        max(0, int(expand_left)),
        max(0, int(expand_right)),
        max(0, int(expand_top)),
        max(0, int(expand_bottom)),
    )
    scope = resolve_repair_scope(
        edit_scope,
        mask_area_ratio=mask_area_ratio,
        mask_bbox=mask_bbox,
        image_size=original_size,
        canvas_expand=canvas_expand,
    )

    if scope == "region":
        return {
            "edit_scope": "region",
            "work_image": image,
            "work_mask": resize_mask_to(processed_mask, original_width, original_height),
            "original_size": original_size,
            "work_size": original_size,
            "crop_box": None,
            "crop_scale": 1.0,
            "canvas_expand": list(canvas_expand),
            "original_box_in_canvas": None,
            "mask_bbox": list(mask_bbox) if mask_bbox is not None else None,
            "mask_area_ratio": mask_area_ratio,
            "recommended_denoise": recommend_denoise("replace", "region", "native_fill", auto_recommend),
            "canvas_fill": canvas_fill,
        }

    if scope == "crop":
        if mask_bbox is None or mask_area_ratio <= 0.0:
            raise RuntimeError("[LLS] crop edit scope requires a non-empty mask.")
        crop_box = compute_crop_box(mask_bbox, original_size, int(crop_context), float(crop_context_factor))
        crop_width = max(1, crop_box[2] - crop_box[0])
        crop_height = max(1, crop_box[3] - crop_box[1])
        work_width, work_height, crop_scale = resolve_work_size(
            (crop_width, crop_height),
            int(min_size),
            int(max_size),
            resize_mode,
        )
        work_image = resize_image_to(crop_image_to(image, crop_box), work_width, work_height)
        work_mask = resize_mask_to(crop_mask_to(processed_mask, crop_box), work_width, work_height)
        return {
            "edit_scope": "crop",
            "work_image": work_image,
            "work_mask": work_mask,
            "original_size": original_size,
            "work_size": (work_width, work_height),
            "crop_box": list(crop_box),
            "crop_scale": crop_scale,
            "canvas_expand": list(canvas_expand),
            "original_box_in_canvas": None,
            "mask_bbox": None,
            "mask_area_ratio": mask_area_ratio,
            "recommended_denoise": recommend_denoise("replace", "crop", "native_fill", auto_recommend),
            "canvas_fill": canvas_fill,
        }

    canvas_info = build_canvas_info(
        original_size,
        int(expand_left),
        int(expand_right),
        int(expand_top),
        int(expand_bottom),
    )
    work_width, work_height = canvas_info["work_size"]
    original_box = canvas_info["original_box"]
    user_mask = expand_canvas_mask(processed_mask, work_width, work_height, original_box=original_box)
    canvas_mask = build_canvas_repair_mask(user_mask, work_width, work_height, original_box=original_box)
    work_mask = merge_masks(user_mask, canvas_mask, (work_width, work_height))
    return {
        "edit_scope": "canvas",
        "work_image": expand_canvas_image(
            image,
            work_width,
            work_height,
            fill_mode=canvas_fill,
            original_box=original_box,
        ),
        "work_mask": work_mask,
        "original_size": original_size,
        "work_size": (work_width, work_height),
        "crop_box": None,
        "crop_scale": 1.0,
        "canvas_expand": list(canvas_expand),
        "original_box_in_canvas": list(original_box),
        "mask_bbox": None,
        "mask_area_ratio": mask_area_ratio,
        "recommended_denoise": recommend_denoise("fill", "canvas", "native_fill", auto_recommend),
        "canvas_fill": canvas_fill,
    }


def build_edit_info(workspace: dict[str, Any], routing, *, backend_hints: dict[str, Any] | None = None) -> dict[str, Any]:
    info = {
        "backend_name": routing.backend_name,
        "routing_reason": routing.routing_reason,
        "model_family": routing.capabilities["model_family"],
        "model_role": routing.capabilities["model_role"],
        "supports_inpaint_native": routing.capabilities["supports_inpaint_native"],
        "supports_image_edit_native": routing.capabilities["supports_image_edit_native"],
        "preferred_edit_backend": routing.capabilities["preferred_edit_backend"],
        "edit_scope": workspace["edit_scope"],
        "original_size": list(workspace["original_size"]),
        "work_size": list(workspace["work_size"]),
        "crop_box": workspace["crop_box"],
        "crop_scale": workspace["crop_scale"],
        "canvas_expand": workspace["canvas_expand"],
        "original_box_in_canvas": workspace["original_box_in_canvas"],
        "recommended_denoise": float(workspace["recommended_denoise"]),
        "edit_payload_version": "1.0",
    }
    if backend_hints:
        info.update(backend_hints)
    return info
```

```python
# pro_edit/backends/sdxl.py
from .registry import register_backend
from ..pro_edit_utils import build_native_conditioning_payload, set_conditioning_values


class SDXLProEditBackend:
    backend_name = "sdxl"

    def supports(self, capabilities):
        family = str(capabilities.get("model_family") or "")
        role = str(capabilities.get("model_role") or "unknown")
        return family.startswith("SDXL") and (
            role in {"inpaint", "edit", "fill"}
            or bool(capabilities.get("supports_inpaint_native"))
            or str(capabilities.get("preferred_edit_backend") or "") == "sdxl"
        )

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, routing, **_kwargs):
        latent, concat_latent_image, concat_mask = build_native_conditioning_payload(
            vae,
            work_image,
            work_mask,
            latent_source=f"pro_edit_prepare_{workspace['edit_scope']}",
            masked_fill_value=0.5,
        )
        values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask,
            "edit_backend": "sdxl",
        }
        return {
            "latent": latent,
            "positive": set_conditioning_values(positive, values),
            "negative": set_conditioning_values(negative, values),
            "backend_hints": {
                "backend_name": "sdxl",
                "routing_reason": routing.routing_reason,
            },
        }

    def prepare_bridge(self, **kwargs):
        return kwargs


register_backend(SDXLProEditBackend())
```

```python
# pro_edit/pro_edit_prepare.py
from .backends.registry import resolve_backend
from .pro_edit_utils import (
    AUTO_RECOMMEND_CHOICES,
    BACKEND_MODE_CHOICES,
    CANVAS_FILL_CHOICES,
    EDIT_SCOPE_CHOICES,
    build_edit_info,
    build_workspace,
)


class LLSProImageEditPrepare:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_EDIT_INFO", "FLOAT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "edit_info", "recommended_denoise", "positive", "negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "edit_scope": (EDIT_SCOPE_CHOICES, {"default": "auto"}),
                "mask_grow": ("INT", {"default": 24, "min": 0, "max": 2048}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("INT", {"default": 64, "min": 0, "max": 512}),
                "crop_context_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "min_size": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (["fit", "pad", "stretch"], {"default": "fit"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "canvas_fill": (CANVAS_FILL_CHOICES, {"default": "edge"}),
                "auto_recommend": (AUTO_RECOMMEND_CHOICES, {"default": "enabled"}),
            },
            "optional": {
                "model": ("MODEL",),
                "model_info": ("STRING",),
            },
        }

    def prepare(
        self,
        image,
        mask,
        vae,
        positive,
        negative,
        backend_mode,
        edit_scope,
        mask_grow,
        mask_blur,
        mask_threshold,
        invert_mask,
        crop_context,
        crop_context_factor,
        min_size,
        max_size,
        resize_mode,
        expand_left,
        expand_right,
        expand_top,
        expand_bottom,
        canvas_fill,
        auto_recommend,
        model=None,
        model_info=None,
    ):
        workspace = build_workspace(
            image,
            mask,
            edit_scope=edit_scope,
            mask_grow=mask_grow,
            mask_blur=mask_blur,
            mask_threshold=mask_threshold,
            invert_mask=invert_mask,
            crop_context=crop_context,
            crop_context_factor=crop_context_factor,
            min_size=min_size,
            max_size=max_size,
            resize_mode=resize_mode,
            expand_left=expand_left,
            expand_right=expand_right,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            canvas_fill=canvas_fill,
            auto_recommend=auto_recommend,
        )
        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=model_info,
        )
        prepared = backend.prepare(
            model=model,
            vae=vae,
            work_image=workspace["work_image"],
            work_mask=workspace["work_mask"],
            positive=positive,
            negative=negative,
            workspace=workspace,
            routing=routing,
        )
        edit_info = build_edit_info(
            workspace,
            routing,
            backend_hints=prepared.get("backend_hints"),
        )
        return (
            prepared["latent"],
            workspace["work_image"],
            workspace["work_mask"],
            edit_info,
            float(workspace["recommended_denoise"]),
            prepared["positive"],
            prepared["negative"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_sdxl.py' -v`

Expected: PASS with two tests.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/pro_edit_utils.py pro_edit/pro_edit_prepare.py pro_edit/backends/sdxl.py tests/test_pro_edit_prepare_sdxl.py
git commit -m "feat: implement sdxl pro edit prepare"
```

### Task 5: Implement FLUX Prepare Path and Edit-Capable Routing

**Files:**
- Create: `tests/test_pro_edit_prepare_flux.py`
- Modify: `pro_edit/backends/flux.py`
- Modify: `pro_edit/pro_edit_prepare.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_prepare_flux.py
import unittest

try:
    from .test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )
except ImportError:
    from test_pro_edit_helpers import (
        FakeMask,
        FakeModel,
        FakeTensor,
        FakeVAE,
        load_plugin_package,
        make_conditioning,
    )


class TestProEditPrepareFlux(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]
        self.node = node_cls()

    def test_prepare_region_routes_to_flux_backend(self):
        image = FakeTensor((1, 1024, 1024, 3), label="image")
        mask = FakeMask((1, 1024, 1024), mask_bbox=(300, 300, 724, 724), mask_area_ratio=0.18)
        vae = FakeVAE(latent_channels=16, downscale_ratio=16)
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="region",
            mask_grow=8,
            mask_blur=4.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.5,
            min_size=512,
            max_size=1024,
            resize_mode="fit",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["backend_name"], "flux")
        self.assertEqual(edit_info["model_family"], "FLUX_DEV")
        self.assertTrue(edit_info["supports_image_edit_native"])
        self.assertIn("concat_latent_image", positive[0][1])
        self.assertIn("concat_mask", positive[0][1])
        self.assertEqual(positive[0][1]["edit_backend"], "flux")
        self.assertEqual(latent["source"], "pro_edit_prepare_region")
        self.assertGreater(recommended, 0.0)

    def test_prepare_canvas_preserves_canvas_geometry_for_flux(self):
        image = FakeTensor((1, 768, 768, 3), label="image")
        mask = FakeMask((1, 768, 768), mask_bbox=(100, 100, 300, 300), mask_area_ratio=0.07)
        vae = FakeVAE(latent_channels=16, downscale_ratio=16)
        model = FakeModel(
            family="FLUX_DEV",
            model_role="fill",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
        )

        latent, work_image, work_mask, edit_info, recommended, positive, negative = self.node.prepare(
            image=image,
            mask=mask,
            vae=vae,
            positive=make_conditioning("positive"),
            negative=make_conditioning("negative"),
            backend_mode="auto",
            edit_scope="canvas",
            mask_grow=0,
            mask_blur=0.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.0,
            min_size=512,
            max_size=1024,
            resize_mode="fit",
            expand_left=64,
            expand_right=64,
            expand_top=32,
            expand_bottom=32,
            canvas_fill="neutral",
            auto_recommend="enabled",
            model=model,
            model_info=None,
        )

        self.assertEqual(edit_info["edit_scope"], "canvas")
        self.assertEqual(edit_info["original_box_in_canvas"], [64, 32, 832, 800])
        self.assertEqual(edit_info["work_size"], [896, 832])
        self.assertEqual(positive[0][1]["edit_backend"], "flux")
        self.assertGreater(recommended, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_flux.py' -v`

Expected: FAIL because the FLUX backend is still a stub.

- [ ] **Step 3: Implement FLUX backend preparation**

```python
# pro_edit/backends/flux.py
from .registry import register_backend
from ..pro_edit_utils import build_native_conditioning_payload, set_conditioning_values


class FluxProEditBackend:
    backend_name = "flux"

    def supports(self, capabilities):
        family = str(capabilities.get("model_family") or "")
        role = str(capabilities.get("model_role") or "unknown")
        return family.startswith("FLUX") and (
            role in {"inpaint", "edit", "fill"}
            or bool(capabilities.get("supports_image_edit_native"))
            or str(capabilities.get("preferred_edit_backend") or "") == "flux"
        )

    def prepare(self, *, vae, work_image, work_mask, positive, negative, workspace, routing, **_kwargs):
        latent, concat_latent_image, concat_mask = build_native_conditioning_payload(
            vae,
            work_image,
            work_mask,
            latent_source=f"pro_edit_prepare_{workspace['edit_scope']}",
            masked_fill_value=0.0,
        )
        values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask,
            "edit_backend": "flux",
        }
        return {
            "latent": latent,
            "positive": set_conditioning_values(positive, values),
            "negative": set_conditioning_values(negative, values),
            "backend_hints": {
                "backend_name": "flux",
                "routing_reason": routing.routing_reason,
            },
        }

    def prepare_bridge(self, *, positive, negative, flux_guidance=None, **kwargs):
        return {
            "positive": positive,
            "negative": negative,
            "flux_guidance": flux_guidance,
            **kwargs,
        }


register_backend(FluxProEditBackend())
```

```python
# pro_edit/pro_edit_prepare.py
backend, routing = resolve_backend(
    backend_mode,
    model=model,
    model_info=model_info,
)

prepared = backend.prepare(
    model=model,
    vae=vae,
    work_image=workspace["work_image"],
    work_mask=workspace["work_mask"],
    positive=positive,
    negative=negative,
    workspace=workspace,
    routing=routing,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_flux.py' -v`

Expected: PASS with two tests.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/backends/flux.py pro_edit/pro_edit_prepare.py tests/test_pro_edit_prepare_flux.py
git commit -m "feat: implement flux pro edit prepare"
```

### Task 6: Add the Professional KSampler Bridge

**Files:**
- Create: `tests/test_pro_edit_bridge.py`
- Modify: `pro_edit/pro_edit_bridge.py`
- Modify: `pro_edit/backends/sdxl.py`
- Modify: `pro_edit/backends/flux.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_bridge.py
import json
import unittest
from unittest import mock

try:
    from .test_pro_edit_helpers import FakeLatentTensor, FakeModel, import_plugin_submodule, load_plugin_package, make_conditioning
except ImportError:
    from test_pro_edit_helpers import FakeLatentTensor, FakeModel, import_plugin_submodule, load_plugin_package, make_conditioning


class TestProEditBridge(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.bridge_module = import_plugin_submodule(self.plugin, "pro_edit.pro_edit_bridge")
        self.node = self.bridge_module.LLSProKSamplerBridge()

    def test_bridge_uses_auto_from_edit_denoise(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "pro_edit_prepare_region"}

        with mock.patch.object(self.bridge_module, "_common_ksampler", side_effect=lambda **kwargs: kwargs["latent"]):
            result_latent, sample_info_json = self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.25,
                denoise_mode="auto_from_edit",
                flux_guidance=3.5,
                model_family="Auto",
                edit_info={
                    "backend_name": "sdxl",
                    "model_family": "SDXL",
                    "model_role": "inpaint",
                    "supports_inpaint_native": True,
                    "supports_image_edit_native": False,
                    "preferred_edit_backend": "sdxl",
                    "recommended_denoise": 0.63,
                },
                model_info=None,
            )

        sample_info = json.loads(sample_info_json)
        self.assertEqual(result_latent["source"], "pro_edit_prepare_region")
        self.assertEqual(sample_info["backend_name"], "sdxl")
        self.assertEqual(sample_info["denoise"], 0.63)
        self.assertEqual(sample_info["denoise_mode"], "auto_from_edit")

    def test_bridge_applies_flux_guidance_for_flux_backend(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
        )
        latent = {"samples": FakeLatentTensor((1, 16, 64, 64)), "source": "pro_edit_prepare_region"}

        with mock.patch.object(self.bridge_module, "_common_ksampler", side_effect=lambda **kwargs: kwargs["latent"]):
            _, sample_info_json = self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=9,
                steps=12,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.8,
                denoise_mode="manual",
                flux_guidance=4.2,
                model_family="Auto",
                edit_info={
                    "backend_name": "flux",
                    "model_family": "FLUX_DEV",
                    "model_role": "edit",
                    "supports_inpaint_native": False,
                    "supports_image_edit_native": True,
                    "preferred_edit_backend": "flux",
                    "recommended_denoise": 0.8,
                },
                model_info=None,
            )

        sample_info = json.loads(sample_info_json)
        self.assertEqual(sample_info["backend_name"], "flux")
        self.assertEqual(sample_info["guidance"], 4.2)

    def test_bridge_manual_backend_mismatch_raises(self):
        model = FakeModel(
            family="SDXL",
            model_role="inpaint",
            supports_inpaint_native=True,
            supports_image_edit_native=False,
            preferred_edit_backend="sdxl",
        )
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "pro_edit_prepare_region"}

        with self.assertRaisesRegex(RuntimeError, "backend 'flux' is incompatible"):
            self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="flux",
                quality_preset="Manual",
                seed=11,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                denoise_mode="manual",
                flux_guidance=3.5,
                model_family="Auto",
                edit_info=None,
                model_info=None,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_bridge.py' -v`

Expected: FAIL because the bridge node still raises `not implemented`.

- [ ] **Step 3: Implement the bridge sampler node**

```python
# pro_edit/pro_edit_bridge.py
from __future__ import annotations

import random

from .backends.registry import resolve_backend
from .pro_edit_utils import BACKEND_MODE_CHOICES, set_conditioning_values
from ..sampling.nodes import (
    _QUALITY_PRESETS,
    _common_ksampler,
    _get_samplers,
    _get_schedulers,
    _normalize_flux_guidance,
)
from ..utils.model_info import (
    FAMILY_DEFAULT_PRESET,
    MODEL_FAMILY_CHOICES,
    get_family_defaults,
    get_sampling_preset,
    info_to_json,
)


class LLSProKSamplerBridge:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sample_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "backend_mode": (BACKEND_MODE_CHOICES, {"default": "auto"}),
                "quality_preset": (_QUALITY_PRESETS, {"default": FAMILY_DEFAULT_PRESET}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_samplers(), {"default": "euler"}),
                "scheduler": (_get_schedulers(), {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (["manual", "auto_from_edit"], {"default": "manual"}),
                "flux_guidance": ("STRING,FLOAT,INT", {"default": 3.5, "widgetType": "FLOAT"}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "edit_info": ("LLS_EDIT_INFO",),
                "model_info": ("STRING",),
            },
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        backend_mode,
        quality_preset,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        denoise_mode,
        flux_guidance,
        model_family,
        edit_info=None,
        model_info=None,
    ):
        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=model_info,
            edit_info=edit_info,
        )
        defaults = get_family_defaults(routing.capabilities["model_family"])
        if quality_preset == FAMILY_DEFAULT_PRESET:
            steps = int(defaults["default_steps"])
            cfg = float(defaults["default_cfg"])
            sampler_name = str(defaults["default_sampler"])
            scheduler = str(defaults["default_scheduler"])
            denoise = float(defaults["default_denoise"])
        else:
            preset = get_sampling_preset(defaults, quality_preset)
            if preset is not None:
                steps = int(preset["steps"])
                cfg = float(preset["cfg"])
                sampler_name = str(preset["sampler_name"])
                scheduler = str(preset["scheduler"])
                denoise = float(preset["denoise"])

        actual_denoise = float(denoise)
        if denoise_mode == "auto_from_edit" and isinstance(edit_info, dict):
            actual_denoise = float(edit_info.get("recommended_denoise", actual_denoise))

        guidance_value = None
        if routing.backend_name == "flux":
            guidance_value = _normalize_flux_guidance(flux_guidance, defaults.get("default_guidance"))
            positive = set_conditioning_values(positive, {"guidance": guidance_value})
            negative = set_conditioning_values(negative, {"guidance": guidance_value})

        actual_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
        result_latent = _common_ksampler(
            model=model,
            seed=actual_seed,
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=actual_denoise,
        )
        sample_info = info_to_json(
            {
                "backend_name": routing.backend_name,
                "routing_reason": routing.routing_reason,
                "family": routing.capabilities["model_family"],
                "model_role": routing.capabilities["model_role"],
                "seed": actual_seed,
                "steps": int(steps),
                "cfg": float(cfg),
                "guidance": guidance_value,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": actual_denoise,
                "denoise_mode": denoise_mode,
                "quality_preset": quality_preset,
            }
        )
        return result_latent, sample_info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_bridge.py' -v`

Expected: PASS with three tests.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/pro_edit_bridge.py tests/test_pro_edit_bridge.py
git commit -m "feat: add pro edit sampler bridge"
```

### Task 7: Implement Real Finish Compositing and Preview Modes

**Files:**
- Create: `tests/test_pro_edit_finish.py`
- Modify: `pro_edit/pro_edit_utils.py`
- Modify: `pro_edit/pro_edit_finish.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_finish.py
import unittest

try:
    from .test_pro_edit_helpers import load_plugin_package, make_torch_image, make_torch_mask, torch
except ImportError:
    from test_pro_edit_helpers import load_plugin_package, make_torch_image, make_torch_mask, torch


class TestProEditFinish(unittest.TestCase):
    def setUp(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditFinish"]
        self.node = node_cls()

    def test_region_composite_only_replaces_masked_pixels(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(4, 4, 1.0)
        mask = make_torch_mask(4, 4, (1, 1, 3, 3))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "sdxl",
                "edit_scope": "region",
                "original_size": [4, 4],
                "work_size": [4, 4],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="final",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(float(final_image[0, 0, 0, 0].item()), 0.0)
        self.assertEqual(float(final_image[0, 1, 1, 0].item()), 1.0)
        self.assertEqual(tuple(preview_image.shape), tuple(final_image.shape))

    def test_crop_composite_pastes_generated_crop_back_into_original(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(2, 2, 0.75)
        mask = make_torch_mask(2, 2, (0, 0, 2, 2))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "sdxl",
                "edit_scope": "crop",
                "original_size": [4, 4],
                "work_size": [2, 2],
                "crop_box": [1, 1, 3, 3],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="mask",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(float(final_image[0, 0, 0, 0].item()), 0.0)
        self.assertEqual(float(final_image[0, 1, 1, 0].item()), 0.75)
        self.assertEqual(tuple(preview_image.shape), tuple(original.shape))

    def test_canvas_output_keeps_expanded_canvas_size(self):
        if torch is None:
            self.skipTest("torch is required for finish compositing tests")

        original = make_torch_image(4, 4, 0.0)
        generated = make_torch_image(6, 6, 0.5)
        mask = make_torch_mask(6, 6, (0, 0, 6, 6))

        final_image, preview_image = self.node.finish(
            original_image=original,
            generated_image=generated,
            edit_info={
                "backend_name": "flux",
                "edit_scope": "canvas",
                "original_size": [4, 4],
                "work_size": [6, 6],
                "original_box_in_canvas": [1, 1, 5, 5],
            },
            feather=0.0,
            color_match="disabled",
            brightness_match="disabled",
            blend_strength=1.0,
            restore_unmasked_area=True,
            edge_fix="none",
            preview_mode="compare",
            work_mask=mask,
            sample_info=None,
        )

        self.assertEqual(tuple(final_image.shape[1:3]), (6, 6))
        self.assertEqual(tuple(preview_image.shape[1:3]), (6, 12))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_finish.py' -v`

Expected: FAIL because the finish node still raises `not implemented`.

- [ ] **Step 3: Implement compositing and preview helpers**

```python
# pro_edit/pro_edit_utils.py
def _resize_mask_for_image(mask, image):
    width, height = get_image_size(image)
    return resize_mask_to(mask, width, height)


def _build_canvas_base(original_image, canvas_image, original_box):
    x1, y1, x2, y2 = original_box
    out = torch.zeros_like(canvas_image)
    out[:, y1:y2, x1:x2, :] = resize_image_to(original_image, x2 - x1, y2 - y1)
    return out


def project_work_mask(work_mask, edit_info, reference_image):
    scope = str(edit_info.get("edit_scope") or "region")
    if scope == "region":
        return _resize_mask_for_image(work_mask, reference_image)

    if scope == "crop":
        crop_box = edit_info.get("crop_box")
        if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
            raise RuntimeError("[LLS] crop edit_info must include crop_box.")
        x1, y1, x2, y2 = [int(value) for value in crop_box]
        projected = torch.zeros(
            (work_mask.shape[0], reference_image.shape[1], reference_image.shape[2]),
            dtype=work_mask.dtype,
            device=work_mask.device,
        )
        projected[:, y1:y2, x1:x2] = resize_mask_to(work_mask, x2 - x1, y2 - y1)
        return projected

    return _resize_mask_for_image(work_mask, reference_image)


def _blend_torch_images(original, generated, mask, blend_strength: float):
    mask_4d = _resize_mask_for_image(mask, original).unsqueeze(-1).clamp(0.0, 1.0) * float(blend_strength)
    return (original * (1.0 - mask_4d)) + (generated * mask_4d)


def overlay_mask_preview(image, mask, *, alpha: float = 0.4):
    if torch is None or not isinstance(image, torch.Tensor):
        return image
    mask_4d = _resize_mask_for_image(mask, image).unsqueeze(-1).clamp(0.0, 1.0) * float(alpha)
    red = torch.zeros_like(image)
    red[..., 0] = 1.0
    return (image * (1.0 - mask_4d)) + (red * mask_4d)


def _paste_torch_patch(base, patch, box):
    x1, y1, x2, y2 = box
    out = base.clone()
    out[:, y1:y2, x1:x2, :] = patch
    return out


def compose_region_result(original_image, generated_image, work_mask, edit_info, blend_strength):
    del edit_info
    original_width, original_height = get_image_size(original_image)
    base_image = resize_image_to(original_image, original_width, original_height)
    edited_image = resize_image_to(generated_image, original_width, original_height)
    return _blend_torch_images(base_image, edited_image, _resize_mask_for_image(work_mask, base_image), blend_strength)


def compose_crop_result(original_image, generated_image, work_mask, edit_info, blend_strength):
    crop_box = edit_info.get("crop_box")
    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
        raise RuntimeError("[LLS] crop edit_info must include crop_box.")
    crop_width = max(1, int(crop_box[2]) - int(crop_box[0]))
    crop_height = max(1, int(crop_box[3]) - int(crop_box[1]))
    resized_patch = resize_image_to(generated_image, crop_width, crop_height)
    resized_mask = resize_mask_to(work_mask, crop_width, crop_height)
    original_patch = crop_image_to(original_image, tuple(int(v) for v in crop_box))
    blended_patch = _blend_torch_images(original_patch, resized_patch, resized_mask, blend_strength)
    return _paste_torch_patch(original_image, blended_patch, tuple(int(v) for v in crop_box))


def compose_canvas_result(original_image, generated_image, work_mask, edit_info, blend_strength):
    original_box = edit_info.get("original_box_in_canvas")
    if not isinstance(original_box, (list, tuple)) or len(original_box) != 4:
        raise RuntimeError("[LLS] canvas edit_info must include original_box_in_canvas.")
    base_canvas = _build_canvas_base(
        original_image,
        generated_image,
        tuple(int(value) for value in original_box),
    )
    return _blend_torch_images(base_canvas, generated_image, _resize_mask_for_image(work_mask, generated_image), blend_strength)


def build_preview_image(original_image, final_image, work_mask, edit_info, preview_mode):
    scope = str(edit_info.get("edit_scope") or "region")
    if preview_mode == "final":
        return final_image
    if preview_mode == "mask":
        if scope == "canvas":
            preview_base = _build_canvas_base(
                original_image,
                final_image,
                tuple(int(value) for value in edit_info["original_box_in_canvas"]),
            )
            projected_mask = _resize_mask_for_image(work_mask, final_image)
            return overlay_mask_preview(preview_base, projected_mask)
        projected_mask = project_work_mask(work_mask, edit_info, original_image)
        return overlay_mask_preview(original_image, projected_mask)
    if scope == "canvas":
        compare_left = _build_canvas_base(
            original_image,
            final_image,
            tuple(int(value) for value in edit_info["original_box_in_canvas"]),
        )
    else:
        compare_left = resize_image_to(original_image, get_image_size(final_image)[0], get_image_size(final_image)[1])
    return torch.cat([compare_left, final_image], dim=2)
```

```python
# pro_edit/pro_edit_finish.py
from .pro_edit_utils import (
    BRIGHTNESS_MATCH_CHOICES,
    COLOR_MATCH_CHOICES,
    EDGE_FIX_CHOICES,
    PREVIEW_MODE_CHOICES,
    build_preview_image,
    compose_canvas_result,
    compose_crop_result,
    compose_region_result,
    normalize_edit_info,
)


class LLSProImageEditFinish:
    CATEGORY = "LLS/Image Edit"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "edit_info": ("LLS_EDIT_INFO",),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "color_match": (COLOR_MATCH_CHOICES, {"default": "disabled"}),
                "brightness_match": (BRIGHTNESS_MATCH_CHOICES, {"default": "enabled"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (EDGE_FIX_CHOICES, {"default": "soft"}),
                "preview_mode": (PREVIEW_MODE_CHOICES, {"default": "final"}),
            },
            "optional": {
                "work_mask": ("MASK",),
                "sample_info": ("STRING",),
            },
        }

    def finish(
        self,
        original_image,
        generated_image,
        edit_info,
        feather,
        color_match,
        brightness_match,
        blend_strength,
        restore_unmasked_area,
        edge_fix,
        preview_mode,
        work_mask=None,
        sample_info=None,
    ):
        del feather
        del color_match
        del brightness_match
        del restore_unmasked_area
        del edge_fix
        del sample_info

        info = normalize_edit_info(edit_info)
        scope = info["edit_scope"]
        if work_mask is None:
            raise RuntimeError("[LLS] LLS Pro Image Edit Finish requires work_mask.")

        if scope == "region":
            final_image = compose_region_result(original_image, generated_image, work_mask, info, blend_strength)
        elif scope == "crop":
            final_image = compose_crop_result(original_image, generated_image, work_mask, info, blend_strength)
        elif scope == "canvas":
            final_image = compose_canvas_result(original_image, generated_image, work_mask, info, blend_strength)
        else:
            raise RuntimeError(f"[LLS] Unsupported edit_scope '{scope}'.")

        preview_image = build_preview_image(original_image, final_image, work_mask, info, preview_mode)
        return final_image, preview_image
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_finish.py' -v`

Expected: PASS with three tests.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/pro_edit_utils.py pro_edit/pro_edit_finish.py tests/test_pro_edit_finish.py
git commit -m "feat: implement pro edit finish compositing"
```

### Task 8: Document the Professional Workflow and Run Full Verification

**Files:**
- Create: `tests/test_pro_edit_docs.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing documentation test**

```python
# tests/test_pro_edit_docs.py
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
            "Load Image -> Load Mask or LLS Simple Mask Draw -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Pro Image Edit Prepare -> LLS Pro KSampler Bridge -> VAE Decode -> LLS Pro Image Edit Finish -> Preview Image",
            "Adding new professional edit models",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_docs.py' -v`

Expected: FAIL because the README does not yet describe the professional edit chain.

- [ ] **Step 3: Update the README**

```markdown
## Pro Image Edit / Inpaint

`LLS-node` also provides a stricter professional local editing chain for models that support native edit or inpaint semantics:

- `LLS Pro Image Edit Prepare`
- `LLS Pro KSampler Bridge`
- `LLS Pro Image Edit Finish`

### Simple vs Pro

- `Simple = lightweight masked latent resampling`
- `Pro = true image edit / inpaint pipeline`

Use the `Pro` chain when the model really supports native local edit or inpaint behavior and prompt-following on masked regions matters more than lenient fallback behavior.

### Professional Workflow

`Load Image -> Load Mask or LLS Simple Mask Draw -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Pro Image Edit Prepare -> LLS Pro KSampler Bridge -> VAE Decode -> LLS Pro Image Edit Finish -> Preview Image`

### Backend Selection

- `backend_mode = auto | sdxl | flux`
- `auto` uses capability metadata and fails explicitly when no professional backend matches
- `sdxl` and `flux` force that backend but still validate the current model

### Capability Requirements

- `model_role`
- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

These fields can come from:

- `model_info`
- `_lls_*` metadata written by `LLS Simple Checkpoint Loader`
- family and name inference in `utils/model_info.py`

### Adding New Professional Edit Models

1. Add or adjust role and capability inference in `utils/model_info.py`
2. Verify `LLS Simple Checkpoint Loader` writes the correct `_lls_*` capability tags
3. Use `backend_mode=auto` after the capability metadata is recognized
4. Use `backend_mode=sdxl` or `backend_mode=flux` while debugging explicit routing
```

- [ ] **Step 4: Run the focused docs test**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit_docs.py' -v`

Expected: PASS with one test.

- [ ] **Step 5: Run the full verification suite**

Run: `python3 -m unittest discover -s tests -p 'test_pro_edit*.py' -v`

Expected: PASS for all new professional edit tests.

Run: `python3 -m unittest discover -s tests -p 'test_model_info_inference.py' -v`

Expected: PASS and confirm capability helpers did not break existing model info inference.

Run: `python3 -m unittest discover -s tests -p 'test_repair*.py' -v`

Expected: PASS and confirm the existing `LLSSimple*` repair chain remains unchanged.

Run: `python3 -m compileall __init__.py pro_edit utils model_loader`

Expected: PASS with no syntax errors.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_pro_edit_docs.py
git commit -m "docs: add pro image edit workflow"
```

## Self-Review Checklist

- Spec coverage:
  - New parallel node chain is covered by Tasks 1, 4, 6, and 7.
  - Capability metadata and auto or manual routing are covered by Tasks 2 and 3.
  - SDXL and FLUX backend-specific prepare behavior is covered by Tasks 4 and 5.
  - Real finish compositing and preview behavior are covered by Task 7.
  - README updates and model-onboarding guidance are covered by Task 8.
  - Existing `LLSSimple*` chain remains untouched and regression-tested in Task 8.
- Placeholder scan:
  - No `TBD`, `TODO`, or deferred “implement later” steps remain.
  - Every task includes explicit files, commands, expected outcomes, and concrete code snippets.
- Type consistency:
  - Node keys: `LLSProImageEditPrepare`, `LLSProKSamplerBridge`, `LLSProImageEditFinish`
  - Payload type: `LLS_EDIT_INFO`
  - Routing choices: `auto | sdxl | flux`
  - Denoise choices: `manual | auto_from_edit`
  - Scope choices: `auto | region | crop | canvas`
