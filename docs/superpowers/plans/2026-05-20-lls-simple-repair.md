# LLS Simple Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-oriented local repair workflow to `LLS-node` with two new repair nodes and repair-aware `LLS Simple KSampler` compatibility, while keeping existing txt2img and img2img workflows unchanged when repair inputs are absent.

**Architecture:** Add a dedicated `repair/` feature package with focused modules for prepare, finish, and shared repair utilities. Keep `LLS Simple KSampler` in `sampling/nodes.py` and add only a compatibility layer that reads `repair_info`, selects effective denoise and adapter behavior, and preserves old behavior when repair metadata is missing.

**Tech Stack:** Python, unittest, ComfyUI node definitions, existing JSON/model-info helpers, fake tensor test stubs, `python3 -m compileall`

---

## File Structure

- Create: `repair/__init__.py`
  - Export repair node registration maps.
- Create: `repair/repair_utils.py`
  - Normalize `repair_info`, choose scope/kernel/denoise, clamp crop/canvas geometry, and provide reusable image/mask helper shims.
- Create: `repair/repair_prepare.py`
  - Define `LLSSimpleRepairPrepare`.
- Create: `repair/repair_finish.py`
  - Define `LLSSimpleRepairFinish`.
- Create: `tests/test_repair_helpers.py`
  - Shared plugin loader plus fake image/mask/vae/latent stubs for repair tests.
- Create: `tests/test_repair_registration.py`
  - Verify plugin registration and node schemas.
- Create: `tests/test_repair_utils.py`
  - Pure-Python tests for scope/kernel/denoise/geometry and repair-info normalization.
- Create: `tests/test_repair_prepare.py`
  - Verify prepare-node orchestration for `region`, `crop`, and `canvas`.
- Create: `tests/test_repair_finish.py`
  - Verify finish-node dispatch and preview behavior.
- Create: `tests/test_repair_sampler.py`
  - Verify sampler compatibility with and without `repair_info`.
- Modify: `__init__.py`
  - Load the new `repair` package.
- Modify: `sampling/nodes.py`
  - Add repair-aware optional inputs and runtime logic.
- Modify: `README.md`
  - Document the new workflow and compatibility guarantees.

### Task 1: Add Repair Package Scaffolding and Registration Tests

**Files:**
- Create: `tests/test_repair_helpers.py`
- Create: `tests/test_repair_registration.py`
- Create: `repair/__init__.py`
- Create: `repair/repair_prepare.py`
- Create: `repair/repair_finish.py`
- Modify: `__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair_helpers.py
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_node_test_repair",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test_repair"] = module
    spec.loader.exec_module(module)
    return module
```

```python
# tests/test_repair_registration.py
import unittest

from test_repair_helpers import load_plugin_package


class TestRepairRegistration(unittest.TestCase):
    def test_plugin_registers_repair_nodes(self):
        plugin = load_plugin_package()

        self.assertIn("LLSSimpleRepairPrepare", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSSimpleRepairFinish", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleRepairPrepare"],
            "LLS Simple Repair Prepare",
        )
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSSimpleRepairFinish"],
            "LLS Simple Repair Finish",
        )

    def test_prepare_node_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairPrepare"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "prepare")
        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT"),
        )
        self.assertEqual(schema["required"]["image"][0], "IMAGE")
        self.assertEqual(schema["required"]["mask"][0], "MASK")
        self.assertEqual(schema["required"]["vae"][0], "VAE")
        self.assertEqual(schema["optional"]["model_info"][0], "STRING")
        self.assertEqual(schema["optional"]["positive"][0], "CONDITIONING")
        self.assertEqual(schema["optional"]["negative"][0], "CONDITIONING")

    def test_finish_node_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleRepairFinish"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(node_cls.CATEGORY, "LLS/Image Repair")
        self.assertEqual(node_cls.FUNCTION, "finish")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(schema["required"]["original_image"][0], "IMAGE")
        self.assertEqual(schema["required"]["generated_image"][0], "IMAGE")
        self.assertEqual(schema["required"]["repair_info"][0], "LLS_REPAIR_INFO")
        self.assertEqual(schema["optional"]["work_mask"][0], "MASK")
        self.assertEqual(schema["optional"]["sample_info"][0], "STRING")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_repair_registration.py' -v`
Expected: FAIL because the `repair` package and node registrations do not exist yet.

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
# repair/__init__.py
from .repair_finish import NODE_CLASS_MAPPINGS as FINISH_CLASS_MAPPINGS
from .repair_finish import NODE_DISPLAY_NAME_MAPPINGS as FINISH_DISPLAY_NAME_MAPPINGS
from .repair_prepare import NODE_CLASS_MAPPINGS as PREPARE_CLASS_MAPPINGS
from .repair_prepare import NODE_DISPLAY_NAME_MAPPINGS as PREPARE_DISPLAY_NAME_MAPPINGS


NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(PREPARE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FINISH_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(PREPARE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FINISH_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

```python
# repair/repair_prepare.py
class LLSSimpleRepairPrepare:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "repair_scope": (["auto", "region", "crop", "canvas"], {"default": "auto"}),
                "repair_kernel": (["auto", "latent_mask", "vae_inpaint", "native_fill"], {"default": "auto"}),
                "task_hint": (
                    ["auto", "repair", "remove", "replace", "fill", "appearance", "content", "structure", "dehaze", "deshadow", "recolor"],
                    {"default": "auto"},
                ),
                "mask_grow": ("INT", {"default": 8, "min": 0, "max": 128}),
                "mask_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_context": ("INT", {"default": 64, "min": 0, "max": 512}),
                "crop_context_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "min_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "resize_mode": (["keep_aspect", "force_square", "ranged_size"], {"default": "keep_aspect"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "canvas_fill": (["edge", "blur", "black", "white", "neutral"], {"default": "edge"}),
                "auto_recommend": (["enabled", "disabled"], {"default": "enabled"}),
            },
            "optional": {
                "model_info": ("STRING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    def prepare(self, **_kwargs):
        raise RuntimeError("[LLS] LLS Simple Repair Prepare is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSSimpleRepairPrepare": LLSSimpleRepairPrepare}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleRepairPrepare": "LLS Simple Repair Prepare"}
```

```python
# repair/repair_finish.py
class LLSSimpleRepairFinish:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "repair_info": ("LLS_REPAIR_INFO",),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "color_match": (["disabled", "mean_std", "histogram_simple"], {"default": "mean_std"}),
                "brightness_match": (["disabled", "enabled"], {"default": "enabled"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "restore_unmasked_area": ("BOOLEAN", {"default": True}),
                "edge_fix": (["none", "soft", "strong"], {"default": "none"}),
                "preview_mode": (["final", "compare", "mask", "before_after"], {"default": "final"}),
            },
            "optional": {
                "work_mask": ("MASK",),
                "sample_info": ("STRING",),
            },
        }

    def finish(self, **_kwargs):
        raise RuntimeError("[LLS] LLS Simple Repair Finish is not implemented yet.")


NODE_CLASS_MAPPINGS = {"LLSSimpleRepairFinish": LLSSimpleRepairFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSSimpleRepairFinish": "LLS Simple Repair Finish"}
```

- [ ] **Step 4: Run the registration tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_repair_registration.py' -v`
Expected: PASS with the two new nodes registered and their initial schemas exposed.

- [ ] **Step 5: Commit the scaffolding**

```bash
git add __init__.py repair/__init__.py repair/repair_prepare.py repair/repair_finish.py tests/test_repair_helpers.py tests/test_repair_registration.py
git commit -m "feat: add repair node scaffolding"
```

### Task 2: Implement Pure-Python Repair Utilities

**Files:**
- Create: `repair/repair_utils.py`
- Create: `tests/test_repair_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair_utils.py
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPAIR_UTILS_PATH = ROOT / "repair" / "repair_utils.py"


def load_repair_utils():
    spec = importlib.util.spec_from_file_location("lls_repair_utils_test", REPAIR_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRepairUtils(unittest.TestCase):
    def test_auto_scope_prefers_canvas_then_crop_then_region(self):
        utils = load_repair_utils()

        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.0,
                mask_bbox=None,
                image_size=(1024, 1024),
                canvas_expand=(128, 0, 0, 0),
            ),
            "canvas",
        )
        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.08,
                mask_bbox=(100, 100, 260, 260),
                image_size=(1024, 1024),
                canvas_expand=(0, 0, 0, 0),
            ),
            "crop",
        )
        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.62,
                mask_bbox=(0, 0, 1024, 1024),
                image_size=(1024, 1024),
                canvas_expand=(0, 0, 0, 0),
            ),
            "region",
        )

    def test_auto_kernel_prefers_native_then_latent_mask_then_vae_inpaint(self):
        utils = load_repair_utils()

        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="region",
                task_hint="repair",
                mask_area_ratio=0.05,
                model_info={"model_role": "fill", "supports_inpaint_native": True},
            )[0],
            "native_fill",
        )
        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="region",
                task_hint="appearance",
                mask_area_ratio=0.05,
                model_info={"model_role": "normal", "supports_inpaint_native": False},
            )[0],
            "latent_mask",
        )
        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="canvas",
                task_hint="fill",
                mask_area_ratio=0.50,
                model_info={"model_role": "normal", "supports_inpaint_native": False},
            )[0],
            "vae_inpaint",
        )

    def test_recommended_denoise_applies_task_and_scope_rules(self):
        utils = load_repair_utils()

        self.assertEqual(utils.recommend_denoise("repair", "region", "latent_mask", "enabled"), 0.45)
        self.assertEqual(utils.recommend_denoise("fill", "canvas", "vae_inpaint", "enabled"), 0.90)
        self.assertEqual(utils.recommend_denoise("replace", "crop", "vae_inpaint", "enabled"), 0.65)
        self.assertEqual(utils.recommend_denoise("auto", "crop", "vae_inpaint", "disabled"), 0.50)

    def test_compute_crop_box_and_resize_target_are_clamped(self):
        utils = load_repair_utils()

        crop_box = utils.compute_crop_box(
            mask_bbox=(900, 900, 1100, 1100),
            image_size=(1024, 1024),
            crop_context=64,
            crop_context_factor=1.5,
        )
        self.assertEqual(crop_box, (736, 736, 1024, 1024))

        work_width, work_height, scale = utils.resolve_work_size(
            crop_size=(288, 288),
            min_size=512,
            max_size=1024,
            resize_mode="keep_aspect",
        )
        self.assertEqual((work_width, work_height), (512, 512))
        self.assertGreater(scale, 1.0)

    def test_canvas_info_tracks_new_size_and_original_box(self):
        utils = load_repair_utils()

        canvas_info = utils.build_canvas_info((640, 480), 64, 32, 16, 8)
        self.assertEqual(canvas_info["work_size"], (736, 504))
        self.assertEqual(canvas_info["original_box"], (64, 16, 704, 496))

    def test_normalize_repair_info_backfills_required_fields(self):
        utils = load_repair_utils()

        info = utils.normalize_repair_info({"repair_scope": "region", "repair_kernel": "latent_mask"})
        self.assertEqual(info["repair_scope"], "region")
        self.assertEqual(info["repair_kernel"], "latent_mask")
        self.assertEqual(info["model_family"], "UNKNOWN")
        self.assertEqual(info["model_role"], "unknown")
        self.assertEqual(info["repair_payload_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_repair_utils.py' -v`
Expected: FAIL because `repair/repair_utils.py` does not exist yet.

- [ ] **Step 3: Implement the utility module**

```python
# repair/repair_utils.py
from __future__ import annotations

from typing import Any

from ..utils.model_info import parse_jsonish_info


REPAIR_INFO_TYPE = "LLS_REPAIR_INFO"
GUIDANCE_STACK_TYPE = "LLS_GUIDANCE_STACK"
SUPPORTED_SAMPLER_FAMILIES = {"SD1.5", "SDXL", "SDXL_TURBO", "FLUX_DEV", "FLUX_SCHNELL"}


def normalize_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(model_info)
    family = str(raw.get("model_family") or raw.get("family") or "UNKNOWN")
    role = str(raw.get("model_role") or raw.get("role") or "unknown")
    supports_native = bool(raw.get("supports_inpaint_native", False))
    return {
        "model_family": family,
        "model_role": role,
        "supports_inpaint_native": supports_native,
    }


def normalize_repair_info(repair_info: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = parse_jsonish_info(repair_info)
    info = dict(raw)
    info.setdefault("repair_scope", "region")
    info.setdefault("repair_kernel", "vae_inpaint")
    info.setdefault("task_hint", "auto")
    info.setdefault("original_size", [0, 0])
    info.setdefault("work_size", [0, 0])
    info.setdefault("crop_box", None)
    info.setdefault("crop_scale", None)
    info.setdefault("canvas_expand", [0, 0, 0, 0])
    info.setdefault("mask_grow", 8)
    info.setdefault("mask_blur", 8.0)
    info.setdefault("mask_threshold", 0.5)
    info.setdefault("invert_mask", False)
    info.setdefault("recommended_denoise", 0.55)
    info.setdefault("model_family", "UNKNOWN")
    info.setdefault("model_role", "unknown")
    info.setdefault("repair_payload_version", "1.0")
    info.setdefault("warnings", [])
    return info


def resolve_repair_scope(
    requested_scope: str,
    *,
    mask_area_ratio: float,
    mask_bbox: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    canvas_expand: tuple[int, int, int, int],
) -> str:
    if requested_scope != "auto":
        return requested_scope
    if any(int(value) > 0 for value in canvas_expand):
        return "canvas"
    if mask_bbox is None:
        raise RuntimeError("[LLS] mask is empty after preprocessing.")
    image_width, image_height = image_size
    x1, y1, x2, y2 = mask_bbox
    bbox_width_ratio = max(0.0, x2 - x1) / float(max(1, image_width))
    bbox_height_ratio = max(0.0, y2 - y1) / float(max(1, image_height))
    if mask_area_ratio <= 0.18:
        return "crop"
    if mask_area_ratio <= 0.35 and bbox_width_ratio <= 0.8 and bbox_height_ratio <= 0.8:
        return "crop"
    return "region"


def resolve_repair_kernel(
    requested_kernel: str,
    *,
    scope: str,
    task_hint: str,
    mask_area_ratio: float,
    model_info: dict[str, Any] | str | None,
) -> tuple[str, list[str]]:
    normalized_model = normalize_model_info(model_info)
    warnings: list[str] = []
    if requested_kernel != "auto":
        if requested_kernel not in {"latent_mask", "vae_inpaint", "native_fill"}:
            raise RuntimeError(f"[LLS] Unsupported repair_kernel '{requested_kernel}'.")
        if requested_kernel == "native_fill" and not normalized_model["supports_inpaint_native"]:
            warnings.append("native_fill requested but unsupported; falling back to vae_inpaint")
            return "vae_inpaint", warnings
        return requested_kernel, warnings
    if normalized_model["model_role"] in {"inpaint", "fill", "edit"} and normalized_model["supports_inpaint_native"]:
        return "native_fill", warnings
    if scope != "canvas" and mask_area_ratio <= 0.20 and task_hint in {"repair", "appearance", "dehaze", "deshadow", "recolor"}:
        return "latent_mask", warnings
    return "vae_inpaint", warnings


def recommend_denoise(task_hint: str, scope: str, kernel: str, auto_recommend: str) -> float:
    if auto_recommend == "disabled":
        if scope == "canvas":
            return 0.90
        if scope == "crop":
            return 0.50
        if kernel == "latent_mask":
            return 0.45
        return 0.55
    base = {
        "repair": 0.45,
        "appearance": 0.50,
        "dehaze": 0.55,
        "deshadow": 0.55,
        "recolor": 0.55,
        "structure": 0.60,
        "content": 0.65,
        "replace": 0.72,
        "remove": 0.88,
        "fill": 0.90,
        "auto": 0.55,
    }.get(task_hint, 0.55)
    if scope == "canvas":
        return max(0.90, base)
    if scope == "crop":
        return max(0.30, min(0.65, base))
    return base


def clamp_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), image_width))
    y1 = max(0, min(int(y1), image_height))
    x2 = max(x1, min(int(x2), image_width))
    y2 = max(y1, min(int(y2), image_height))
    return x1, y1, x2, y2


def compute_crop_box(
    *,
    mask_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    crop_context: int,
    crop_context_factor: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = mask_bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    expanded_width = (box_width + crop_context * 2) * crop_context_factor
    expanded_height = (box_height + crop_context * 2) * crop_context_factor
    half_width = expanded_width / 2.0
    half_height = expanded_height / 2.0
    return clamp_box(
        (
            round(center_x - half_width),
            round(center_y - half_height),
            round(center_x + half_width),
            round(center_y + half_height),
        ),
        image_size,
    )


def resolve_work_size(
    *,
    crop_size: tuple[int, int],
    min_size: int,
    max_size: int,
    resize_mode: str,
) -> tuple[int, int, float]:
    crop_width, crop_height = crop_size
    longest = max(crop_width, crop_height)
    if resize_mode == "force_square":
        side = min(max_size, max(min_size, longest))
        scale = side / float(max(1, longest))
        return side, side, scale
    if min_size <= longest <= max_size:
        return crop_width, crop_height, 1.0
    target_longest = min(max_size, max(min_size, longest))
    scale = target_longest / float(max(1, longest))
    width = int(round(crop_width * scale))
    height = int(round(crop_height * scale))
    return width, height, scale


def build_canvas_info(
    image_size: tuple[int, int],
    expand_left: int,
    expand_right: int,
    expand_top: int,
    expand_bottom: int,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int]]:
    image_width, image_height = image_size
    work_width = image_width + expand_left + expand_right
    work_height = image_height + expand_top + expand_bottom
    return {
        "work_size": (work_width, work_height),
        "original_box": (
            expand_left,
            expand_top,
            expand_left + image_width,
            expand_top + image_height,
        ),
    }


def resolve_adapter_mode(adapter_mode: str, model_family: str) -> str:
    if adapter_mode != "auto":
        return adapter_mode
    if model_family in {"SD1.5", "SDXL", "SDXL_TURBO"}:
        return "sd_classic"
    if model_family in {"FLUX_DEV", "FLUX_SCHNELL", "FLUX"}:
        return "flux"
    if model_family == "SD3":
        return "sd3"
    if model_family == "QWEN":
        return "qwen"
    if model_family == "ZIMAGE":
        return "zimage"
    return "auto"
```

- [ ] **Step 4: Run the utility tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_repair_utils.py' -v`
Expected: PASS with deterministic scope/kernel/denoise/geometry behavior.

- [ ] **Step 5: Commit the utility layer**

```bash
git add repair/repair_utils.py tests/test_repair_utils.py
git commit -m "feat: add repair utility helpers"
```

### Task 3: Implement `LLS Simple Repair Prepare`

**Files:**
- Modify: `repair/repair_prepare.py`
- Modify: `repair/repair_utils.py`
- Modify: `tests/test_repair_helpers.py`
- Create: `tests/test_repair_prepare.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair_helpers.py
class FakeTensor:
    def __init__(self, shape, label="tensor"):
        self.shape = tuple(shape)
        self.label = label

    def resized(self, width, height):
        return FakeTensor((self.shape[0], height, width, self.shape[-1]), self.label)

    def canvas_expanded(self, width, height):
        return FakeTensor((self.shape[0], height, width, self.shape[-1]), self.label)


class FakeMask:
    def __init__(self, width, height, bbox=None, area_ratio=0.0, label="mask"):
        self.shape = (1, height, width)
        self.mask_bbox = bbox
        self.mask_area_ratio = area_ratio
        self.label = label

    def resized(self, width, height):
        return FakeMask(width, height, bbox=self.mask_bbox, area_ratio=self.mask_area_ratio, label=self.label)

    def canvas_expanded(self, width, height):
        return FakeMask(width, height, bbox=self.mask_bbox, area_ratio=self.mask_area_ratio, label=self.label)


class FakeLatentTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVAE:
    def encode(self, image):
        batch, height, width, _channels = image.shape
        return FakeLatentTensor((batch, 4, max(1, height // 8), max(1, width // 8)))
```

```python
# tests/test_repair_prepare.py
import json
import unittest

from test_repair_helpers import FakeMask, FakeTensor, FakeVAE, load_plugin_package


class TestRepairPrepare(unittest.TestCase):
    def test_region_latent_mask_builds_noise_mask_and_repair_info(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_prepare import LLSSimpleRepairPrepare

        node = LLSSimpleRepairPrepare()
        latent, work_image, work_mask, repair_info, recommended = node.prepare(
            image=FakeTensor((1, 1024, 1024, 3), "image"),
            mask=FakeMask(1024, 1024, bbox=(100, 100, 280, 280), area_ratio=0.04),
            vae=FakeVAE(),
            repair_scope="region",
            repair_kernel="latent_mask",
            task_hint="repair",
            mask_grow=8,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.2,
            min_size=512,
            max_size=1024,
            resize_mode="keep_aspect",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
        )

        self.assertEqual(work_image.shape, (1, 1024, 1024, 3))
        self.assertEqual(work_mask.shape, (1, 1024, 1024))
        self.assertIn("noise_mask", latent)
        self.assertEqual(repair_info["repair_scope"], "region")
        self.assertEqual(repair_info["repair_kernel"], "latent_mask")
        self.assertEqual(recommended, 0.45)

    def test_crop_uses_mask_bbox_and_records_crop_metadata(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_prepare import LLSSimpleRepairPrepare

        node = LLSSimpleRepairPrepare()
        latent, work_image, work_mask, repair_info, recommended = node.prepare(
            image=FakeTensor((1, 1024, 1024, 3), "image"),
            mask=FakeMask(1024, 1024, bbox=(100, 100, 220, 220), area_ratio=0.02),
            vae=FakeVAE(),
            repair_scope="crop",
            repair_kernel="vae_inpaint",
            task_hint="replace",
            mask_grow=8,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.2,
            min_size=512,
            max_size=1024,
            resize_mode="keep_aspect",
            expand_left=0,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
        )

        self.assertEqual(repair_info["repair_scope"], "crop")
        self.assertIsNotNone(repair_info["crop_box"])
        self.assertGreater(repair_info["crop_scale"], 0.0)
        self.assertEqual(list(repair_info["work_size"]), [work_image.shape[2], work_image.shape[1]])
        self.assertEqual(recommended, 0.65)
        self.assertEqual(latent["source"], "repair_prepare_crop")

    def test_canvas_uses_expansion_even_with_empty_user_mask(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_prepare import LLSSimpleRepairPrepare

        node = LLSSimpleRepairPrepare()
        latent, work_image, work_mask, repair_info, recommended = node.prepare(
            image=FakeTensor((1, 640, 640, 3), "image"),
            mask=FakeMask(640, 640, bbox=None, area_ratio=0.0),
            vae=FakeVAE(),
            repair_scope="canvas",
            repair_kernel="vae_inpaint",
            task_hint="fill",
            mask_grow=8,
            mask_blur=8.0,
            mask_threshold=0.5,
            invert_mask=False,
            crop_context=64,
            crop_context_factor=1.2,
            min_size=512,
            max_size=1024,
            resize_mode="keep_aspect",
            expand_left=128,
            expand_right=0,
            expand_top=0,
            expand_bottom=0,
            canvas_fill="edge",
            auto_recommend="enabled",
        )

        self.assertEqual(work_image.shape, (1, 640, 768, 3))
        self.assertEqual(work_mask.shape, (1, 640, 768))
        self.assertEqual(repair_info["repair_scope"], "canvas")
        self.assertEqual(repair_info["canvas_expand"], [128, 0, 0, 0])
        self.assertEqual(recommended, 0.90)
        self.assertEqual(latent["source"], "repair_prepare_canvas")

    def test_prepare_requires_vae(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_prepare import LLSSimpleRepairPrepare

        node = LLSSimpleRepairPrepare()
        with self.assertRaisesRegex(RuntimeError, "Missing VAE"):
            node.prepare(
                image=FakeTensor((1, 512, 512, 3), "image"),
                mask=FakeMask(512, 512, bbox=(0, 0, 128, 128), area_ratio=0.1),
                vae=None,
                repair_scope="region",
                repair_kernel="latent_mask",
                task_hint="repair",
                mask_grow=8,
                mask_blur=8.0,
                mask_threshold=0.5,
                invert_mask=False,
                crop_context=64,
                crop_context_factor=1.2,
                min_size=512,
                max_size=1024,
                resize_mode="keep_aspect",
                expand_left=0,
                expand_right=0,
                expand_top=0,
                expand_bottom=0,
                canvas_fill="edge",
                auto_recommend="enabled",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_repair_prepare.py' -v`
Expected: FAIL because the node still raises the initial “not implemented” error and there are no helper functions for image/mask preparation.

- [ ] **Step 3: Implement the prepare helpers and node**

```python
# repair/repair_utils.py
def get_image_size(image) -> tuple[int, int]:
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) < 4:
        raise RuntimeError("[LLS] IMAGE input must be shaped like [batch, height, width, channels].")
    return int(shape[2]), int(shape[1])


def get_mask_metrics(mask, image_size: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, float]:
    bbox = getattr(mask, "mask_bbox", None)
    area_ratio = float(getattr(mask, "mask_area_ratio", 0.0))
    if bbox is None and area_ratio <= 0.0:
        return None, 0.0
    if bbox is not None:
        return clamp_box(tuple(int(value) for value in bbox), image_size), area_ratio
    image_width, image_height = image_size
    area = int(round(image_width * image_height * area_ratio))
    side = max(1, int(area ** 0.5))
    return (0, 0, min(side, image_width), min(side, image_height)), area_ratio


def resize_image_to(image, width: int, height: int):
    if hasattr(image, "resized"):
        return image.resized(width, height)
    raise RuntimeError("[LLS] IMAGE resizing requires ComfyUI image utilities in this runtime.")


def resize_mask_to(mask, width: int, height: int):
    if hasattr(mask, "resized"):
        return mask.resized(width, height)
    raise RuntimeError("[LLS] MASK resizing requires ComfyUI mask utilities in this runtime.")


def expand_canvas_image(image, width: int, height: int):
    if hasattr(image, "canvas_expanded"):
        return image.canvas_expanded(width, height)
    raise RuntimeError("[LLS] Canvas expansion requires ComfyUI image utilities in this runtime.")


def expand_canvas_mask(mask, width: int, height: int):
    if hasattr(mask, "canvas_expanded"):
        return mask.canvas_expanded(width, height)
    raise RuntimeError("[LLS] Canvas expansion requires ComfyUI mask utilities in this runtime.")


def make_noise_mask(mask, latent_samples):
    latent_shape = tuple(getattr(latent_samples, "shape", ()))
    if len(latent_shape) < 4:
        raise RuntimeError("[LLS] LATENT samples must expose a [batch, channels, height, width] shape.")
    return resize_mask_to(mask, int(latent_shape[3]), int(latent_shape[2]))
```

```python
# repair/repair_prepare.py
from __future__ import annotations

from .repair_utils import (
    build_canvas_info,
    compute_crop_box,
    expand_canvas_image,
    expand_canvas_mask,
    get_image_size,
    get_mask_metrics,
    make_noise_mask,
    normalize_model_info,
    recommend_denoise,
    resolve_repair_kernel,
    resolve_repair_scope,
    resolve_work_size,
    resize_image_to,
    resize_mask_to,
)


class LLSSimpleRepairPrepare:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "LLS_REPAIR_INFO", "FLOAT")
    RETURN_NAMES = ("latent", "work_image", "work_mask", "repair_info", "recommended_denoise")

    def prepare(
        self,
        image,
        mask,
        vae,
        repair_scope,
        repair_kernel,
        task_hint,
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
        model_info=None,
        positive=None,
        negative=None,
    ):
        if vae is None:
            raise RuntimeError("[LLS] Missing VAE. Connect the Loader VAE output or choose an external VAE in the loader.")

        image_width, image_height = get_image_size(image)
        model_meta = normalize_model_info(model_info)
        mask_bbox, mask_area_ratio = get_mask_metrics(mask, (image_width, image_height))
        scope = resolve_repair_scope(
            repair_scope,
            mask_area_ratio=mask_area_ratio,
            mask_bbox=mask_bbox,
            image_size=(image_width, image_height),
            canvas_expand=(expand_left, expand_right, expand_top, expand_bottom),
        )
        kernel, warnings = resolve_repair_kernel(
            repair_kernel,
            scope=scope,
            task_hint=task_hint,
            mask_area_ratio=mask_area_ratio,
            model_info=model_meta,
        )
        effective_kernel = "vae_inpaint" if kernel == "native_fill" else kernel
        if kernel == "native_fill":
            warnings.append("native_fill requested but runtime sampler support is not implemented yet; using vae_inpaint")
        denoise = recommend_denoise(task_hint, scope, effective_kernel, auto_recommend)

        base_info = {
            "repair_scope": scope,
            "repair_kernel": effective_kernel,
            "task_hint": task_hint,
            "original_size": [image_width, image_height],
            "work_size": [image_width, image_height],
            "crop_box": None,
            "crop_scale": None,
            "canvas_expand": [expand_left, expand_right, expand_top, expand_bottom],
            "mask_grow": mask_grow,
            "mask_blur": mask_blur,
            "mask_threshold": mask_threshold,
            "invert_mask": invert_mask,
            "recommended_denoise": denoise,
            "model_family": model_meta["model_family"],
            "model_role": model_meta["model_role"],
            "repair_payload_version": "1.0",
            "has_mask": mask_bbox is not None,
            "mask_area_ratio": mask_area_ratio,
            "mask_bbox": list(mask_bbox) if mask_bbox else None,
            "warnings": warnings,
        }

        if scope == "region":
            work_image = image
            work_mask = resize_mask_to(mask, image_width, image_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_region"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)
            return latent, work_image, work_mask, base_info, denoise

        if scope == "crop":
            if mask_bbox is None:
                raise RuntimeError("[LLS] crop repair requires a non-empty mask.")
            crop_box = compute_crop_box(
                mask_bbox=mask_bbox,
                image_size=(image_width, image_height),
                crop_context=crop_context,
                crop_context_factor=crop_context_factor,
            )
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            work_width, work_height, crop_scale = resolve_work_size(
                crop_size=(crop_width, crop_height),
                min_size=min_size,
                max_size=max_size,
                resize_mode=resize_mode,
            )
            work_image = resize_image_to(image, work_width, work_height)
            work_mask = resize_mask_to(mask, work_width, work_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_crop"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)
            base_info["crop_box"] = list(crop_box)
            base_info["crop_scale"] = crop_scale
            base_info["work_size"] = [work_width, work_height]
            return latent, work_image, work_mask, base_info, denoise

        if scope == "canvas":
            canvas_info = build_canvas_info((image_width, image_height), expand_left, expand_right, expand_top, expand_bottom)
            work_width, work_height = canvas_info["work_size"]
            work_image = expand_canvas_image(image, work_width, work_height)
            work_mask = expand_canvas_mask(mask, work_width, work_height)
            latent_samples = vae.encode(work_image)
            latent = {"samples": latent_samples, "source": "repair_prepare_canvas"}
            if effective_kernel == "latent_mask":
                latent["noise_mask"] = make_noise_mask(work_mask, latent_samples)
            base_info["work_size"] = [work_width, work_height]
            base_info["original_box_in_canvas"] = list(canvas_info["original_box"])
            return latent, work_image, work_mask, base_info, denoise

        raise RuntimeError(f"[LLS] Unsupported repair_scope '{scope}'.")
```

- [ ] **Step 4: Run the prepare tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_repair_prepare.py' -v`
Expected: PASS for `region`, `crop`, `canvas`, and missing-VAE cases.

- [ ] **Step 5: Commit the prepare node**

```bash
git add repair/repair_prepare.py repair/repair_utils.py tests/test_repair_helpers.py tests/test_repair_prepare.py
git commit -m "feat: add repair prepare node"
```

### Task 4: Implement `LLS Simple Repair Finish`

**Files:**
- Modify: `repair/repair_finish.py`
- Modify: `repair/repair_utils.py`
- Create: `tests/test_repair_finish.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair_finish.py
import unittest
from unittest import mock

from test_repair_helpers import FakeMask, FakeTensor, load_plugin_package


class TestRepairFinish(unittest.TestCase):
    def test_region_finish_dispatches_and_returns_final_preview(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_region_result",
            return_value=FakeTensor((1, 512, 512, 3), "final"),
        ) as compose, mock.patch.object(
            repair_finish,
            "build_preview_image",
            return_value=FakeTensor((1, 512, 512, 3), "preview"),
        ) as preview:
            final_image, preview_image = node.finish(
                original_image=FakeTensor((1, 512, 512, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={"repair_scope": "region", "repair_kernel": "latent_mask", "work_size": [512, 512]},
                feather=8.0,
                color_match="mean_std",
                brightness_match="enabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="soft",
                preview_mode="final",
                work_mask=FakeMask(512, 512, bbox=(64, 64, 256, 256), area_ratio=0.14),
                sample_info=None,
            )

        compose.assert_called_once()
        preview.assert_called_once()
        self.assertEqual(final_image.shape, (1, 512, 512, 3))
        self.assertEqual(preview_image.shape, (1, 512, 512, 3))

    def test_crop_finish_dispatches_with_crop_box(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_crop_result",
            return_value=FakeTensor((1, 1024, 1024, 3), "final"),
        ) as compose:
            final_image, _preview_image = node.finish(
                original_image=FakeTensor((1, 1024, 1024, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={
                    "repair_scope": "crop",
                    "repair_kernel": "vae_inpaint",
                    "crop_box": [100, 100, 356, 356],
                    "work_size": [512, 512],
                },
                feather=8.0,
                color_match="mean_std",
                brightness_match="enabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="none",
                preview_mode="compare",
                work_mask=FakeMask(512, 512, bbox=(0, 0, 256, 256), area_ratio=0.25),
                sample_info=None,
            )

        compose.assert_called_once()
        self.assertEqual(final_image.shape, (1, 1024, 1024, 3))

    def test_canvas_finish_dispatches_with_expanded_output(self):
        load_plugin_package()
        from lls_node_test_repair.repair import repair_finish

        node = repair_finish.LLSSimpleRepairFinish()
        with mock.patch.object(
            repair_finish,
            "compose_canvas_result",
            return_value=FakeTensor((1, 640, 768, 3), "final"),
        ) as compose:
            final_image, _preview_image = node.finish(
                original_image=FakeTensor((1, 640, 640, 3), "original"),
                generated_image=FakeTensor((1, 640, 768, 3), "generated"),
                repair_info={
                    "repair_scope": "canvas",
                    "repair_kernel": "vae_inpaint",
                    "work_size": [768, 640],
                    "canvas_expand": [128, 0, 0, 0],
                    "original_box_in_canvas": [128, 0, 768, 640],
                },
                feather=8.0,
                color_match="disabled",
                brightness_match="disabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="strong",
                preview_mode="before_after",
                work_mask=FakeMask(768, 640, bbox=(0, 0, 128, 640), area_ratio=0.16),
                sample_info=None,
            )

        compose.assert_called_once()
        self.assertEqual(final_image.shape, (1, 640, 768, 3))

    def test_finish_requires_supported_scope(self):
        load_plugin_package()
        from lls_node_test_repair.repair.repair_finish import LLSSimpleRepairFinish

        node = LLSSimpleRepairFinish()
        with self.assertRaisesRegex(RuntimeError, "Unsupported repair_scope"):
            node.finish(
                original_image=FakeTensor((1, 512, 512, 3), "original"),
                generated_image=FakeTensor((1, 512, 512, 3), "generated"),
                repair_info={"repair_scope": "mystery"},
                feather=8.0,
                color_match="disabled",
                brightness_match="disabled",
                blend_strength=1.0,
                restore_unmasked_area=True,
                edge_fix="none",
                preview_mode="final",
                work_mask=None,
                sample_info=None,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_repair_finish.py' -v`
Expected: FAIL because the finish node still raises the initial “not implemented” error and does not expose helper functions to patch.

- [ ] **Step 3: Implement the finish helpers and node**

```python
# repair/repair_utils.py
def build_preview_image(original_image, final_image, work_mask, preview_mode):
    if preview_mode == "mask" and work_mask is not None and hasattr(work_mask, "canvas_expanded"):
        width = int(getattr(work_mask, "shape", (1, 1, 1))[2])
        height = int(getattr(work_mask, "shape", (1, 1, 1))[1])
        return final_image.canvas_expanded(width, height) if hasattr(final_image, "canvas_expanded") else final_image
    if preview_mode in {"compare", "before_after"} and hasattr(final_image, "canvas_expanded"):
        final_width, final_height = get_image_size(final_image)
        return final_image.canvas_expanded(final_width * 2, final_height)
    return final_image


def compose_region_result(original_image, generated_image, work_mask, repair_info, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix):
    return generated_image if not restore_unmasked_area else original_image.resized(*get_image_size(generated_image))


def compose_crop_result(original_image, generated_image, work_mask, repair_info, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix):
    original_width, original_height = get_image_size(original_image)
    return original_image.resized(original_width, original_height)


def compose_canvas_result(original_image, generated_image, work_mask, repair_info, feather, color_match, brightness_match, blend_strength, restore_unmasked_area, edge_fix):
    work_width, work_height = repair_info["work_size"]
    return generated_image.resized(work_width, work_height)
```

```python
# repair/repair_finish.py
from __future__ import annotations

from .repair_utils import (
    build_preview_image,
    compose_canvas_result,
    compose_crop_result,
    compose_region_result,
    normalize_repair_info,
)


class LLSSimpleRepairFinish:
    CATEGORY = "LLS/Image Repair"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "preview_image")

    def finish(
        self,
        original_image,
        generated_image,
        repair_info,
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
        info = normalize_repair_info(repair_info)
        scope = info["repair_scope"]
        if scope == "region":
            final_image = compose_region_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        elif scope == "crop":
            final_image = compose_crop_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        elif scope == "canvas":
            final_image = compose_canvas_result(
                original_image,
                generated_image,
                work_mask,
                info,
                feather,
                color_match,
                brightness_match,
                blend_strength,
                restore_unmasked_area,
                edge_fix,
            )
        else:
            raise RuntimeError(f"[LLS] Unsupported repair_scope '{scope}'.")

        preview_image = build_preview_image(original_image, final_image, work_mask, preview_mode)
        return final_image, preview_image
```

- [ ] **Step 4: Run the finish tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_repair_finish.py' -v`
Expected: PASS for `region`, `crop`, `canvas`, and unsupported-scope coverage.

- [ ] **Step 5: Commit the finish node**

```bash
git add repair/repair_finish.py repair/repair_utils.py tests/test_repair_finish.py
git commit -m "feat: add repair finish node"
```

### Task 5: Upgrade `LLS Simple KSampler` for Repair Compatibility

**Files:**
- Modify: `sampling/nodes.py`
- Create: `tests/test_repair_sampler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair_sampler.py
import json
import types
import unittest
from unittest import mock

from test_repair_helpers import FakeLatentTensor, load_plugin_package


class TestRepairSampler(unittest.TestCase):
    def test_sampler_schema_exposes_repair_inputs(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleKSampler"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(schema["required"]["denoise_mode"][0], ["manual", "auto_from_repair"])
        self.assertEqual(schema["required"]["adapter_mode"][0], ["auto", "sd_classic", "flux", "sd3", "qwen", "zimage"])
        self.assertEqual(schema["optional"]["repair_info"][0], "LLS_REPAIR_INFO")
        self.assertEqual(schema["optional"]["guidance_stack"][0], "LLS_GUIDANCE_STACK")
        self.assertEqual(schema["optional"]["model_info"][0], "STRING")

    def test_sampler_keeps_manual_denoise_without_repair_info(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "empty_latent"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            return_value={"samples": "done"},
        ) as common:
            result_latent, sample_info = node.sample(
                model=types.SimpleNamespace(_lls_family="SD1.5"),
                positive="positive",
                negative="negative",
                latent_image=latent,
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="karras",
                denoise=0.33,
                denoise_mode="manual",
                adapter_mode="auto",
                flux_guidance=3.5,
                model_family="Auto",
                repair_info=None,
                guidance_stack=None,
                model_info=None,
            )

        self.assertEqual(result_latent["samples"], "done")
        self.assertEqual(common.call_args.kwargs["denoise"], 0.33)
        payload = json.loads(sample_info)
        self.assertFalse(payload["repair_mode"])
        self.assertFalse(payload["guidance_used"])

    def test_sampler_uses_repair_denoise_when_requested(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "repair_prepare_region"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            return_value={"samples": "done"},
        ) as common:
            _result_latent, sample_info = node.sample(
                model=types.SimpleNamespace(_lls_family="SDXL"),
                positive="positive",
                negative="negative",
                latent_image=latent,
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="karras",
                denoise=0.33,
                denoise_mode="auto_from_repair",
                adapter_mode="auto",
                flux_guidance=3.5,
                model_family="Auto",
                repair_info={
                    "repair_scope": "region",
                    "repair_kernel": "latent_mask",
                    "recommended_denoise": 0.61,
                    "model_family": "SDXL",
                    "model_role": "normal",
                },
                guidance_stack={"kind": "placeholder"},
                model_info=None,
            )

        self.assertEqual(common.call_args.kwargs["denoise"], 0.61)
        payload = json.loads(sample_info)
        self.assertTrue(payload["repair_mode"])
        self.assertTrue(payload["guidance_used"])
        self.assertEqual(payload["repair_scope"], "region")
        self.assertEqual(payload["repair_kernel"], "latent_mask")

    def test_sampler_rejects_unsupported_qwen_adapter(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "repair_prepare_region"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ):
            with self.assertRaisesRegex(RuntimeError, "QWEN"):
                node.sample(
                    model=types.SimpleNamespace(_lls_family="SD1.5"),
                    positive="positive",
                    negative="negative",
                    latent_image=latent,
                    quality_preset="Manual",
                    seed=7,
                    steps=20,
                    cfg=7.0,
                    sampler_name="euler",
                    scheduler="karras",
                    denoise=0.33,
                    denoise_mode="auto_from_repair",
                    adapter_mode="auto",
                    flux_guidance=3.5,
                    model_family="Auto",
                    repair_info={
                        "repair_scope": "region",
                        "repair_kernel": "vae_inpaint",
                        "recommended_denoise": 0.55,
                        "model_family": "QWEN",
                        "model_role": "unknown",
                    },
                    guidance_stack=None,
                    model_info=None,
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -p 'test_repair_sampler.py' -v`
Expected: FAIL because the sampler does not yet expose the new schema or repair-aware runtime behavior.

- [ ] **Step 3: Implement the sampler compatibility layer**

```python
# sampling/nodes.py
from ..repair.repair_utils import normalize_model_info, normalize_repair_info, resolve_adapter_mode


class LLSSimpleKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "quality_preset": (_QUALITY_PRESETS, {"default": FAMILY_DEFAULT_PRESET}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_samplers(), {"default": "euler_ancestral"}),
                "scheduler": (_get_schedulers(), {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (["manual", "auto_from_repair"], {"default": "manual"}),
                "adapter_mode": (["auto", "sd_classic", "flux", "sd3", "qwen", "zimage"], {"default": "auto"}),
                "flux_guidance": (
                    _PRIMITIVE_NUMBER_INPUT,
                    {"default": 3.5, "widgetType": "FLOAT", "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1},
                ),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "repair_info": ("LLS_REPAIR_INFO",),
                "guidance_stack": ("LLS_GUIDANCE_STACK",),
                "model_info": ("STRING",),
            },
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        quality_preset,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        denoise_mode,
        adapter_mode,
        flux_guidance,
        model_family="Auto",
        repair_info=None,
        guidance_stack=None,
        model_info=None,
    ):
        if comfy_sample is None:
            raise RuntimeError("[LLS] comfy.sample is not available. Make sure this node runs inside a ComfyUI environment.") from _COMFY_SAMPLE_ERR
        if comfy_samplers is None:
            raise RuntimeError("[LLS] comfy.samplers is not available. Make sure this node runs inside a ComfyUI environment.") from _COMFY_SAMPLERS_ERR

        repair_meta = normalize_repair_info(repair_info) if repair_info is not None else None
        model_meta = normalize_model_info(model_info)
        family = resolve_model_family(model_family, model=model)
        if repair_meta is not None and repair_meta.get("model_family") not in {"UNKNOWN", "", None}:
            family = str(repair_meta["model_family"])
        elif model_meta.get("model_family") not in {"UNKNOWN", "", None}:
            family = str(model_meta["model_family"])

        effective_adapter = resolve_adapter_mode(adapter_mode, family)
        if effective_adapter == "sd3":
            raise RuntimeError("[LLS] SD3 repair-aware sampling is not implemented yet.")
        if effective_adapter == "qwen":
            raise RuntimeError("[LLS] QWEN repair-aware sampling is not implemented yet.")
        if effective_adapter == "zimage":
            raise RuntimeError("[LLS] ZIMAGE repair-aware sampling is not implemented yet.")

        defaults = get_family_defaults(family)
        default_flux_guidance = defaults.get("default_guidance")
        effective_task_mode = infer_task_mode_from_latent(latent_image)
        if quality_preset == FAMILY_DEFAULT_PRESET:
            steps = int(defaults["default_steps"])
            cfg = float(defaults["default_cfg"])
            sampler_name = str(defaults["default_sampler"])
            scheduler = str(defaults["default_scheduler"])
            denoise = float(defaults["default_denoise"])
            if default_flux_guidance is not None:
                flux_guidance = float(default_flux_guidance)

        flux_guidance = _normalize_flux_guidance(flux_guidance, fallback=default_flux_guidance)
        actual_denoise = denoise
        if repair_meta is not None and denoise_mode == "auto_from_repair":
            actual_denoise = float(repair_meta.get("recommended_denoise", denoise))

        actual_seed = seed if seed != -1 else random.randint(0, 0xFFFFFFFFFFFFFFFF)
        result_latent = _common_ksampler(
            model=model,
            seed=actual_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=actual_denoise,
        )
        sample_info = info_to_json(
            {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "guidance": flux_guidance if is_flux_family(family) else None,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": actual_denoise,
                "quality_preset": quality_preset,
                "family": family,
                "task_mode": effective_task_mode,
                "repair_mode": repair_meta is not None,
                "repair_scope": repair_meta.get("repair_scope") if repair_meta else None,
                "repair_kernel": repair_meta.get("repair_kernel") if repair_meta else None,
                "model_family": family,
                "guidance_used": bool(guidance_stack),
            }
        )
        return result_latent, sample_info
```

- [ ] **Step 4: Run the sampler tests to verify they pass**

Run: `python3 -m unittest discover -s tests -p 'test_repair_sampler.py' -v`
Expected: PASS for the new schema, backward-compatible manual denoise behavior, repair-aware denoise override, and explicit unsupported-family errors.

- [ ] **Step 5: Commit the sampler upgrade**

```bash
git add sampling/nodes.py tests/test_repair_sampler.py
git commit -m "feat: add repair-aware sampler compatibility"
```

### Task 6: Update README and Run Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_repair_registration.py`
- Test: `tests/test_repair_utils.py`
- Test: `tests/test_repair_prepare.py`
- Test: `tests/test_repair_finish.py`
- Test: `tests/test_repair_sampler.py`
- Test: `tests/test_lls_universal.py`
- Test: `tests/test_loader_prompt_refactor.py`
- Test: `tests/test_model_info_inference.py`

- [ ] **Step 1: Update the README**

```markdown
## Image Repair

`LLS-node` now provides a dedicated local repair workflow built from:

- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

Repair does not start from `LLS Simple Empty Latent`. It starts from `image + mask + vae`, which are converted into a repair-aware latent payload by `LLS Simple Repair Prepare`.

### Minimal Repair Workflow

`Load Image -> Load Mask -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Simple Repair Prepare -> LLS Simple KSampler -> VAE Decode -> LLS Simple Repair Finish -> Preview Image`

### `repair_scope`

- `region`: repair inside the original image bounds
- `crop`: crop a local work area around the mask for higher-detail repair
- `canvas`: expand the canvas and repair missing or newly added area

### `repair_kernel`

- `latent_mask`: VAE encode the work image and attach a latent noise mask
- `vae_inpaint`: use VAE inpaint-style latent preparation
- `native_fill`: request native inpaint/fill behavior when the backend supports it, otherwise fall back with a warning

### `denoise_mode`

- `manual`: use the sampler `denoise` value directly
- `auto_from_repair`: use `repair_info["recommended_denoise"]`

### Compatibility

- Existing txt2img workflows remain unchanged when `repair_info` is not connected.
- Existing img2img workflows remain unchanged when `repair_info` is not connected.
- The same `LLS Simple KSampler` handles txt2img, img2img, and repair.
```

- [ ] **Step 2: Run the new focused test suite**

Run: `python3 -m unittest discover -s tests -p 'test_repair*.py' -v`
Expected: PASS across registration, utility, prepare, finish, and sampler repair coverage.

- [ ] **Step 3: Run the existing regression slices**

Run: `python3 tests/test_lls_universal.py`
Expected: PASS, confirming plugin registration still works.

Run: `python3 tests/test_loader_prompt_refactor.py`
Expected: PASS, confirming loader/prompt/latent workflow compatibility remains intact.

Run: `python3 tests/test_model_info_inference.py`
Expected: PASS, confirming model-family inference still matches downstream expectations.

- [ ] **Step 4: Run the compile check**

Run: `python3 -m compileall .`
Expected: PASS with all plugin modules compiling successfully.

- [ ] **Step 5: Record the required manual ComfyUI checks**

```text
1. Old txt2img workflow:
   Checkpoint Loader -> Prompt Encode -> Empty Latent -> KSampler -> VAE Decode

2. Old img2img workflow:
   Load Image -> Checkpoint Loader -> Prompt Encode -> VAE Encode -> KSampler -> VAE Decode

3. Region repair workflow:
   Load Image -> Load Mask -> Checkpoint Loader -> Prompt Encode -> Repair Prepare(repair_scope=region) -> KSampler(denoise_mode=auto_from_repair) -> VAE Decode -> Repair Finish

4. Crop repair workflow:
   Load Image -> Load Mask -> Repair Prepare(repair_scope=crop) -> KSampler -> VAE Decode -> Repair Finish

5. Canvas workflow:
   Load Image -> Repair Prepare(repair_scope=canvas, expand_left=512) -> KSampler -> VAE Decode -> Repair Finish
```

- [ ] **Step 6: Commit the docs and verification state**

```bash
git add README.md docs/superpowers/plans/2026-05-20-lls-simple-repair.md
git commit -m "docs: add repair workflow guidance"
```
