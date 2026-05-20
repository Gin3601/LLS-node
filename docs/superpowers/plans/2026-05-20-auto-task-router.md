# Auto Task Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `LLSAutoTaskRouter` with automatic task recognition, manual override fallback, model capability validation, and backward-compatible `LLS_TASK_CONTEXT` payloads.

**Architecture:** Shared route inference and validation live in `utils/task_context.py`. `task/nodes.py` exposes the new router and refactors `LLSTaskController` onto the same context builder. Compatibility is preserved by teaching `parse_task_context()` to normalize old and new schema shapes so current downstream nodes continue to work.

**Tech Stack:** Python 3.12, ComfyUI custom node APIs, `unittest`

---

### Task 1: Build Routing Core And Schema Normalization

**Files:**
- Create: `tests/test_task_router.py`
- Modify: `utils/task_context.py`

- [ ] **Step 1: Write the failing tests for automatic route detection and old/new context normalization**

```python
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK_CONTEXT_PATH = ROOT / "utils" / "task_context.py"


def load_task_context_module():
    spec = importlib.util.spec_from_file_location("lls_task_context_test", TASK_CONTEXT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTaskRoutingCore(unittest.TestCase):
    def test_build_routed_task_context_detects_outpaint_and_ordered_modifiers(self):
        task_context = load_task_context_module()

        context = task_context.build_routed_task_context(
            task_override="auto",
            model_family="SDXL",
            model_name="sdxl.safetensors",
            input_image=object(),
            mask_image=object(),
            control_image=object(),
            reference_image=object(),
            control_type="canny",
            instruction_prompt="make it cinematic",
            enable_upscale=True,
            enable_face_fix=True,
            enable_detail_refine=False,
            outpaint_left=64,
            outpaint_right=0,
            outpaint_top=0,
            outpaint_bottom=0,
        )

        self.assertEqual(context["mode"], "auto")
        self.assertEqual(context["primary_task"], "outpaint")
        self.assertEqual(
            context["modifiers"],
            ["controlnet", "reference", "instruction_edit", "upscale", "face_fix"],
        )
        self.assertEqual(
            context["route_key"],
            "outpaint.controlnet.reference.instruction_edit.upscale.face_fix",
        )
        self.assertEqual(context["task_mode"], "outpaint")
        self.assertTrue(context["enable_controlnet"])
        self.assertTrue(context["enable_reference"])
        self.assertTrue(context["enable_upscale"])

    def test_parse_task_context_backfills_legacy_and_modern_route_fields(self):
        task_context = load_task_context_module()

        legacy = task_context.parse_task_context(
            {
                "task_mode": "img2img",
                "enable_reference": True,
                "model_family": "SDXL",
            }
        )
        self.assertEqual(legacy["mode"], "manual")
        self.assertEqual(legacy["primary_task"], "img2img")
        self.assertEqual(legacy["modifiers"], ["reference"])
        self.assertEqual(legacy["route_key"], "img2img.reference")

        modern = task_context.parse_task_context(
            {
                "mode": "auto",
                "primary_task": "txt2img",
                "modifiers": ["upscale"],
                "model_family": "SD1.5",
            }
        )
        self.assertEqual(modern["task_mode"], "txt2img")
        self.assertTrue(modern["enable_upscale"])
        self.assertFalse(modern["enable_reference"])
        self.assertEqual(modern["route_key"], "txt2img.upscale")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python3 -B -m unittest tests.test_task_router.TestTaskRoutingCore`

Expected: FAIL with missing `build_routed_task_context` and missing route normalization behavior in `parse_task_context()`.

- [ ] **Step 3: Implement routing constants, inference helpers, and compatibility normalization**

```python
# utils/task_context.py

TASK_OVERRIDE_CHOICES = ["auto", "txt2img", "img2img", "inpaint", "outpaint"]
PRIMARY_TASK_CHOICES = ["txt2img", "img2img", "inpaint", "outpaint"]
_MODIFIER_ORDER = [
    "controlnet",
    "reference",
    "instruction_edit",
    "upscale",
    "face_fix",
    "detail_refine",
]


def _normalize_task_override(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    return normalized if normalized in TASK_OVERRIDE_CHOICES else "auto"


def _is_present(value: Any) -> bool:
    return value not in (None, "", False)


def infer_primary_task(
    task_override: str,
    input_image: Any = None,
    mask_image: Any = None,
    outpaint_left: int = 0,
    outpaint_right: int = 0,
    outpaint_top: int = 0,
    outpaint_bottom: int = 0,
) -> str:
    override = _normalize_task_override(task_override)
    if override != "auto":
        return override
    has_input = _is_present(input_image)
    has_mask = _is_present(mask_image)
    has_outpaint = any(int(value or 0) > 0 for value in (outpaint_left, outpaint_right, outpaint_top, outpaint_bottom))
    if has_input and has_mask and has_outpaint:
        return "outpaint"
    if has_input and has_mask:
        return "inpaint"
    if has_input:
        return "img2img"
    return "txt2img"


def infer_modifiers(
    control_image: Any = None,
    control_type: str = "none",
    reference_image: Any = None,
    instruction_prompt: str = "",
    enable_upscale: bool = False,
    enable_face_fix: bool = False,
    enable_detail_refine: bool = False,
) -> list[str]:
    modifiers: list[str] = []
    if _is_present(control_image) or str(control_type or "none").strip().lower() != "none":
        modifiers.append("controlnet")
    if _is_present(reference_image):
        modifiers.append("reference")
    if str(instruction_prompt or "").strip():
        modifiers.append("instruction_edit")
    if bool(enable_upscale):
        modifiers.append("upscale")
    if bool(enable_face_fix):
        modifiers.append("face_fix")
    if bool(enable_detail_refine):
        modifiers.append("detail_refine")
    return [name for name in _MODIFIER_ORDER if name in modifiers]


def build_route_key(primary_task: str, modifiers: list[str]) -> str:
    return primary_task if not modifiers else ".".join([primary_task, *modifiers])


def build_routed_task_context(
    task_override: str = "auto",
    model_family: str = "auto",
    model_name: str = "",
    input_image: Any = None,
    mask_image: Any = None,
    control_image: Any = None,
    reference_image: Any = None,
    control_type: str = "none",
    instruction_prompt: str = "",
    enable_upscale: bool = False,
    enable_face_fix: bool = False,
    enable_detail_refine: bool = False,
    outpaint_left: int = 0,
    outpaint_right: int = 0,
    outpaint_top: int = 0,
    outpaint_bottom: int = 0,
    base_context: dict[str, Any] | str | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    context = parse_task_context(base_context)
    primary_task = infer_primary_task(
        task_override=task_override,
        input_image=input_image,
        mask_image=mask_image,
        outpaint_left=outpaint_left,
        outpaint_right=outpaint_right,
        outpaint_top=outpaint_top,
        outpaint_bottom=outpaint_bottom,
    )
    modifiers = infer_modifiers(
        control_image=control_image,
        control_type=control_type,
        reference_image=reference_image,
        instruction_prompt=instruction_prompt,
        enable_upscale=enable_upscale,
        enable_face_fix=enable_face_fix,
        enable_detail_refine=enable_detail_refine,
    )
    context.update(
        {
            "mode": "manual" if _normalize_task_override(task_override) != "auto" else "auto",
            "primary_task": primary_task,
            "task_mode": primary_task,
            "modifiers": modifiers,
            "route_key": build_route_key(primary_task, modifiers),
            "model_family": model_family or context.get("model_family", "auto"),
            "model_name": model_name or context.get("model_name", ""),
            "enable_controlnet": "controlnet" in modifiers,
            "enable_reference": "reference" in modifiers,
            "enable_upscale": "upscale" in modifiers,
            "_has_input_image": _is_present(input_image),
            "_has_mask_image": _is_present(mask_image),
            "_has_control_image": _is_present(control_image),
            "_has_reference_image": _is_present(reference_image),
            "required_inputs": [],
            "optional_inputs": [],
            "warnings": [],
            "errors": [],
        }
    )
    return _finalize_task_context(context)


def parse_task_context(task_context: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = _parse_jsonish_dict(task_context)
    if isinstance(task_context, dict):
        raw = dict(task_context)
    if "primary_task" not in raw and raw.get("task_mode"):
        raw["primary_task"] = raw["task_mode"]
    if "task_mode" not in raw and raw.get("primary_task"):
        raw["task_mode"] = raw["primary_task"]
    if "mode" not in raw:
        raw["mode"] = "manual"
    if "modifiers" not in raw:
        modifiers = []
        if _normalize_bool(raw.get("enable_controlnet"), False):
            modifiers.append("controlnet")
        if _normalize_bool(raw.get("enable_reference"), False):
            modifiers.append("reference")
        if _normalize_bool(raw.get("enable_upscale"), False):
            modifiers.append("upscale")
        raw["modifiers"] = modifiers
    if "route_key" not in raw and raw.get("primary_task"):
        raw["route_key"] = build_route_key(str(raw["primary_task"]), list(raw.get("modifiers", [])))
    return _finalize_task_context(raw)
```

- [ ] **Step 4: Run the tests again to verify they pass**

Run: `python3 -B -m unittest tests.test_task_router.TestTaskRoutingCore`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_task_router.py utils/task_context.py
git commit -m "feat: add auto task routing core"
```

### Task 2: Add Model Capability Validation And Input Requirements

**Files:**
- Modify: `utils/task_context.py`
- Modify: `tests/test_task_router.py`

- [ ] **Step 1: Extend the test file with failing validation tests**

```python
class TestTaskRouteValidation(unittest.TestCase):
    def test_build_routed_task_context_rejects_unsupported_primary_task(self):
        task_context = load_task_context_module()

        with self.assertRaisesRegex(RuntimeError, "does not support primary task 'outpaint'"):
            task_context.build_routed_task_context(
                task_override="auto",
                model_family="FLUX_SCHNELL",
                model_name="flux1-schnell.safetensors",
                input_image=object(),
                mask_image=object(),
                outpaint_left=32,
                outpaint_right=0,
                outpaint_top=0,
                outpaint_bottom=0,
            )

    def test_build_routed_task_context_rejects_unsupported_modifier(self):
        task_context = load_task_context_module()

        with self.assertRaisesRegex(RuntimeError, "modifier 'controlnet'"):
            task_context.build_routed_task_context(
                task_override="auto",
                model_family="FLUX_DEV",
                model_name="flux-dev.safetensors",
                control_image=object(),
                control_type="canny",
            )

    def test_build_routed_task_context_exposes_required_and_optional_inputs(self):
        task_context = load_task_context_module()

        context = task_context.build_routed_task_context(
            task_override="auto",
            model_family="SDXL",
            model_name="sdxl.safetensors",
            input_image=object(),
            mask_image=object(),
            reference_image=object(),
            outpaint_left=0,
            outpaint_right=0,
            outpaint_top=0,
            outpaint_bottom=0,
        )

        self.assertEqual(context["primary_task"], "inpaint")
        self.assertEqual(context["required_inputs"], ["input_image", "mask_image"])
        self.assertIn("reference_image", context["optional_inputs"])
        self.assertEqual(context["warnings"], [])
        self.assertEqual(context["errors"], [])
```

- [ ] **Step 2: Run the validation tests to verify they fail**

Run: `python3 -B -m unittest tests.test_task_router.TestTaskRouteValidation`

Expected: FAIL because `MODEL_REGISTRY`, required input derivation, and route validation are not implemented yet.

- [ ] **Step 3: Implement registry lookup, input requirements, and validation**

```python
# utils/task_context.py

MODEL_REGISTRY = {
    "SD1.5": {
        "primary_tasks": {"txt2img", "img2img", "inpaint", "outpaint"},
        "modifiers": {"controlnet", "reference", "instruction_edit", "upscale", "face_fix", "detail_refine"},
    },
    "SDXL": {
        "primary_tasks": {"txt2img", "img2img", "inpaint", "outpaint"},
        "modifiers": {"controlnet", "reference", "instruction_edit", "upscale", "face_fix", "detail_refine"},
    },
    "SDXL_TURBO": {
        "primary_tasks": {"txt2img", "img2img"},
        "modifiers": {"upscale"},
    },
    "FLUX_DEV": {
        "primary_tasks": {"txt2img", "img2img", "inpaint", "outpaint"},
        "modifiers": {"reference", "instruction_edit", "upscale", "face_fix", "detail_refine"},
    },
    "FLUX_SCHNELL": {
        "primary_tasks": {"txt2img", "img2img"},
        "modifiers": {"upscale"},
    },
}

MODEL_NAME_OVERRIDES: dict[str, dict[str, set[str]]] = {}


def _resolve_registry_family(model_family: str) -> str:
    requested_family = _normalize_requested_family(model_family)
    if requested_family == "FLUX":
        return "FLUX_DEV"
    if requested_family == "auto":
        return "SD1.5"
    return _normalize_resolved_family(requested_family, requested_family)


def resolve_model_capabilities(model_family: str, model_name: str) -> dict[str, set[str]]:
    if model_name in MODEL_NAME_OVERRIDES:
        return MODEL_NAME_OVERRIDES[model_name]
    registry_family = _resolve_registry_family(model_family)
    return MODEL_REGISTRY.get(registry_family, MODEL_REGISTRY["SD1.5"])


def derive_route_inputs(primary_task: str, modifiers: list[str]) -> tuple[list[str], list[str]]:
    required: list[str] = []
    optional: list[str] = []
    if primary_task in {"img2img", "inpaint", "outpaint"}:
        required.append("input_image")
    if primary_task in {"inpaint", "outpaint"}:
        required.append("mask_image")
    if "controlnet" in modifiers:
        required.append("control_image")
    if "reference" in modifiers:
        optional.append("reference_image")
    if "instruction_edit" in modifiers:
        optional.append("instruction_prompt")
    if "upscale" in modifiers:
        optional.append("enable_upscale")
    if "face_fix" in modifiers:
        optional.append("enable_face_fix")
    if "detail_refine" in modifiers:
        optional.append("enable_detail_refine")
    return required, optional


def validate_route_context(context: dict[str, Any]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    capabilities = resolve_model_capabilities(
        str(context.get("resolved_model_family") or context.get("model_family") or "SD1.5"),
        str(context.get("model_name") or ""),
    )
    primary_task = str(context["primary_task"])
    modifiers = list(context["modifiers"])
    if primary_task not in capabilities["primary_tasks"]:
        errors.append(
            f"model family '{context['resolved_model_family']}' does not support primary task '{primary_task}'"
        )
    for modifier in modifiers:
        if modifier not in capabilities["modifiers"]:
            errors.append(
                f"model family '{context['resolved_model_family']}' does not support modifier '{modifier}'"
            )
    if primary_task in {"inpaint", "outpaint"} and not context.get("_has_mask_image", False):
        errors.append(f"primary task '{primary_task}' requires mask_image")
    if primary_task in {"img2img", "inpaint", "outpaint"} and not context.get("_has_input_image", False):
        errors.append(f"primary task '{primary_task}' requires input_image")
    if "controlnet" in modifiers and not context.get("_has_control_image", False):
        errors.append("modifier 'controlnet' requires control_image")
    return warnings, errors


def build_routed_task_context(
    task_override: str = "auto",
    model_family: str = "auto",
    model_name: str = "",
    input_image: Any = None,
    mask_image: Any = None,
    control_image: Any = None,
    reference_image: Any = None,
    control_type: str = "none",
    instruction_prompt: str = "",
    enable_upscale: bool = False,
    enable_face_fix: bool = False,
    enable_detail_refine: bool = False,
    outpaint_left: int = 0,
    outpaint_right: int = 0,
    outpaint_top: int = 0,
    outpaint_bottom: int = 0,
    base_context: dict[str, Any] | str | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    context = parse_task_context(base_context)
    primary_task = infer_primary_task(
        task_override=task_override,
        input_image=input_image,
        mask_image=mask_image,
        outpaint_left=outpaint_left,
        outpaint_right=outpaint_right,
        outpaint_top=outpaint_top,
        outpaint_bottom=outpaint_bottom,
    )
    modifiers = infer_modifiers(
        control_image=control_image,
        control_type=control_type,
        reference_image=reference_image,
        instruction_prompt=instruction_prompt,
        enable_upscale=enable_upscale,
        enable_face_fix=enable_face_fix,
        enable_detail_refine=enable_detail_refine,
    )
    required_inputs, optional_inputs = derive_route_inputs(primary_task, modifiers)
    context.update(
        {
            "mode": "manual" if _normalize_task_override(task_override) != "auto" else "auto",
            "primary_task": primary_task,
            "task_mode": primary_task,
            "modifiers": modifiers,
            "route_key": build_route_key(primary_task, modifiers),
            "model_family": model_family or context.get("model_family", "auto"),
            "resolved_model_family": _resolve_registry_family(model_family or context.get("model_family", "auto")),
            "model_name": model_name or context.get("model_name", ""),
            "enable_controlnet": "controlnet" in modifiers,
            "enable_reference": "reference" in modifiers,
            "enable_upscale": "upscale" in modifiers,
            "_has_input_image": _is_present(input_image),
            "_has_mask_image": _is_present(mask_image),
            "_has_control_image": _is_present(control_image),
            "_has_reference_image": _is_present(reference_image),
            "required_inputs": required_inputs,
            "optional_inputs": optional_inputs,
        }
    )
    warnings, errors = validate_route_context(context)
    context["warnings"] = warnings
    context["errors"] = errors
    if errors and raise_on_error:
        raise RuntimeError("[LLS] Auto task routing failed: " + "; ".join(errors))
    return _finalize_task_context(context)
```

- [ ] **Step 4: Run the validation tests again to verify they pass**

Run: `python3 -B -m unittest tests.test_task_router.TestTaskRouteValidation`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_task_router.py utils/task_context.py
git commit -m "feat: validate auto task routes"
```

### Task 3: Add `LLSAutoTaskRouter` Node

**Files:**
- Modify: `task/nodes.py`
- Modify: `tests/test_task_router.py`

- [ ] **Step 1: Add failing node tests for schema and runtime behavior**

```python
import importlib.util
import pathlib
import sys


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_task_router_plugin_test",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_task_router_plugin_test"] = module
    spec.loader.exec_module(module)
    return module


class TestAutoTaskRouterNode(unittest.TestCase):
    def test_auto_task_router_schema_matches_design(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSAutoTaskRouter"]
        schema = node_cls.INPUT_TYPES()
        self.assertEqual(node_cls.CATEGORY, "LLS/Task")
        self.assertEqual(node_cls.FUNCTION, "route")
        self.assertEqual(node_cls.RETURN_TYPES, ("LLS_TASK_CONTEXT",))
        self.assertIn("task_override", schema["required"])
        self.assertIn("model_family", schema["required"])
        self.assertIn("model_name", schema["required"])
        self.assertIn("input_image", schema["optional"])
        self.assertIn("mask_image", schema["optional"])
        self.assertIn("control_image", schema["optional"])
        self.assertIn("reference_image", schema["optional"])

    def test_auto_task_router_returns_context_for_auto_route(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSAutoTaskRouter"]()
        (context,) = node.route(
            "auto",
            "SDXL",
            "sdxl.safetensors",
            "none",
            "",
            False,
            False,
            False,
            0,
            0,
            0,
            0,
            input_image=object(),
        )
        self.assertEqual(context["mode"], "auto")
        self.assertEqual(context["primary_task"], "img2img")
        self.assertEqual(context["route_key"], "img2img")

    def test_auto_task_router_raises_for_invalid_supported_route(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSAutoTaskRouter"]()
        with self.assertRaisesRegex(RuntimeError, "does not support primary task 'outpaint'"):
            node.route(
                "auto",
                "FLUX_SCHNELL",
                "flux1-schnell.safetensors",
                "none",
                "",
                False,
                False,
                False,
                32,
                0,
                0,
                0,
                input_image=object(),
                mask_image=object(),
            )
```

- [ ] **Step 2: Run the node tests to verify they fail**

Run: `python3 -B -m unittest tests.test_task_router.TestAutoTaskRouterNode`

Expected: FAIL because `LLSAutoTaskRouter` is not registered yet.

- [ ] **Step 3: Implement the new node and register it**

```python
# task/nodes.py

from ..utils.task_context import (
    LLS_TASK_CONTEXT_TYPE,
    QUALITY_PRESET_CHOICES,
    TASK_CONTROLLER_MODEL_FAMILY_CHOICES,
    TASK_MODE_CHOICES,
    TASK_OVERRIDE_CHOICES,
    WORKFLOW_PRESET_CHOICES,
    build_routed_task_context,
    create_task_context,
    task_context_to_json,
)


class LLSAutoTaskRouter:
    CATEGORY = "LLS/Task"
    FUNCTION = "route"
    RETURN_TYPES = (LLS_TASK_CONTEXT_TYPE,)
    RETURN_NAMES = ("task_context",)
    DESCRIPTION = "Automatically detect the primary generation task and emit a validated LLS routing context."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_override": (TASK_OVERRIDE_CHOICES, {"default": "auto"}),
                "model_family": (TASK_CONTROLLER_MODEL_FAMILY_CHOICES, {"default": "auto"}),
                "model_name": ("STRING", {"default": ""}),
                "control_type": ("STRING", {"default": "none"}),
                "instruction_prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "enable_face_fix": ("BOOLEAN", {"default": False}),
                "enable_detail_refine": ("BOOLEAN", {"default": False}),
                "outpaint_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "mask_image": ("MASK",),
                "control_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            },
        }

    def route(
        self,
        task_override: str,
        model_family: str,
        model_name: str,
        control_type: str,
        instruction_prompt: str,
        enable_upscale: bool,
        enable_face_fix: bool,
        enable_detail_refine: bool,
        outpaint_left: int,
        outpaint_right: int,
        outpaint_top: int,
        outpaint_bottom: int,
        input_image=None,
        mask_image=None,
        control_image=None,
        reference_image=None,
    ):
        return (
            build_routed_task_context(
                task_override=task_override,
                model_family=model_family,
                model_name=model_name,
                input_image=input_image,
                mask_image=mask_image,
                control_image=control_image,
                reference_image=reference_image,
                control_type=control_type,
                instruction_prompt=instruction_prompt,
                enable_upscale=enable_upscale,
                enable_face_fix=enable_face_fix,
                enable_detail_refine=enable_detail_refine,
                outpaint_left=outpaint_left,
                outpaint_right=outpaint_right,
                outpaint_top=outpaint_top,
                outpaint_bottom=outpaint_bottom,
            ),
        )


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSTaskController": LLSTaskController,
    "LLSTaskInspector": LLSTaskInspector,
    "LLSAutoTaskRouter": LLSAutoTaskRouter,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSTaskController": "LLS Task Controller",
    "LLSTaskInspector": "LLS Task Inspector",
    "LLSAutoTaskRouter": "LLS Auto Task Router",
}
```

- [ ] **Step 4: Run the node tests again to verify they pass**

Run: `python3 -B -m unittest tests.test_task_router.TestAutoTaskRouterNode`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add task/nodes.py tests/test_task_router.py utils/task_context.py
git commit -m "feat: add auto task router node"
```

### Task 4: Refactor `LLSTaskController` And Lock Down Backward Compatibility

**Files:**
- Modify: `task/nodes.py`
- Modify: `tests/test_loader_prompt_refactor.py`
- Modify: `tests/test_task_router.py`

- [ ] **Step 1: Extend existing compatibility tests to require the new route fields from `LLSTaskController`**

```python
def test_task_controller_generates_context_with_recommended_defaults(self):
    plugin = load_plugin_package()
    node_cls = plugin.NODE_CLASS_MAPPINGS["LLSTaskController"]
    node = node_cls()

    (task_context,) = node.execute(
        "img2img",
        "SDXL",
        "standard",
        "balanced",
        True,
        False,
        False,
        False,
        False,
    )

    self.assertEqual(task_context["mode"], "manual")
    self.assertEqual(task_context["primary_task"], "img2img")
    self.assertEqual(task_context["modifiers"], ["upscale"])
    self.assertEqual(task_context["route_key"], "img2img.upscale")
    self.assertEqual(task_context["task_mode"], "img2img")
    self.assertTrue(task_context["enable_upscale"])
    self.assertEqual(task_context["recommended_width"], 1024)
    self.assertEqual(task_context["recommended_height"], 1024)
```

- [ ] **Step 2: Add a regression test proving downstream nodes accept the new router-shaped context**

```python
def test_ksampler_accepts_router_context_shape_without_legacy_only_fields(self):
    load_plugin_package()
    from lls_node_test_refactor.sampling import nodes as sampling_nodes

    recorded = {}

    def fake_common_ksampler(**kwargs):
        recorded.update(kwargs)
        return kwargs["latent"]

    with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
        sampling_nodes,
        "comfy_samplers",
        object(),
    ), mock.patch.object(
        sampling_nodes,
        "_common_ksampler",
        side_effect=fake_common_ksampler,
    ):
        node = sampling_nodes.LLSSimpleKSampler()
        latent, sample_info, task_context = node.sample(
            model="model",
            positive=[["pos", {}]],
            negative=[["neg", {}]],
            latent_image={"samples": FakeTensor((1, 4, 64, 64)), "source": "image_encode"},
            quality_preset="Family Default",
            seed=1,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            flux_guidance=3.5,
            task_context={
                "mode": "auto",
                "primary_task": "img2img",
                "modifiers": ["upscale"],
                "route_key": "img2img.upscale",
                "resolved_model_family": "SDXL_TURBO",
                "recommended_steps": 4,
                "recommended_cfg": 1.0,
                "recommended_denoise": 0.5,
                "recommended_sampler": "euler",
                "recommended_scheduler": "normal",
            },
        )

    self.assertEqual(recorded["steps"], 4)
    self.assertEqual(recorded["denoise"], 0.5)
    self.assertEqual(task_context["task_mode"], "img2img")
```

- [ ] **Step 3: Refactor `LLSTaskController` to use the shared routed context builder**

```python
# task/nodes.py

class LLSTaskController:
    CATEGORY = "LLS-node"
    FUNCTION = "execute"
    RETURN_TYPES = (LLS_TASK_CONTEXT_TYPE,)
    RETURN_NAMES = ("task_context",)

    def execute(
        self,
        task_mode: str,
        model_family: str,
        workflow_preset: str,
        quality_preset: str,
        enable_upscale: bool,
        enable_controlnet: bool,
        enable_reference: bool,
        use_external_vae: bool,
        use_external_text_encoder: bool,
    ):
        modifiers = {
            "control_type": "manual" if enable_controlnet else "none",
            "enable_upscale": enable_upscale,
        }
        if enable_reference:
            modifiers["reference_image"] = object()
        context = build_routed_task_context(
            task_override=task_mode,
            model_family=model_family,
            model_name="",
            control_type=modifiers["control_type"],
            enable_upscale=enable_upscale,
            reference_image=modifiers.get("reference_image"),
            enable_face_fix=False,
            enable_detail_refine=False,
            base_context=create_task_context(
                task_mode=task_mode,
                model_family=model_family,
                workflow_preset=workflow_preset,
                quality_preset=quality_preset,
                enable_upscale=enable_upscale,
                enable_controlnet=enable_controlnet,
                enable_reference=enable_reference,
                use_external_vae=use_external_vae,
                use_external_text_encoder=use_external_text_encoder,
            ),
        )
        context["mode"] = "manual"
        return (context,)
```

- [ ] **Step 4: Run the compatibility-focused tests**

Run: `python3 -B -m unittest tests.test_task_router tests.test_loader_prompt_refactor`

Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python3 -B -m unittest discover -s tests`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add task/nodes.py tests/test_loader_prompt_refactor.py tests/test_task_router.py utils/task_context.py
git commit -m "feat: unify manual and auto task routing"
```
