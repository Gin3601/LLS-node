# LLS Concat By Target Two-Port UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `LLS Concat By Target` so the ComfyUI node shows exactly two inputs labeled `image/mask_A` and `image/mask_B`, while still accepting either `IMAGE` or `MASK` on each side and preserving current concat behavior.

**Architecture:** Revert the backend schema to two internal wildcard inputs, `a` and `b`, because the `IMAGE,MASK` union type is what causes ComfyUI to render four visible sockets. Add a tiny `web/js` extension that relabels those two internal inputs to the user-facing names `image/mask_A` and `image/mask_B`, while keeping the existing runtime auto-detection and legacy keyword aliases intact.

**Tech Stack:** Python, unittest, torch, ComfyUI node contracts, lightweight frontend extension JavaScript

---

## File Map

- `utils/concat_by_target.py`
  Restore the schema to two wildcard inputs and keep runtime support for both internal `a` / `b` and legacy `image/mask_A` / `image/mask_B` keyword aliases.
- `web/js/lls_concat_by_target.js`
  Register a ComfyUI extension that renames the visible labels for the `a` and `b` inputs.
- `tests/test_concat_by_target_registration.py`
  Lock the backend schema contract to two wildcard inputs.
- `tests/test_concat_by_target_node.py`
  Keep the runtime contract explicit, including legacy keyword support and the existing image/mask auto-detection behavior.
- `tests/test_concat_by_target_frontend.py`
  Lock the new frontend asset and its label-rewrite contract.

### Task 1: Lock The Two-Port Contract With Failing Tests

**Files:**
- Modify: `tests/test_concat_by_target_registration.py`
- Create: `tests/test_concat_by_target_frontend.py`

- [ ] **Step 1: Update the registration test to expect two internal inputs**

```python
import unittest

try:
    from .test_mask_draw_helpers import load_plugin_package
except ImportError:
    from test_mask_draw_helpers import load_plugin_package


class TestConcatByTargetRegistration(unittest.TestCase):
    def test_plugin_registers_concat_by_target_node(self):
        plugin = load_plugin_package()

        self.assertIn("LLSConcatByTarget", plugin.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            plugin.NODE_DISPLAY_NAME_MAPPINGS["LLSConcatByTarget"],
            "LLS Concat By Target",
        )

    def test_concat_by_target_schema_matches_contract(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSConcatByTarget"]
        schema = node_cls.INPUT_TYPES()
        required = schema["required"]
        optional = schema["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Utils")
        self.assertEqual(node_cls.FUNCTION, "concat")
        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE", "MASK", "INT", "INT"))
        self.assertEqual(node_cls.RETURN_NAMES, ("image", "mask", "width", "height"))
        self.assertEqual(required["data_type"][0], ["IMAGE", "MASK"])
        self.assertEqual(required["target"][0], ["A", "B"])
        self.assertEqual(required["position"][0], ["top", "bottom", "left", "right"])
        self.assertEqual(required["resize_mode"][0], ["keep_proportion", "stretch", "none"])
        self.assertEqual(required["align"][0], ["start", "center", "end"])
        self.assertEqual(required["match_target_size"][0], "BOOLEAN")
        self.assertEqual(required["gap"][0], "INT")
        self.assertEqual(required["gap"][1]["default"], 0)
        self.assertEqual(required["background_color"][0], "STRING")
        self.assertEqual(required["background_color"][1]["default"], "#000000")
        self.assertEqual(required["background_value"][0], "FLOAT")
        self.assertEqual(required["background_value"][1]["default"], 0.0)
        self.assertEqual(required["multiple_of"][0], "INT")
        self.assertEqual(required["allow_batch_broadcast"][0], "BOOLEAN")
        self.assertEqual(str(optional["a"][0]), "*")
        self.assertEqual(str(optional["b"][0]), "*")
        self.assertNotIn("image/mask_A", optional)
        self.assertNotIn("image/mask_B", optional)
        self.assertNotIn("image_a", optional)
        self.assertNotIn("image_b", optional)
        self.assertNotIn("mask_a", optional)
        self.assertNotIn("mask_b", optional)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Add a failing frontend-contract test for the label adapter**

```python
import unittest

try:
    from .test_mask_draw_helpers import ROOT, load_plugin_package
except ImportError:
    from test_mask_draw_helpers import ROOT, load_plugin_package


class TestConcatByTargetFrontend(unittest.TestCase):
    def test_plugin_exports_web_directory(self):
        plugin = load_plugin_package()
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_frontend_asset_exists(self):
        asset = ROOT / "web" / "js" / "lls_concat_by_target.js"
        self.assertTrue(asset.exists(), msg=f"Missing frontend asset: {asset}")

    def test_frontend_asset_registers_concat_by_target_extension(self):
        asset = (ROOT / "web" / "js" / "lls_concat_by_target.js").read_text(encoding="utf-8")

        self.assertIn("app.registerExtension", asset)
        self.assertIn("LLSConcatByTarget", asset)
        self.assertIn("LLS Concat By Target", asset)
        self.assertIn("image/mask_A", asset)
        self.assertIn("image/mask_B", asset)
        self.assertIn("beforeRegisterNodeDef", asset)
        self.assertIn("onNodeCreated", asset)
        self.assertIn("onGraphConfigured", asset)
        self.assertIn("onConnectionsChange", asset)
        self.assertIn("input.label", asset)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the contract tests to verify they fail**

Run: `python3 -m unittest tests.test_concat_by_target_registration tests.test_concat_by_target_frontend -v`
Expected: FAIL because the schema still exposes `image/mask_A` / `image/mask_B` and `web/js/lls_concat_by_target.js` does not exist yet.


### Task 2: Restore The Backend To Two Internal Inputs

**Files:**
- Modify: `utils/concat_by_target.py`
- Modify: `tests/test_concat_by_target_node.py`
- Test: `tests/test_concat_by_target_registration.py`
- Test: `tests/test_concat_by_target_node.py`

- [ ] **Step 1: Rename the legacy-keyword runtime test so its purpose stays explicit after the schema change**

```python
def test_legacy_port_names_are_accepted(self):
    image_a = self.make_image(width=2, height=2, value=0.2)
    image_b = self.make_image(width=1, height=2, value=0.8)

    image, mask, width, height = self.node.concat(
        data_type="IMAGE",
        target="A",
        position="right",
        match_target_size=True,
        resize_mode="none",
        align="center",
        gap=0,
        background_color="#000000",
        background_value=0.0,
        multiple_of=0,
        allow_batch_broadcast=True,
        **{"image/mask_A": image_a, "image/mask_B": image_b},
    )

    self.assert_image_output(image, mask, width, height, (1, 2, 3, 3))
    self.assertTrue(torch.allclose(image[:, :, 0:2, :], torch.full((1, 2, 2, 3), 0.2)))
    self.assertTrue(torch.allclose(image[:, :, 2:3, :], torch.full((1, 2, 1, 3), 0.8)))
```

- [ ] **Step 2: Change the node schema to wildcard `a` / `b` inputs while keeping legacy alias resolution**

```python
DATA_TYPE_CHOICES = ["IMAGE", "MASK"]
TARGET_CHOICES = ["A", "B"]
POSITION_CHOICES = ["top", "bottom", "left", "right"]
RESIZE_MODE_CHOICES = ["keep_proportion", "stretch", "none"]
ALIGN_CHOICES = ["start", "center", "end"]
PORT_A_NAME = "image/mask_A"
PORT_B_NAME = "image/mask_B"


class AnyType(str):
    def __ne__(self, other):  # pragma: no cover - exercised by ComfyUI type checks
        return False


ANY_TYPE = AnyType("*")
```

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "data_type": (DATA_TYPE_CHOICES, {"default": "IMAGE"}),
            "target": (TARGET_CHOICES, {"default": "A"}),
            "position": (POSITION_CHOICES, {"default": "right"}),
            "match_target_size": ("BOOLEAN", {"default": True}),
            "resize_mode": (RESIZE_MODE_CHOICES, {"default": "keep_proportion"}),
            "align": (ALIGN_CHOICES, {"default": "center"}),
            "gap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "background_color": ("STRING", {"default": "#000000", "multiline": False}),
            "background_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "multiple_of": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            "allow_batch_broadcast": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "a": (ANY_TYPE,),
            "b": (ANY_TYPE,),
        },
    }
```

```python
def concat(
    self,
    data_type,
    target,
    position,
    match_target_size,
    resize_mode,
    align,
    gap,
    background_color,
    background_value,
    multiple_of,
    allow_batch_broadcast,
    a=None,
    b=None,
    **kwargs,
):
    _require_torch()
    data_type = str(data_type or "IMAGE")
    background_value = max(0.0, min(1.0, float(background_value)))
    input_a = kwargs[PORT_A_NAME] if PORT_A_NAME in kwargs else a
    input_b = kwargs[PORT_B_NAME] if PORT_B_NAME in kwargs else b
    resolved_data_type = resolve_concat_data_type(input_a, input_b, data_type)
```

- [ ] **Step 3: Run the backend tests to verify the two-port schema and runtime behavior pass**

Run: `python3 -m unittest tests.test_concat_by_target_registration tests.test_concat_by_target_node -v`
Expected: PASS, including the legacy keyword test and the existing image/mask behavior tests.

- [ ] **Step 4: Commit the backend schema change**

```bash
git add utils/concat_by_target.py tests/test_concat_by_target_registration.py tests/test_concat_by_target_node.py
git commit -m "refactor: restore two-port concat backend schema"
```


### Task 3: Add The Frontend Label Adapter

**Files:**
- Create: `web/js/lls_concat_by_target.js`
- Test: `tests/test_concat_by_target_frontend.py`
- Test: `tests/test_concat_by_target_registration.py`

- [ ] **Step 1: Implement the frontend extension that relabels `a` and `b`**

```javascript
import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "LLS.ConcatByTarget";
const TARGET_NODE_CLASS = "LLSConcatByTarget";
const TARGET_NODE_DISPLAY_NAME = "LLS Concat By Target";
const INPUT_LABELS = {
  a: "image/mask_A",
  b: "image/mask_B",
};

function relabelConcatInputs(node) {
  let changed = false;

  for (const input of node.inputs ?? []) {
    const targetLabel = INPUT_LABELS[input.name];
    if (targetLabel && input.label !== targetLabel) {
      input.label = targetLabel;
      changed = true;
    }
  }

  if (changed) {
    node.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS && nodeData.display_name !== TARGET_NODE_DISPLAY_NAME) {
      return;
    }

    const previousOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onConcatByTargetNodeCreated() {
      const result = previousOnNodeCreated?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };

    const previousOnGraphConfigured = nodeType.prototype.onGraphConfigured;
    nodeType.prototype.onGraphConfigured = function onConcatByTargetGraphConfigured() {
      const result = previousOnGraphConfigured?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };

    const previousOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onConcatByTargetConnectionsChange() {
      const result = previousOnConnectionsChange?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };
  },
});
```

- [ ] **Step 2: Run the frontend and contract tests to verify the label adapter passes**

Run: `python3 -m unittest tests.test_concat_by_target_frontend tests.test_concat_by_target_registration -v`
Expected: PASS because the asset exists, registers the extension, and the backend contract now exposes two internal inputs.

- [ ] **Step 3: Commit the frontend label adapter**

```bash
git add web/js/lls_concat_by_target.js tests/test_concat_by_target_frontend.py
git commit -m "feat: relabel concat by target ports in frontend"
```


### Task 4: Verify The Final Behavior End-To-End

**Files:**
- Test: `tests/test_concat_by_target_registration.py`
- Test: `tests/test_concat_by_target_node.py`
- Test: `tests/test_concat_by_target_frontend.py`

- [ ] **Step 1: Run the full concat test slice**

Run: `python3 -m unittest tests.test_concat_by_target_registration tests.test_concat_by_target_node tests.test_concat_by_target_frontend -v`
Expected: PASS with the full concat registration, runtime, and frontend suite green.

- [ ] **Step 2: Perform a manual ComfyUI smoke check**

Manual:
- Restart or reload ComfyUI so the new `web/js/lls_concat_by_target.js` asset is loaded.
- Create a fresh `LLS Concat By Target` node.
- Confirm the node shows exactly two visible input sockets.
- Confirm their labels are `image/mask_A` and `image/mask_B`.
- Confirm an `IMAGE` can be connected to each socket and the node still executes.
- Confirm a `MASK` can be connected to each socket and the node still executes.

- [ ] **Step 3: Check old-workflow compatibility explicitly**

Manual:
- If you have an existing workflow JSON that already contains `LLS Concat By Target`, load it after the backend schema change.
- Confirm whether the old links reconnect automatically.
- If they do not reconnect, record the limitation in the implementation handoff: old graphs need manual reconnection even though direct Python calls still accept `image/mask_A` / `image/mask_B` as legacy keyword aliases.

- [ ] **Step 4: Leave the worktree clean before merge or review**

Run: `git status --short`
Expected: no output.
