# LLS Simple Mask Create Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LLS Simple Mask Create` node that generates geometric repair masks, preview overlays, and area metadata directly from an input image.

**Architecture:** Implement the feature in the existing `mask/` package with a focused utility module for geometry, mask processing, and preview composition plus a node module for ComfyUI schema/orchestration. Reuse the existing repair workflow by outputting a standard ComfyUI `MASK` tensor and a dict-like `LLS_MASK_INFO` payload.

**Tech Stack:** Python, unittest, torch, ComfyUI node definitions, existing LLS repair pipeline.

---

## File Structure

- Create: `mask/mask_create.py`
  - Node class and public schema.
- Create: `mask/mask_utils.py`
  - Geometry creation, preview rendering, mask combine logic, and area calculations.
- Modify: `mask/nodes.py`
  - Import and expose the new node registration maps.
- Modify: `tests/test_mask_draw_helpers.py`
  - Reuse helpers for mask-create tensor tests if additional builders are needed.
- Create: `tests/test_mask_create_registration.py`
  - Registration and schema checks.
- Create: `tests/test_mask_create_node.py`
  - Shape generation, coordinate modes, overlay preview, area_info, and repair compatibility.
- Create: `tests/test_mask_create_docs.py`
  - README coverage checks.
- Modify: `README.md`
  - Add node documentation and workflow examples.

### Task 1: Lock Registration and Schema with Failing Tests

**Files:**
- Create: `tests/test_mask_create_registration.py`
- Modify: `mask/nodes.py`
- Create: `mask/mask_create.py`

- [ ] **Step 1: Write the failing test**

```python
def test_plugin_registers_mask_create_node(self):
    plugin = load_plugin_package()
    self.assertIn("LLSSimpleMaskCreate", plugin.NODE_CLASS_MAPPINGS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_mask_create_registration -v`
Expected: FAIL because `LLSSimpleMaskCreate` is not registered yet.

- [ ] **Step 3: Write minimal implementation**

```python
class LLSSimpleMaskCreate:
    CATEGORY = "LLS/Mask"
    FUNCTION = "create_mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "LLS_MASK_INFO")
    RETURN_NAMES = ("image", "mask", "preview_image", "area_info")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_mask_create_registration -v`
Expected: PASS

### Task 2: Add Failing Runtime Tests for Geometry and Metadata

**Files:**
- Create: `tests/test_mask_create_node.py`
- Create: `mask/mask_utils.py`
- Modify: `mask/mask_create.py`

- [ ] **Step 1: Write failing tests for rectangle, square, circle, ellipse, and percent/pixel modes**

```python
def test_circle_percent_mode_creates_center_mask(self):
    image = make_image(width=100, height=80, color=0.2)
    image_out, mask, preview, area_info = self.node.create_mask(
        image=image,
        shape_type="circle",
        coordinate_mode="percent",
        center_x=0.5,
        center_y=0.5,
        width=0.3,
        height=0.3,
        radius=0.15,
        feather=0.0,
        blur=0.0,
        invert_mask=False,
        combine_mode="replace",
        overlay_alpha=0.4,
        overlay_color="red",
        input_mask=None,
    )
    self.assertGreater(area_info["binary_area_px"], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_mask_create_node -v`
Expected: FAIL because geometry creation and metadata are not implemented yet.

- [ ] **Step 3: Implement minimal geometry utilities and node orchestration**

```python
def create_shape_mask(...):
    if shape_type == "rectangle":
        ...
    elif shape_type == "circle":
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_mask_create_node -v`
Expected: PASS

### Task 3: Cover Combine Modes, Preview Overlay, and Repair Compatibility

**Files:**
- Modify: `tests/test_mask_create_node.py`
- Modify: `mask/mask_utils.py`

- [ ] **Step 1: Write failing tests for `union`, `subtract`, `intersect`, and `LLS Simple Repair Prepare` compatibility**

```python
def test_output_mask_connects_to_repair_prepare(self):
    latent, work_image, work_mask, repair_info, *_ = self.repair_prepare_node.prepare(...)
    self.assertTrue(repair_info["has_mask"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_mask_create_node -v`
Expected: FAIL until combine logic and output normalization are correct.

- [ ] **Step 3: Implement combine logic, mask resize, preview composition, and area_info finalization**

```python
if combine_mode == "union":
    final_mask = torch.maximum(input_mask, created_mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_mask_create_node -v`
Expected: PASS

### Task 4: Document the Node and Verify README Coverage

**Files:**
- Create: `tests/test_mask_create_docs.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing docs test**

```python
def test_readme_documents_mask_create_node(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    self.assertIn("LLS Simple Mask Create", readme)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_mask_create_docs -v`
Expected: FAIL because the README does not mention the new node yet.

- [ ] **Step 3: Update README with node description, inputs/outputs, area_info, and workflow examples**

```markdown
### `LLS Simple Mask Create`

`Load Image -> LLS Simple Mask Create -> LLS Simple Repair Prepare`
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_mask_create_docs -v`
Expected: PASS

### Task 5: Final Verification

**Files:**
- Modify: `mask/mask_create.py`
- Modify: `mask/mask_utils.py`
- Modify: `mask/nodes.py`
- Modify: `README.md`
- Modify: `tests/test_mask_create_registration.py`
- Modify: `tests/test_mask_create_node.py`
- Modify: `tests/test_mask_create_docs.py`

- [ ] **Step 1: Run the focused test suite**

Run: `python3 -m unittest tests.test_mask_create_registration tests.test_mask_create_node tests.test_mask_create_docs -v`
Expected: PASS

- [ ] **Step 2: Run compatibility tests for the existing repair chain**

Run: `python3 -m unittest tests.test_mask_draw_node tests.test_repair_prepare -v`
Expected: PASS

- [ ] **Step 3: Compile touched packages**

Run: `python3 -m compileall mask repair`
Expected: PASS with no syntax errors

- [ ] **Step 4: Review changed files**

Run: `git diff -- mask README.md tests`
Expected: Shows only mask-create implementation, tests, and README changes.

