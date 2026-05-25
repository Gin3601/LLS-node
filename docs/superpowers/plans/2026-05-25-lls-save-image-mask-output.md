# LLS Save Image MASK Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `LLS Save Image` so one node can independently save or preview both `IMAGE` and `MASK`, and remove the obsolete `LLS Simple Mask Preview` node.

**Architecture:** Keep all output-node behavior inside `image/nodes.py`, reusing `mask.mask_utils.mask_to_image()` for `MASK -> IMAGE` conversion. Remove the standalone mask preview registration and update docs/tests so the workflow points directly at `LLS Save Image` or plain `Preview Image`.

**Tech Stack:** Python, unittest, ComfyUI node contracts, existing test doubles in `tests/test_loader_prompt_refactor.py`

---

### Task 1: Lock The New Save Image Contract With Failing Tests

**Files:**
- Modify: `tests/test_loader_prompt_refactor.py`
- Test: `tests/test_loader_prompt_refactor.py`

- [ ] **Step 1: Write the failing schema test for the optional `mask` input**

```python
    def test_save_image_schema_accepts_optional_mask_input(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSaveImage"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(optional["mask"], ("MASK",))
```

- [ ] **Step 2: Run the schema test to verify it fails**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "save_image_schema_accepts_optional_mask_input" -v`
Expected: FAIL with `KeyError: 'mask'` or an equivalent assertion showing the schema does not expose `mask`.

- [ ] **Step 3: Write the failing preview test for dual output**

```python
    def test_save_image_preview_only_emits_image_and_mask_previews(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 4, 4, 3)),
                mask=FakeTensor((1, 4, 4)),
                filename_prefix="LLS",
                save_metadata=True,
                output_mode="preview_only",
            )

        self.assertEqual(CoreNodesStub.save_calls, [])
        self.assertEqual(len(CoreNodesStub.preview_calls), 2)
        self.assertEqual(tuple(CoreNodesStub.preview_calls[1]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(result["ui"]["images"][0]["type"], "temp")
        self.assertEqual(len(result["ui"]["images"]), 2)
```

- [ ] **Step 4: Run the preview test to verify it fails**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "save_image_preview_only_emits_image_and_mask_previews" -v`
Expected: FAIL because only one preview call is made and the returned UI list only contains the image preview.

- [ ] **Step 5: Write the failing save-mode test for the mask suffix and metadata split**

```python
    def test_save_image_save_mode_emits_separate_mask_file_without_lls_metadata(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 4, 4, 3)),
                mask=FakeTensor((1, 4, 4)),
                filename_prefix="LLS",
                save_metadata=True,
                prompt_info=json.dumps({"positive_prompt": "cat"}),
            )

        self.assertEqual(len(CoreNodesStub.save_calls), 2)
        self.assertEqual(CoreNodesStub.save_calls[0]["filename_prefix"], "LLS")
        self.assertEqual(CoreNodesStub.save_calls[1]["filename_prefix"], "LLS_mask")
        self.assertIn("lls_metadata", CoreNodesStub.save_calls[0]["extra_pnginfo"])
        self.assertEqual(CoreNodesStub.save_calls[1]["extra_pnginfo"], {})
        self.assertEqual(tuple(CoreNodesStub.save_calls[1]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(len(result["ui"]["images"]), 2)
```

- [ ] **Step 6: Run the save-mode test to verify it fails**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "save_image_save_mode_emits_separate_mask_file_without_lls_metadata" -v`
Expected: FAIL because only one save call is made and no `_mask` file is produced.


### Task 2: Lock The Removal Of The Old Preview Node With Failing Tests

**Files:**
- Modify: `tests/test_mask_create_registration.py`
- Modify: `tests/test_mask_create_docs.py`
- Delete: `tests/test_mask_preview_node.py`
- Test: `tests/test_mask_create_registration.py`
- Test: `tests/test_mask_create_docs.py`

- [ ] **Step 1: Replace the old registration assertions with removal assertions**

```python
    def test_plugin_does_not_register_mask_preview_node(self):
        plugin = load_plugin_package()

        self.assertNotIn("LLSSimpleMaskPreview", plugin.NODE_CLASS_MAPPINGS)
        self.assertNotIn("LLSSimpleMaskPreview", plugin.NODE_DISPLAY_NAME_MAPPINGS)
```

- [ ] **Step 2: Run the registration test to verify it fails**

Run: `python -m pytest tests/test_mask_create_registration.py -k "does_not_register_mask_preview_node" -v`
Expected: FAIL because the plugin still registers `LLSSimpleMaskPreview`.

- [ ] **Step 3: Replace README expectations so they describe the new workflow**

```python
    def test_readme_documents_mask_create_node(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("LLS Simple Mask Create", readme)
        self.assertNotIn("LLS Simple Mask Preview", readme)
        self.assertIn("LLS Save Image.mask", readme)
        self.assertIn("LLS Simple Mask Create.mask_image -> Preview Image", readme)
```

- [ ] **Step 4: Run the README test to verify it fails**

Run: `python -m pytest tests/test_mask_create_docs.py -k "readme_documents_mask_create_node" -v`
Expected: FAIL because README still references `LLS Simple Mask Preview`.

- [ ] **Step 5: Remove the obsolete runtime test file**

```bash
git rm tests/test_mask_preview_node.py
```

- [ ] **Step 6: Verify the deleted file is no longer tracked**

Run: `git status --short tests/test_mask_preview_node.py`
Expected: output contains `D tests/test_mask_preview_node.py`


### Task 3: Implement The Save Image Changes And Remove The Old Preview Node

**Files:**
- Modify: `image/nodes.py`
- Modify: `mask/nodes.py`
- Delete: `mask/mask_preview.py`
- Modify: `README.md`
- Test: `tests/test_loader_prompt_refactor.py`
- Test: `tests/test_mask_create_registration.py`
- Test: `tests/test_mask_create_docs.py`

- [ ] **Step 1: Add the minimal `mask` schema and conversion support to `LLSSaveImage`**

```python
from ..mask.mask_utils import mask_to_image

            "optional": {
                "mask": ("MASK",),
                "prompt_info": ("STRING", {"forceInput": True}),
```

```python
    def _merge_ui_results(self, primary: dict, secondary: dict | None) -> dict:
        if not secondary:
            return primary
        merged = dict(primary)
        merged_ui = dict(primary.get("ui", {}))
        merged_images = list(merged_ui.get("images", []))
        merged_images.extend(secondary.get("ui", {}).get("images", []))
        merged_ui["images"] = merged_images
        merged["ui"] = merged_ui
        return merged
```

- [ ] **Step 2: Implement preview-mode mask output with the smallest possible change**

```python
        if output_mode == "preview_only":
            previewer = comfy_core_nodes.PreviewImage()
            image_result = previewer.save_images(
                image,
                prompt=prompt,
                extra_pnginfo=None,
            )
            if mask is None:
                return image_result
            mask_result = previewer.save_images(
                mask_to_image(mask),
                prompt=prompt,
                extra_pnginfo=None,
            )
            return self._merge_ui_results(image_result, mask_result)
```

- [ ] **Step 3: Implement save-mode mask output without metadata leakage**

```python
        image_result = saver.save_images(
            image,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=merged_extra_pnginfo,
        )
        if mask is None:
            return image_result

        mask_result = saver.save_images(
            mask_to_image(mask),
            filename_prefix=f"{filename_prefix}_mask",
            prompt=prompt,
            extra_pnginfo={},
        )
        return self._merge_ui_results(image_result, mask_result)
```

- [ ] **Step 4: Remove `LLS Simple Mask Preview` from registration and source**

```python
from .mask_create import (
    NODE_CLASS_MAPPINGS as MASK_CREATE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_CREATE_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(MASK_CREATE_NODE_CLASS_MAPPINGS)
```

```bash
git rm mask/mask_preview.py
```

- [ ] **Step 5: Update README workflow examples to use `LLS Save Image.mask`**

```markdown
- `LLS Save Image`
- `Load Image.image -> LLS Save Image.image`
- `LLS Simple Mask Create.mask -> LLS Save Image.mask`
- `LLS Simple Mask Draw.mask -> LLS Save Image.mask`
```

- [ ] **Step 6: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_loader_prompt_refactor.py tests/test_mask_create_registration.py tests/test_mask_create_docs.py -v`
Expected: PASS with `0 failed`.

- [ ] **Step 7: Commit the implementation**

```bash
git add image/nodes.py mask/nodes.py README.md tests/test_loader_prompt_refactor.py tests/test_mask_create_registration.py tests/test_mask_create_docs.py
git add docs/superpowers/plans/2026-05-25-lls-save-image-mask-output.md
git commit -m "Add mask output support to LLS Save Image"
```
