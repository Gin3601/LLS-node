# Empty Latent Unified Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `LLSSimpleEmptyLatent` so it supports both txt2img and img2img without breaking existing txt2img workflows.

**Architecture:** Keep `LLSSimpleEmptyLatent` as the single latent-entry node. When no `image` is connected it preserves the current zero-latent path; when `image` is connected it routes through the existing VAE encode logic and returns img2img-flavored latent metadata. Reuse the current image resize and VAE encode semantics instead of introducing a second implementation.

**Tech Stack:** Python, unittest, ComfyUI node definitions, existing fake tensor test stubs

---

### Task 1: Add failing tests for the unified node behavior

**Files:**
- Modify: `tests/test_loader_prompt_refactor.py`
- Test: `tests/test_loader_prompt_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_empty_latent_schema_accepts_optional_img2img_inputs(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleEmptyLatent"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(required["resize_mode"][0], ["keep_aspect", "crop_center", "stretch", "none"])
        self.assertEqual(optional["image"][0], "IMAGE")
        self.assertEqual(optional["vae"][0], "VAE")
        self.assertEqual(optional["model"][0], "MODEL")

    def test_empty_latent_can_encode_image_using_family_default_size(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes
        from lls_node_test_refactor.image import nodes as image_nodes

        comfy_utils = FakeComfyUtils()
        model = TaggedValue("MODEL::SDXL")
        model._lls_family = "SDXL"
        vae = FakeVAE()

        with mock.patch.object(image_nodes, "comfy_utils", comfy_utils):
            node = latent_nodes.LLSSimpleEmptyLatent()
            latent, width, height, latent_info = node.create_empty_latent(
                "Family Default",
                512,
                512,
                3,
                model_family="Auto",
                resize_mode="keep_aspect",
                model=model,
                image=FakeTensor((1, 768, 1536, 3)),
                vae=vae,
            )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (1024, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 128))
        self.assertEqual(payload["task_mode"], "img2img")
        self.assertEqual(payload["latent_source"], "image_encode")
        self.assertEqual(payload["size_preset"], "Family Default")

    def test_empty_latent_requires_vae_when_image_is_connected(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()
        with self.assertRaisesRegex(RuntimeError, "Missing VAE"):
            node.create_empty_latent(
                "Custom",
                640,
                640,
                1,
                model_family="SD1.5",
                resize_mode="stretch",
                image=FakeTensor((1, 512, 512, 3)),
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "empty_latent" -v`
Expected: FAIL because `LLSSimpleEmptyLatent` does not yet expose `resize_mode` / `image` / `vae` or route image input to img2img.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_loader_prompt_refactor.py
git commit -m "test: cover empty latent unified entry behavior"
```

### Task 2: Implement unified txt2img/img2img behavior in the node

**Files:**
- Modify: `latent/nodes.py`
- Test: `tests/test_loader_prompt_refactor.py`

- [ ] **Step 1: Implement the minimal code**

```python
from ..image.nodes import LLSSimpleVAEEncode
from ..utils.model_info import parse_jsonish_info

class LLSSimpleEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_preset": (_SIZE_PRESETS, {"default": SIZE_PRESET_AUTO}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "resize_mode": (["keep_aspect", "crop_center", "stretch", "none"], {"default": "keep_aspect"}),
                "model_family": (MODEL_FAMILY_CHOICES, {"default": "Auto"}),
            },
            "optional": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    def create_empty_latent(..., resize_mode: str = "keep_aspect", model=None, image=None, vae=None):
        family = resolve_model_family(model_family, model=model)
        defaults = get_family_defaults(family)
        ...
        if image is not None:
            if vae is None:
                raise RuntimeError("[LLS] Missing VAE. Connect the Loader VAE output or choose an external VAE in the loader.")
            encode_node = LLSSimpleVAEEncode()
            size_source = "model_recommended" if size_preset == SIZE_PRESET_AUTO else "custom"
            latent_payload, final_width, final_height, latent_info = encode_node.encode(...)
            latent_meta = parse_jsonish_info(latent_info)
            latent_meta["size_preset"] = size_preset
            return latent_payload, final_width, final_height, info_to_json(latent_meta)
        ...
```

- [ ] **Step 2: Run the focused tests to verify they pass**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "empty_latent" -v`
Expected: PASS for the new schema, txt2img compatibility, img2img routing, and missing-VAE error coverage.

- [ ] **Step 3: Refactor only if needed**

```python
# Keep the existing size-preset resolution in one place so both txt2img and img2img
# use the same width/height decisions before branching.
```

- [ ] **Step 4: Commit the implementation**

```bash
git add latent/nodes.py tests/test_loader_prompt_refactor.py
git commit -m "feat: unify empty latent txt2img and img2img entry"
```

### Task 3: Run regression verification for the affected workflow

**Files:**
- Modify: `tests/test_loader_prompt_refactor.py` (only if the focused run exposes a mismatch)
- Test: `tests/test_loader_prompt_refactor.py`

- [ ] **Step 1: Run the broader regression slice**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -v`
Expected: PASS with zero failures in the loader/prompt refactor regression suite.

- [ ] **Step 2: Re-run the exact img2img/txt2img checks if any regression appears**

Run: `python -m pytest tests/test_loader_prompt_refactor.py -k "empty_latent or vae_encode or ksampler" -v`
Expected: PASS, confirming latent source and task-mode inference still match downstream sampler expectations.

- [ ] **Step 3: Commit the verified final state**

```bash
git add latent/nodes.py tests/test_loader_prompt_refactor.py
git commit -m "test: verify unified empty latent workflow"
```
