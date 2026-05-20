# Qwen Minimal Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two minimal high-level Qwen nodes that hide model/clip/vae wiring and directly output `IMAGE` for text-to-image and official image edit workflows.

**Architecture:** Introduce a dedicated `qwen/` subpackage with three focused modules: discovery for compatible model/resource filtering, runtime for thin wrappers around official ComfyUI Qwen nodes, and user-facing nodes that expose the compressed UX. Keep Qwen isolated from the existing SD1.5/SDXL/FLUX simple-chain code and validate compatibility both in the widget choices and at runtime.

**Tech Stack:** Python, unittest, ComfyUI core nodes, `comfy_extras.nodes_qwen`, `comfy_extras.nodes_flux`, `comfy_extras.nodes_model_advanced`, `comfy_extras.nodes_cfg`

---

### Task 1: Add Qwen discovery helpers and red tests

**Files:**
- Create: `qwen/discovery.py`
- Create: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
class QwenFolderPathsStub:
    def __init__(self):
        self._files = {
            "diffusion_models": [
                "qwen_image_fp8_e4m3fn.safetensors",
                "qwen_image_2512_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2511_bf16.safetensors",
                "qwen_image_layered_bf16.safetensors",
                "flux1-dev.safetensors",
            ],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
        }

    def get_filename_list(self, category):
        return list(self._files.get(category, []))


def test_qwen_text_node_filters_only_text_models(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen.nodes import LLSQwenTextToImage

    with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
        choices = LLSQwenTextToImage.INPUT_TYPES()["required"]["model_name"][0]

    assert choices == [
        "qwen_image_2512_fp8_e4m3fn.safetensors",
        "qwen_image_fp8_e4m3fn.safetensors",
    ]


def test_qwen_edit_node_filters_only_edit_models(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen.nodes import LLSQwenImageEdit

    with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
        choices = LLSQwenImageEdit.INPUT_TYPES()["required"]["model_name"][0]

    assert choices == ["qwen_image_edit_2511_bf16.safetensors"]


def test_qwen_text_node_uses_placeholder_when_no_compatible_models_exist(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen.nodes import LLSQwenTextToImage

    stub = QwenFolderPathsStub()
    stub._files["diffusion_models"] = ["flux1-dev.safetensors", "sdxl.safetensors"]
    with mock.patch.object(qwen_discovery, "folder_paths", stub):
        choices = LLSQwenTextToImage.INPUT_TYPES()["required"]["model_name"][0]

    assert choices == ["(no qwen text-to-image models found)"]
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_filters_only_text_models tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_filters_only_edit_models tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_uses_placeholder_when_no_compatible_models_exist -v`
Expected: FAIL because the `qwen/` package and node classes do not exist yet.

- [ ] **Step 3: Write the minimal discovery implementation**

```python
TEXT_PLACEHOLDER = "(no qwen text-to-image models found)"
EDIT_PLACEHOLDER = "(no qwen image edit models found)"


def is_qwen_text_to_image_model(name: str) -> bool:
    lowered = name.lower()
    return (
        "qwen" in lowered
        and "image" in lowered
        and "edit" not in lowered
        and "layered" not in lowered
    )


def is_qwen_image_edit_model(name: str) -> bool:
    lowered = name.lower()
    return "qwen" in lowered and "image" in lowered and "edit" in lowered and "layered" not in lowered


def get_qwen_text_model_choices() -> list[str]:
    names = [name for name in _get_filename_list("diffusion_models") if is_qwen_text_to_image_model(name)]
    return sorted(dict.fromkeys(names)) or [TEXT_PLACEHOLDER]


def get_qwen_edit_model_choices() -> list[str]:
    names = [name for name in _get_filename_list("diffusion_models") if is_qwen_image_edit_model(name)]
    return sorted(dict.fromkeys(names)) or [EDIT_PLACEHOLDER]
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_filters_only_text_models tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_filters_only_edit_models tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_uses_placeholder_when_no_compatible_models_exist -v`
Expected: PASS with the correct model filtering and placeholder behavior.

- [ ] **Step 5: Commit the discovery slice**

```bash
git add qwen/discovery.py tests/test_qwen_nodes.py
git commit -m "feat: add qwen model discovery helpers"
```

### Task 2: Add runtime wrappers for official Qwen flows with red tests

**Files:**
- Create: `qwen/runtime.py`
- Modify: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
def test_qwen_text_runtime_rejects_incompatible_model_name(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen import runtime as qwen_runtime

    with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
        with self.assertRaisesRegex(RuntimeError, "not compatible with LLSQwenTextToImage"):
            qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_edit_2511_bf16.safetensors",
                prompt="a cat",
                width=1024,
                height=1024,
                steps=20,
                seed=1,
                batch_size=1,
            )


def test_qwen_text_runtime_raises_when_qwen_text_encoder_is_missing(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen import runtime as qwen_runtime

    stub = QwenFolderPathsStub()
    stub._files["text_encoders"] = []
    with mock.patch.object(qwen_discovery, "folder_paths", stub):
        with self.assertRaisesRegex(RuntimeError, "Missing Qwen text encoder"):
            qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                width=1024,
                height=1024,
                steps=20,
                seed=1,
                batch_size=1,
            )


def test_qwen_text_runtime_executes_minimal_official_pipeline(self):
    load_plugin_package()
    from lls_node_test_qwen.qwen import discovery as qwen_discovery
    from lls_node_test_qwen.qwen import runtime as qwen_runtime

    with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()), \
         mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub()), \
         mock.patch.object(qwen_runtime, "nodes_sd3", SD3NodesStub()), \
         mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub()):
        image = qwen_runtime.run_qwen_text_to_image(
            model_name="qwen_image_fp8_e4m3fn.safetensors",
            prompt="a cat",
            width=1024,
            height=1024,
            steps=20,
            seed=99,
            batch_size=2,
        )

    self.assertEqual(image, "IMAGE::decoded")
```

- [ ] **Step 2: Run the focused runtime tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_rejects_incompatible_model_name tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_raises_when_qwen_text_encoder_is_missing tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_executes_minimal_official_pipeline -v`
Expected: FAIL because the Qwen runtime module and wrappers do not exist yet.

- [ ] **Step 3: Write the minimal runtime implementation**

```python
def run_qwen_text_to_image(model_name, prompt, width, height, steps, seed, batch_size):
    validate_qwen_text_model_name(model_name)
    clip_name = resolve_qwen_text_encoder_name()
    vae_name = resolve_qwen_vae_name()
    if clip_name is None:
        raise RuntimeError("[LLS] Missing Qwen text encoder. Place a compatible Qwen VL text encoder in ComfyUI/models/text_encoders/.")
    if vae_name is None:
        raise RuntimeError("[LLS] Missing Qwen VAE. Place qwen_image_vae in ComfyUI/models/vae/.")

    model = comfy_core_nodes.UNETLoader().load_unet(model_name, "default")[0]
    clip = comfy_core_nodes.CLIPLoader().load_clip(clip_name, type="qwen_image", device="default")[0]
    vae = comfy_core_nodes.VAELoader().load_vae(vae_name)[0]

    model = nodes_model_advanced.ModelSamplingAuraFlow().patch_aura(model, 3.1)[0]
    positive = comfy_core_nodes.CLIPTextEncode().encode(clip, prompt)[0]
    negative = comfy_core_nodes.CLIPTextEncode().encode(clip, "")[0]
    latent = nodes_sd3.EmptySD3LatentImage().generate(width, height, batch_size)[0]
    sampled = comfy_core_nodes.KSampler().sample(
        model, seed, steps, 4.0, "euler", "simple", positive, negative, latent, denoise=1.0
    )[0]
    return comfy_core_nodes.VAEDecode().decode(vae, sampled)[0]
```

- [ ] **Step 4: Extend the runtime implementation for official image edit**

```python
def run_qwen_image_edit(model_name, image, prompt, steps, seed):
    validate_qwen_edit_model_name(model_name)
    clip_name = resolve_qwen_text_encoder_name()
    vae_name = resolve_qwen_vae_name()
    ...
    scaled = nodes_flux.FluxKontextImageScale.execute(image).result[0]
    positive = nodes_qwen.TextEncodeQwenImageEditPlus.execute(clip, prompt, vae=vae, image1=scaled).result[0]
    negative = nodes_qwen.TextEncodeQwenImageEditPlus.execute(clip, "", vae=vae, image1=scaled).result[0]
    positive = nodes_flux.FluxKontextMultiReferenceLatentMethod.execute(positive, "index_timestep_zero").result[0]
    negative = nodes_flux.FluxKontextMultiReferenceLatentMethod.execute(negative, "index_timestep_zero").result[0]
    latent = comfy_core_nodes.VAEEncode().encode(vae, scaled)[0]
    model = nodes_model_advanced.ModelSamplingAuraFlow().patch_aura(model, 3.1)[0]
    model = nodes_cfg.CFGNorm.execute(model, 1.0).result[0]
    sampled = comfy_core_nodes.KSampler().sample(
        model, seed, steps, 4.0, "euler", "simple", positive, negative, latent, denoise=1.0
    )[0]
    return comfy_core_nodes.VAEDecode().decode(vae, sampled)[0]
```

- [ ] **Step 5: Run the focused runtime tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_rejects_incompatible_model_name tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_raises_when_qwen_text_encoder_is_missing tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_executes_minimal_official_pipeline tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_executes_minimal_official_pipeline -v`
Expected: PASS with clear validation failures and both minimal pipelines returning decoded images through the official ComfyUI node wrappers.

- [ ] **Step 6: Commit the runtime slice**

```bash
git add qwen/runtime.py tests/test_qwen_nodes.py
git commit -m "feat: add qwen runtime wrappers"
```

### Task 3: Add user-facing nodes and register the new subpackage

**Files:**
- Create: `qwen/__init__.py`
- Create: `qwen/nodes.py`
- Modify: `__init__.py`
- Modify: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing node registration and schema tests**

```python
def test_plugin_registers_qwen_nodes(self):
    plugin = load_plugin_package()
    self.assertIn("LLSQwenTextToImage", plugin.NODE_CLASS_MAPPINGS)
    self.assertIn("LLSQwenImageEdit", plugin.NODE_CLASS_MAPPINGS)


def test_qwen_text_node_returns_image_only(self):
    plugin = load_plugin_package()
    node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenTextToImage"]
    required = node_cls.INPUT_TYPES()["required"]

    self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
    self.assertEqual(
        tuple(required.keys()),
        ("model_name", "prompt", "width", "height", "steps", "seed", "batch_size"),
    )


def test_qwen_edit_node_returns_image_only(self):
    plugin = load_plugin_package()
    node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenImageEdit"]
    required = node_cls.INPUT_TYPES()["required"]

    self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
    self.assertEqual(tuple(required.keys()), ("model_name", "image", "prompt", "steps", "seed"))
```

- [ ] **Step 2: Run the focused node tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_plugin_registers_qwen_nodes tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_returns_image_only tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_returns_image_only -v`
Expected: FAIL because the Qwen subpackage is not yet registered and the node classes do not exist.

- [ ] **Step 3: Write the minimal node layer**

```python
class LLSQwenTextToImage:
    CATEGORY = "LLS/Qwen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_qwen_text_model_choices(),),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    def generate(self, model_name, prompt, width, height, steps, seed, batch_size):
        return (run_qwen_text_to_image(model_name, prompt, width, height, steps, seed, batch_size),)
```

- [ ] **Step 4: Register the subpackage**

```python
_SUBPACKAGES: list[str] = [
    "model_loader",
    "conditioning",
    "sampling",
    "qwen",
    "latent",
    ...
]
```

- [ ] **Step 5: Run the node tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_plugin_registers_qwen_nodes tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_returns_image_only tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_returns_image_only tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_executes_runtime tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_executes_runtime -v`
Expected: PASS with both nodes registered, exposing only `IMAGE`, and delegating to the runtime layer.

- [ ] **Step 6: Commit the node layer**

```bash
git add __init__.py qwen/__init__.py qwen/nodes.py tests/test_qwen_nodes.py
git commit -m "feat: add minimal qwen image nodes"
```

### Task 4: Run the full regression suite for the plugin

**Files:**
- Modify: `tests/test_qwen_nodes.py` (only if a verification run exposes a mismatch)
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Run the dedicated Qwen test file**

Run: `python3 -m unittest tests.test_qwen_nodes -v`
Expected: PASS with zero failures for discovery, validation, runtime, registration, and execution-path coverage.

- [ ] **Step 2: Run the full existing test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS with zero regressions in the existing loader/prompt/latent/upscale suites.

- [ ] **Step 3: Review the changed files before closeout**

```bash
git status --short
```

Expected:
- `M __init__.py`
- `A qwen/__init__.py`
- `A qwen/discovery.py`
- `A qwen/runtime.py`
- `A qwen/nodes.py`
- `A tests/test_qwen_nodes.py`
- `A docs/superpowers/plans/2026-05-20-qwen-minimal-nodes.md`

- [ ] **Step 4: Commit the verified final state**

```bash
git add __init__.py qwen tests/test_qwen_nodes.py docs/superpowers/plans/2026-05-20-qwen-minimal-nodes.md
git commit -m "test: verify minimal qwen nodes"
```
