# Qwen Advanced Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `LLSQwenTextToImage` and `LLSQwenImageEdit` with official/common advanced controls, an optional standard `MODEL` input compatible with `LoraLoaderModelOnly`, optional turbo/lightning LoRA support, and full regression coverage while preserving the compressed single-node UX.

**Architecture:** Keep the current dedicated `qwen/` package and expand it in place. `qwen/discovery.py` will own turbo LoRA filtering, auto-resolution, and turbo presets; `qwen/runtime.py` will own advanced parameter passthrough plus internal-vs-external model selection; `qwen/nodes.py` will expose the expanded Qwen node schemas with optional `MODEL` inputs while the main generation nodes still return only `IMAGE`.

**Tech Stack:** Python, unittest, ComfyUI core nodes, `comfy_extras.nodes_qwen`, `comfy_extras.nodes_flux`, `comfy_extras.nodes_model_advanced`, `comfy_extras.nodes_cfg`

---

## File Structure

- Modify: `qwen/discovery.py`
  - Keep turbo/lightning LoRA discovery, `(auto)` handling, compatible LoRA validation, and turbo preset lookup; remove now-unneeded custom LoRA-stack helpers.
- Modify: `qwen/runtime.py`
  - Add advanced parameter passthrough, external `MODEL` preference when connected, turbo preset overrides, and image-edit multi-reference handling.
- Modify: `qwen/nodes.py`
  - Expand the two public node schemas with advanced inputs, add optional `MODEL` inputs, and remove the custom `LLSQwenLoRAStack` helper while preserving hidden resource loading and `IMAGE`-only outputs.
- Modify: `tests/test_qwen_nodes.py`
  - Replace custom LoRA-stack tests with schema, external `MODEL` passthrough, turbo, and failure-path tests with small deterministic stubs.

## Plan Amendment

This plan now assumes two changes beyond the original draft:

- both public Qwen generation nodes accept an optional standard `MODEL` input
- `qwen_image_edit_2511_*` turbo/lightning support is allowed when a matching 2511 LoRA exists locally

The intended model selection order is:

- external path: use connected `MODEL`, then optionally apply turbo/lightning LoRA
- internal path: load model from `model_name`, then optionally apply turbo/lightning LoRA

### Task 1: Expand schemas and turbo LoRA discovery

**Files:**
- Modify: `qwen/discovery.py`
- Modify: `qwen/nodes.py`
- Modify: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
class QwenFolderPathsStub:
    def __init__(self):
        self._files = {
            "diffusion_models": [
                "qwen_image_fp8_e4m3fn.safetensors",
                "qwen_image_2512_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2511_bf16.safetensors",
                "qwen_image_layered_bf16.safetensors",
                "flux1-dev.safetensors",
            ],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
            "loras": [
                "Qwen-Image-Lightning-8steps-V1.0.safetensors",
                "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
                "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                "flux-dev-style.safetensors",
            ],
        }

    def get_filename_list(self, category):
        return list(self._files.get(category, []))


class TestQwenNodes(unittest.TestCase):
    def test_qwen_text_node_schema_exposes_advanced_inputs(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenTextToImage"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(
            tuple(required.keys()),
            (
                "model_name",
                "prompt",
                "width",
                "height",
                "steps",
                "seed",
                "batch_size",
                "negative_prompt",
                "cfg",
                "sampler_name",
                "scheduler",
                "shift",
                "enable_turbo_mode",
                "turbo_lora_name",
                "turbo_strength",
            ),
        )
        self.assertEqual(required["negative_prompt"][1]["default"], "")
        self.assertEqual(required["cfg"][1]["default"], 4.0)
        self.assertIn("euler", required["sampler_name"][0])
        self.assertIn("simple", required["scheduler"][0])
        self.assertEqual(required["shift"][1]["default"], 3.1)
        self.assertEqual(required["enable_turbo_mode"][1]["default"], False)

    def test_qwen_edit_node_schema_exposes_advanced_inputs_and_optional_images(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenImageEdit"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(
            tuple(required.keys()),
            (
                "model_name",
                "image",
                "prompt",
                "steps",
                "seed",
                "negative_prompt",
                "cfg",
                "sampler_name",
                "scheduler",
                "shift",
                "cfg_norm_strength",
                "reference_latents_method",
                "enable_turbo_mode",
                "turbo_lora_name",
                "turbo_strength",
            ),
        )
        self.assertEqual(optional["image2"][0], "IMAGE")
        self.assertEqual(optional["image3"][0], "IMAGE")
        self.assertEqual(
            required["reference_latents_method"][0],
            ["offset", "index", "uxo/uno", "index_timestep_zero"],
        )

    def test_qwen_text_node_turbo_lora_choices_filter_only_text_loras(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenTextToImage.INPUT_TYPES()["required"]["turbo_lora_name"][0]

        self.assertEqual(
            choices,
            [
                "(auto)",
                "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
                "Qwen-Image-Lightning-8steps-V1.0.safetensors",
            ],
        )

    def test_qwen_edit_node_turbo_lora_choices_filter_only_edit_loras(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenImageEdit.INPUT_TYPES()["required"]["turbo_lora_name"][0]

        self.assertEqual(
            choices,
            [
                "(auto)",
                "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            ],
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_schema_exposes_advanced_inputs tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_schema_exposes_advanced_inputs_and_optional_images tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_turbo_lora_choices_filter_only_text_loras tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_turbo_lora_choices_filter_only_edit_loras -v`
Expected: FAIL because the node schemas do not yet expose advanced inputs and discovery does not yet expose turbo LoRA choices.

- [ ] **Step 3: Write the minimal discovery and schema implementation**

```python
# qwen/discovery.py
AUTO_TURBO_LORA_CHOICE = "(auto)"
NO_TEXT_TURBO_LORA_PLACEHOLDER = "(no qwen text turbo loras found)"
NO_EDIT_TURBO_LORA_PLACEHOLDER = "(no qwen edit turbo loras found)"
REFERENCE_LATENTS_METHOD_CHOICES = ["offset", "index", "uxo/uno", "index_timestep_zero"]


def is_qwen_text_turbo_lora(name: str | None) -> bool:
    lowered = (name or "").lower()
    return "qwen-image" in lowered and "lightning" in lowered and "edit" not in lowered


def is_qwen_edit_turbo_lora(name: str | None) -> bool:
    lowered = (name or "").lower()
    return "qwen-image-edit" in lowered and "lightning" in lowered


def get_qwen_text_turbo_lora_choices() -> list[str]:
    names = _sorted_unique(
        [name for name in _get_filename_list("loras") if is_qwen_text_turbo_lora(name)]
    )
    if not names:
        return [AUTO_TURBO_LORA_CHOICE, NO_TEXT_TURBO_LORA_PLACEHOLDER]
    return [AUTO_TURBO_LORA_CHOICE] + names


def get_qwen_edit_turbo_lora_choices() -> list[str]:
    names = _sorted_unique(
        [name for name in _get_filename_list("loras") if is_qwen_edit_turbo_lora(name)]
    )
    if not names:
        return [AUTO_TURBO_LORA_CHOICE, NO_EDIT_TURBO_LORA_PLACEHOLDER]
    return [AUTO_TURBO_LORA_CHOICE] + names
```

```python
# qwen/nodes.py
from ..sampling.nodes import _get_samplers, _get_schedulers


class LLSQwenTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (discovery.get_qwen_text_model_choices(),),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 16, "max": _MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": _MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "advanced": True}),
                "sampler_name": (_get_samplers(), {"default": "euler", "advanced": True}),
                "scheduler": (_get_schedulers(), {"default": "simple", "advanced": True}),
                "shift": ("FLOAT", {"default": 3.1, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "enable_turbo_mode": ("BOOLEAN", {"default": False, "advanced": True}),
                "turbo_lora_name": (discovery.get_qwen_text_turbo_lora_choices(), {"default": discovery.AUTO_TURBO_LORA_CHOICE, "advanced": True}),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "advanced": True}),
            }
        }
```

```python
class LLSQwenImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (discovery.get_qwen_edit_model_choices(),),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "advanced": True}),
                "sampler_name": (_get_samplers(), {"default": "euler", "advanced": True}),
                "scheduler": (_get_schedulers(), {"default": "simple", "advanced": True}),
                "shift": ("FLOAT", {"default": 3.1, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "cfg_norm_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "reference_latents_method": (discovery.REFERENCE_LATENTS_METHOD_CHOICES, {"default": "index_timestep_zero", "advanced": True}),
                "enable_turbo_mode": ("BOOLEAN", {"default": False, "advanced": True}),
                "turbo_lora_name": (discovery.get_qwen_edit_turbo_lora_choices(), {"default": discovery.AUTO_TURBO_LORA_CHOICE, "advanced": True}),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "advanced": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_schema_exposes_advanced_inputs tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_schema_exposes_advanced_inputs_and_optional_images tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_node_turbo_lora_choices_filter_only_text_loras tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_node_turbo_lora_choices_filter_only_edit_loras -v`
Expected: PASS with the expanded schemas and correctly filtered turbo LoRA choices.

- [ ] **Step 5: Commit**

```bash
git add qwen/discovery.py qwen/nodes.py tests/test_qwen_nodes.py
git commit -m "feat: expand qwen node schemas"
```

### Task 2: Add text-to-image advanced runtime controls and turbo support

**Files:**
- Modify: `qwen/discovery.py`
- Modify: `qwen/runtime.py`
- Modify: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
class CoreQwenNodesStub:
    last_lora_call = None

    @classmethod
    def reset(cls):
        cls.last_lora_call = None
        cls.last_unet_call = None
        cls.last_clip_call = None
        cls.last_text_encode_calls = []
        cls.last_ksampler_call = None
        cls.last_vae_encode_call = None
        cls.last_vae_decode_call = None
        cls.last_vae_load_call = None

    class LoraLoaderModelOnly:
        def load_lora_model_only(self, model, lora_name, strength_model):
            CoreQwenNodesStub.last_lora_call = {
                "model": model,
                "lora_name": lora_name,
                "strength_model": strength_model,
            }
            return (f"LORA::{model}::{lora_name}::{strength_model}",)


class TestQwenNodes(unittest.TestCase):
    def test_qwen_text_runtime_passes_negative_prompt_and_sampling_controls(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_sd3", SD3NodesStub), \
             mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub):
            qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                negative_prompt="low quality",
                width=1024,
                height=1024,
                steps=33,
                seed=99,
                batch_size=2,
                cfg=7.5,
                sampler_name="heun",
                scheduler="normal",
                shift=2.25,
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
            )

        self.assertEqual(CoreQwenNodesStub.last_text_encode_calls[1]["text"], "low quality")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 7.5)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["sampler_name"], "heun")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["scheduler"], "normal")
        self.assertEqual(ModelAdvancedStub.last_call["shift"], 2.25)

    def test_qwen_text_runtime_enables_turbo_lora_and_uses_turbo_preset(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_sd3", SD3NodesStub), \
             mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub):
            qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                negative_prompt="",
                width=1024,
                height=1024,
                steps=50,
                seed=99,
                batch_size=1,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                enable_turbo_mode=True,
                turbo_lora_name="(auto)",
                turbo_strength=0.75,
            )

        self.assertEqual(
            CoreQwenNodesStub.last_lora_call["lora_name"],
            "Qwen-Image-Lightning-8steps-V1.0.safetensors",
        )
        self.assertEqual(CoreQwenNodesStub.last_lora_call["strength_model"], 0.75)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["steps"], 8)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 1.0)

    def test_qwen_text_runtime_rejects_incompatible_manual_turbo_lora(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "Turbo LoRA .* is not compatible"):
                qwen_runtime.run_qwen_text_to_image(
                    model_name="qwen_image_fp8_e4m3fn.safetensors",
                    prompt="a cat",
                    negative_prompt="",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    enable_turbo_mode=True,
                    turbo_lora_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                    turbo_strength=1.0,
                )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_passes_negative_prompt_and_sampling_controls tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_enables_turbo_lora_and_uses_turbo_preset tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_rejects_incompatible_manual_turbo_lora -v`
Expected: FAIL because the runtime still hardcodes empty negative prompt, fixed `cfg`/sampler/scheduler/shift, and has no turbo LoRA logic.

- [ ] **Step 3: Write the minimal discovery and runtime implementation**

```python
# qwen/discovery.py
_QWEN_TURBO_PROFILES = {
    "qwen_image_fp8_e4m3fn.safetensors": {"steps": 8, "cfg": 1.0},
    "qwen_image_2512_fp8_e4m3fn.safetensors": {"steps": 4, "cfg": 1.0},
    "qwen_image_edit_2509_fp8_e4m3fn.safetensors": {"steps": 4, "cfg": 1.0},
}


def get_qwen_turbo_profile(model_name: str) -> dict[str, float] | None:
    return _QWEN_TURBO_PROFILES.get(model_name)


def resolve_qwen_text_turbo_lora(model_name: str, requested_name: str) -> str:
    available = [name for name in _get_filename_list("loras") if is_qwen_text_turbo_lora(name)]
    if "2512" in model_name.lower():
        candidates = [name for name in available if "2512" in name.lower()]
    else:
        candidates = [name for name in available if "2512" not in name.lower()]

    if requested_name == AUTO_TURBO_LORA_CHOICE:
        if candidates:
            return _sorted_unique(candidates)[0]
        raise RuntimeError(f"[LLS] No compatible turbo/lightning LoRA exists for '{model_name}'.")

    if requested_name not in available or requested_name not in candidates:
        raise RuntimeError(
            f"[LLS] Turbo LoRA '{requested_name}' is not compatible with text-to-image model '{model_name}'."
        )
    return requested_name
```

```python
# qwen/runtime.py
def _patch_qwen_model_sampling(model, shift: float):
    patcher_cls = _require_class(nodes_model_advanced, "ModelSamplingAuraFlow", _MODEL_ADVANCED_ERR)
    return patcher_cls().patch_aura(model, float(shift))[0]


def _sample_qwen(model, positive, negative, latent, steps: int, seed: int, cfg: float, sampler_name: str, scheduler: str):
    sampler_cls = _require_class(comfy_core_nodes, "KSampler", _CORE_NODES_ERR)
    return sampler_cls().sample(
        model,
        int(seed),
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]


def _load_model_only_lora(model, lora_name: str, strength_model: float):
    lora_loader_cls = _require_class(comfy_core_nodes, "LoraLoaderModelOnly", _CORE_NODES_ERR)
    return lora_loader_cls().load_lora_model_only(model, lora_name, float(strength_model))[0]


def _apply_text_turbo(model, model_name: str, steps: int, cfg: float, enable_turbo_mode: bool, turbo_lora_name: str, turbo_strength: float):
    if not enable_turbo_mode:
        return model, int(steps), float(cfg)

    resolved_lora = discovery.resolve_qwen_text_turbo_lora(model_name, turbo_lora_name)
    profile = discovery.get_qwen_turbo_profile(model_name)
    if profile is None:
        raise RuntimeError(f"[LLS] No supported turbo preset exists for '{model_name}'.")

    model = _load_model_only_lora(model, resolved_lora, turbo_strength)
    return model, int(profile["steps"]), float(profile["cfg"])


def run_qwen_text_to_image(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    batch_size: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    shift: float,
    enable_turbo_mode: bool,
    turbo_lora_name: str,
    turbo_strength: float,
):
    model_name, clip_name, vae_name = _resolve_qwen_resources(
        model_name,
        discovery.validate_qwen_text_model_name,
    )
    model = _load_qwen_model(model_name)
    model, effective_steps, effective_cfg = _apply_text_turbo(
        model,
        model_name,
        steps,
        cfg,
        enable_turbo_mode,
        turbo_lora_name,
        turbo_strength,
    )
    model = _patch_qwen_model_sampling(model, shift)
    clip = _load_qwen_clip(clip_name)
    vae = _load_qwen_vae(vae_name)
    positive = _encode_clip_text(clip, prompt)
    negative = _encode_clip_text(clip, negative_prompt)
    latent = _create_empty_qwen_latent(int(width), int(height), int(batch_size))
    sampled = _sample_qwen(model, positive, negative, latent, effective_steps, int(seed), effective_cfg, sampler_name, scheduler)
    return _decode_qwen_latent(vae, sampled)
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_passes_negative_prompt_and_sampling_controls tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_enables_turbo_lora_and_uses_turbo_preset tests.test_qwen_nodes.TestQwenNodes.test_qwen_text_runtime_rejects_incompatible_manual_turbo_lora -v`
Expected: PASS with advanced text controls routed through and turbo mode correctly applying the text turbo LoRA and preset.

- [ ] **Step 5: Commit**

```bash
git add qwen/discovery.py qwen/runtime.py tests/test_qwen_nodes.py
git commit -m "feat: add qwen text advanced controls"
```

### Task 3: Add image-edit advanced runtime controls and turbo support

**Files:**
- Modify: `qwen/discovery.py`
- Modify: `qwen/runtime.py`
- Modify: `qwen/nodes.py`
- Modify: `tests/test_qwen_nodes.py`
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
class TestQwenNodes(unittest.TestCase):
    def test_qwen_edit_runtime_passes_optional_images_negative_prompt_and_edit_controls(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub), \
             mock.patch.object(qwen_runtime, "nodes_qwen", QwenExtraNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_flux", FluxNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_cfg", CFGNodesStub):
            qwen_runtime.run_qwen_image_edit(
                model_name="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                image="IMAGE::1",
                image2="IMAGE::2",
                image3="IMAGE::3",
                prompt="turn the cat blue",
                negative_prompt="bad anatomy",
                steps=30,
                seed=123,
                cfg=2.5,
                sampler_name="heun",
                scheduler="normal",
                shift=3.0,
                cfg_norm_strength=0.75,
                reference_latents_method="index",
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
            )

        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image1"], "SCALED::IMAGE::1")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image2"], "IMAGE::2")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image3"], "IMAGE::3")
        self.assertEqual(QwenExtraNodesStub.last_calls[1]["prompt"], "bad anatomy")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 2.5)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["sampler_name"], "heun")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["scheduler"], "normal")
        self.assertEqual(ModelAdvancedStub.last_call["shift"], 3.0)
        self.assertEqual(CFGNodesStub.last_call["strength"], 0.75)
        self.assertEqual(FluxNodesStub.last_reference_calls[0]["reference_latents_method"], "index")

    def test_qwen_edit_runtime_enables_turbo_lora_and_uses_turbo_preset(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "folder_paths", stub), \
             mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub), \
             mock.patch.object(qwen_runtime, "nodes_qwen", QwenExtraNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_flux", FluxNodesStub), \
             mock.patch.object(qwen_runtime, "nodes_cfg", CFGNodesStub):
            qwen_runtime.run_qwen_image_edit(
                model_name="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                image="IMAGE::1",
                image2=None,
                image3=None,
                prompt="turn the cat blue",
                negative_prompt="",
                steps=40,
                seed=123,
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.0,
                cfg_norm_strength=1.0,
                reference_latents_method="index_timestep_zero",
                enable_turbo_mode=True,
                turbo_lora_name="(auto)",
                turbo_strength=0.8,
            )

        self.assertEqual(
            CoreQwenNodesStub.last_lora_call["lora_name"],
            "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        )
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["steps"], 4)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 1.0)

    def test_qwen_edit_runtime_fails_when_2511_turbo_is_requested(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "No compatible turbo/lightning LoRA exists"):
                qwen_runtime.run_qwen_image_edit(
                    model_name="qwen_image_edit_2511_bf16.safetensors",
                    image="IMAGE::1",
                    image2=None,
                    image3=None,
                    prompt="turn the cat blue",
                    negative_prompt="",
                    steps=20,
                    seed=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    cfg_norm_strength=1.0,
                    reference_latents_method="index_timestep_zero",
                    enable_turbo_mode=True,
                    turbo_lora_name="(auto)",
                    turbo_strength=1.0,
                )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_passes_optional_images_negative_prompt_and_edit_controls tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_enables_turbo_lora_and_uses_turbo_preset tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_fails_when_2511_turbo_is_requested -v`
Expected: FAIL because the image-edit runtime does not yet pass optional images or advanced controls through and has no edit turbo logic.

- [ ] **Step 3: Write the minimal edit runtime implementation**

```python
# qwen/discovery.py
def resolve_qwen_edit_turbo_lora(model_name: str, requested_name: str) -> str:
    available = [name for name in _get_filename_list("loras") if is_qwen_edit_turbo_lora(name)]
    if "2509" in model_name.lower():
        candidates = [name for name in available if "2509" in name.lower()]
    else:
        candidates = []

    if requested_name == AUTO_TURBO_LORA_CHOICE:
        if candidates:
            return _sorted_unique(candidates)[0]
        raise RuntimeError(f"[LLS] No compatible turbo/lightning LoRA exists for '{model_name}'.")

    if requested_name not in available or requested_name not in candidates:
        raise RuntimeError(
            f"[LLS] Turbo LoRA '{requested_name}' is not compatible with image-edit model '{model_name}'."
        )
    return requested_name
```

```python
# qwen/runtime.py
def _patch_qwen_model_sampling(model, shift: float):
    patcher_cls = _require_class(nodes_model_advanced, "ModelSamplingAuraFlow", _MODEL_ADVANCED_ERR)
    return patcher_cls().patch_aura(model, float(shift))[0]


def _apply_cfg_norm(model, strength: float):
    cfg_cls = _require_class(nodes_cfg, "CFGNorm", _CFG_ERR)
    return _unwrap_first(cfg_cls.execute(model, float(strength)))


def _apply_reference_latents_method(conditioning, reference_latents_method: str):
    method_cls = _require_class(nodes_flux, "FluxKontextMultiReferenceLatentMethod", _FLUX_ERR)
    return _unwrap_first(method_cls.execute(conditioning, reference_latents_method))


def _encode_qwen_edit_conditioning(clip, prompt: str, vae, image1, image2=None, image3=None):
    encoder_cls = _require_class(nodes_qwen, "TextEncodeQwenImageEditPlus", _QWEN_ERR)
    return _unwrap_first(
        encoder_cls.execute(clip, prompt, vae=vae, image1=image1, image2=image2, image3=image3)
    )


def _apply_edit_turbo(model, model_name: str, steps: int, cfg: float, enable_turbo_mode: bool, turbo_lora_name: str, turbo_strength: float):
    if not enable_turbo_mode:
        return model, int(steps), float(cfg)

    resolved_lora = discovery.resolve_qwen_edit_turbo_lora(model_name, turbo_lora_name)
    profile = discovery.get_qwen_turbo_profile(model_name)
    if profile is None:
        raise RuntimeError(f"[LLS] No supported turbo preset exists for '{model_name}'.")

    model = _load_model_only_lora(model, resolved_lora, turbo_strength)
    return model, int(profile["steps"]), float(profile["cfg"])


def run_qwen_image_edit(
    model_name: str,
    image,
    image2,
    image3,
    prompt: str,
    negative_prompt: str,
    steps: int,
    seed: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    shift: float,
    cfg_norm_strength: float,
    reference_latents_method: str,
    enable_turbo_mode: bool,
    turbo_lora_name: str,
    turbo_strength: float,
):
    if image is None:
        raise RuntimeError("[LLS] Missing source image. Connect an IMAGE input.")

    model_name, clip_name, vae_name = _resolve_qwen_resources(
        model_name,
        discovery.validate_qwen_edit_model_name,
    )
    model = _load_qwen_model(model_name)
    model, effective_steps, effective_cfg = _apply_edit_turbo(
        model,
        model_name,
        steps,
        cfg,
        enable_turbo_mode,
        turbo_lora_name,
        turbo_strength,
    )
    model = _patch_qwen_model_sampling(model, shift)
    clip = _load_qwen_clip(clip_name)
    vae = _load_qwen_vae(vae_name)
    scaled = _scale_qwen_edit_image(image)
    positive = _encode_qwen_edit_conditioning(clip, prompt, vae, scaled, image2=image2, image3=image3)
    negative = _encode_qwen_edit_conditioning(clip, negative_prompt, vae, scaled, image2=image2, image3=image3)
    positive = _apply_reference_latents_method(positive, reference_latents_method)
    negative = _apply_reference_latents_method(negative, reference_latents_method)
    latent = _encode_image_to_latent(vae, scaled)
    model = _apply_cfg_norm(model, cfg_norm_strength)
    sampled = _sample_qwen(model, positive, negative, latent, effective_steps, int(seed), effective_cfg, sampler_name, scheduler)
    return _decode_qwen_latent(vae, sampled)
```

- [ ] **Step 4: Update node method calls to pass the new parameters**

```python
# qwen/nodes.py
def generate(
    self,
    model_name: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    batch_size: int,
    negative_prompt: str,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    shift: float,
    enable_turbo_mode: bool,
    turbo_lora_name: str,
    turbo_strength: float,
):
    return (
        runtime.run_qwen_text_to_image(
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            batch_size=batch_size,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            shift=shift,
            enable_turbo_mode=enable_turbo_mode,
            turbo_lora_name=turbo_lora_name,
            turbo_strength=turbo_strength,
        ),
    )
```

```python
def generate(
    self,
    model_name: str,
    image,
    prompt: str,
    steps: int,
    seed: int,
    negative_prompt: str,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    shift: float,
    cfg_norm_strength: float,
    reference_latents_method: str,
    enable_turbo_mode: bool,
    turbo_lora_name: str,
    turbo_strength: float,
    image2=None,
    image3=None,
):
    return (
        runtime.run_qwen_image_edit(
            model_name=model_name,
            image=image,
            image2=image2,
            image3=image3,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            shift=shift,
            cfg_norm_strength=cfg_norm_strength,
            reference_latents_method=reference_latents_method,
            enable_turbo_mode=enable_turbo_mode,
            turbo_lora_name=turbo_lora_name,
            turbo_strength=turbo_strength,
        ),
    )
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_passes_optional_images_negative_prompt_and_edit_controls tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_enables_turbo_lora_and_uses_turbo_preset tests.test_qwen_nodes.TestQwenNodes.test_qwen_edit_runtime_fails_when_2511_turbo_is_requested -v`
Expected: PASS with image-edit advanced controls routed through, edit turbo working for 2509, and 2511 turbo failing clearly.

- [ ] **Step 6: Commit**

```bash
git add qwen/discovery.py qwen/runtime.py qwen/nodes.py tests/test_qwen_nodes.py
git commit -m "feat: add qwen edit advanced controls"
```

### Task 4: Run dedicated and full regression verification

**Files:**
- Modify: `tests/test_qwen_nodes.py` (only if a regression run exposes a mismatch)
- Test: `tests/test_qwen_nodes.py`

- [ ] **Step 1: Run the dedicated Qwen test file**

Run: `python3 -m unittest discover -s tests -p 'test_qwen_nodes.py' -v`
Expected: PASS with zero failures across schema, discovery, advanced passthrough, turbo, and failure-path coverage.

- [ ] **Step 2: Run the full plugin test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS with zero regressions in the existing conditioning, loader/prompt, latent, universal, and upscale suites.

- [ ] **Step 3: Inspect the final worktree state**

```bash
git status --short
```

Expected:
- no output if you committed each task as instructed
- `?? docs/superpowers/plans/2026-05-20-qwen-advanced-params.md` is acceptable if you choose not to commit the plan file itself

- [ ] **Step 4: If Step 2 exposed any regression fixes after the earlier commits, commit only those final verification fixes**

```bash
git add qwen/discovery.py qwen/runtime.py qwen/nodes.py tests/test_qwen_nodes.py
git commit -m "test: verify qwen advanced params"
```
