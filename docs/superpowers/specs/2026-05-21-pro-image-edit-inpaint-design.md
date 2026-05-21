# LLS Pro Image Edit And Inpaint Design

## Context

`LLS-node` already has a minimal local repair chain:

- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

That chain is useful as a lightweight masked resampling helper, but it is not a full native image edit / inpaint pipeline.

The current simplified implementation:

- encodes the full work image into latent space
- optionally attaches `noise_mask`
- reuses the existing sampler
- composites back with placeholder finish logic

This explains the current user-visible behavior:

- edits can be weak or unstable
- prompt-following for small masked regions is poor
- generated content can collapse into blurred patches
- finish behavior does not yet behave like a real masked composite

The requested change is to add a new full-featured image edit / inpaint workflow that:

- supports both `SDXL` and `FLUX`
- supports `auto` routing and manual backend override
- keeps the current `LLSSimple*` repair chain untouched
- exposes a stable workflow for true local edit / inpaint instead of only masked latent resampling

## Goals

- Add a new professional edit pipeline without breaking existing `LLSSimple*` repair nodes.
- Support `SDXL` native inpaint/edit-capable models.
- Support `FLUX` native inpaint/edit-capable models.
- Provide `auto | sdxl | flux` backend selection on the new nodes.
- Route automatically using model capability metadata instead of family-name-only guesses.
- Build backend-specific edit conditioning instead of relying on `noise_mask` alone.
- Produce real masked finish compositing with preview output.
- Make new-model onboarding predictable by isolating backend logic behind a registry.

## Non-Goals

This iteration will not:

- replace or silently mutate the existing `LLSSimpleRepairPrepare`, `LLSSimpleKSampler`, or `LLSSimpleRepairFinish`
- silently fall back from the new professional pipeline into the old simplified `latent_mask` path
- add `SD3`, `QWEN`, or `ZIMAGE` professional edit backends
- add new frontend drawing UI beyond the existing mask workflow
- implement magic automatic support for edit-capable models whose capability cannot be inferred or declared

## Chosen Architecture

Add a new feature package dedicated to professional image edit and inpaint:

- `pro_edit/__init__.py`
- `pro_edit/pro_edit_prepare.py`
- `pro_edit/pro_edit_bridge.py`
- `pro_edit/pro_edit_finish.py`
- `pro_edit/pro_edit_utils.py`
- `pro_edit/backends/__init__.py`
- `pro_edit/backends/base.py`
- `pro_edit/backends/registry.py`
- `pro_edit/backends/sdxl.py`
- `pro_edit/backends/flux.py`

Registration remains centralized through the root package loader by appending `pro_edit` to `_SUBPACKAGES` in the top-level `__init__.py`.

The existing `repair/` package remains the simplified path. The new `pro_edit/` package is a separate feature family instead of a refactor-in-place. This preserves compatibility, keeps responsibilities clear, and prevents the simplified sampler compatibility work from being mixed with native edit logic.

## New Type

Introduce a new logical payload type:

- `LLS_EDIT_INFO`

It is implemented as a Python `dict` payload in practice and may also be passed around as a JSON-like `STRING`, following the same normalization pattern already used by repair metadata.

`LLS_EDIT_INFO` is the contract between:

- `LLS Pro Image Edit Prepare`
- `LLS Pro KSampler Bridge`
- `LLS Pro Image Edit Finish`

## Node Set

### `LLS Pro Image Edit Prepare`

#### Display Name

`LLS Pro Image Edit Prepare`

#### Category

`LLS/Image Edit`

#### Required Inputs

- `image: IMAGE`
- `mask: MASK`
- `vae: VAE`
- `positive: CONDITIONING`
- `negative: CONDITIONING`

#### Optional Inputs

- `model: MODEL`
- `model_info: STRING`

#### Parameters

- `backend_mode: auto | sdxl | flux`
- `edit_scope: auto | region | crop | canvas`
- `mask_grow: INT`
- `mask_blur: FLOAT`
- `mask_threshold: FLOAT`
- `invert_mask: BOOLEAN`
- `crop_context: INT`
- `crop_context_factor: FLOAT`
- `min_size: INT`
- `max_size: INT`
- `resize_mode: fit | pad | stretch`
- `expand_left: INT`
- `expand_right: INT`
- `expand_top: INT`
- `expand_bottom: INT`
- `canvas_fill: edge | blur | black | white | neutral`
- `auto_recommend: enabled | disabled`

#### Outputs

- `latent: LATENT`
- `work_image: IMAGE`
- `work_mask: MASK`
- `edit_info: LLS_EDIT_INFO`
- `recommended_denoise: FLOAT`
- `positive: CONDITIONING`
- `negative: CONDITIONING`

#### Responsibility

This node prepares a true backend-aware edit payload. It must not merely encode the work image and attach a `noise_mask`. Instead, it must construct the conditioning structure expected by the chosen backend.

For native inpaint/image edit backends, this means following the same conceptual contract as ComfyUI’s inpaint conditioning path:

- prepare masked pixel-space input
- build `concat_latent_image`
- build `concat_mask`
- preserve the original latent as the primary latent payload
- attach `noise_mask` only when appropriate for backend behavior

This brings the new path closer to ComfyUI’s `InpaintModelConditioning` semantics instead of the current simplified `noise_mask`-only behavior.

### `LLS Pro KSampler Bridge`

#### Display Name

`LLS Pro KSampler Bridge`

#### Category

`LLS/Image Edit`

#### Required Inputs

- `model: MODEL`
- `positive: CONDITIONING`
- `negative: CONDITIONING`
- `latent_image: LATENT`

#### Optional Inputs

- `edit_info: LLS_EDIT_INFO`
- `model_info: STRING`

#### Parameters

- `backend_mode: auto | sdxl | flux`
- `quality_preset`
- `seed`
- `steps`
- `cfg`
- `sampler_name`
- `scheduler`
- `denoise`
- `denoise_mode: manual | auto_from_edit`
- `flux_guidance`
- `model_family`

#### Outputs

- `latent: LATENT`
- `sample_info: STRING`

#### Responsibility

This node is the sampling boundary for the professional edit chain. It remains a full sampler node, not just a passive adapter.

It must:

- resolve the effective backend using `edit_info` plus current model metadata
- validate that the chosen backend is actually compatible with the current model
- apply backend-specific conditioning adaptation before sampling
- use `recommended_denoise` when `denoise_mode = auto_from_edit`
- preserve explicit manual override behavior for debugging and special models

The bridge may internally reuse the same low-level sampling helper used by `LLSSimpleKSampler`, but it must not inherit the simplified repair semantics.

### `LLS Pro Image Edit Finish`

#### Display Name

`LLS Pro Image Edit Finish`

#### Category

`LLS/Image Edit`

#### Required Inputs

- `original_image: IMAGE`
- `generated_image: IMAGE`
- `edit_info: LLS_EDIT_INFO`

#### Optional Inputs

- `work_mask: MASK`
- `sample_info: STRING`

#### Parameters

- `feather: FLOAT`
- `color_match: disabled | mean_std | histogram_simple`
- `brightness_match: disabled | enabled`
- `blend_strength: FLOAT`
- `restore_unmasked_area: BOOLEAN`
- `edge_fix: none | soft | strong`
- `preview_mode: final | compare | mask | before_after`

#### Outputs

- `final_image: IMAGE`
- `preview_image: IMAGE`

#### Responsibility

This node must perform real masked compositing according to `edit_scope`, `work_size`, `crop_box`, `canvas_expand`, and `work_mask`.

It must not behave like the current placeholder finish path that simply resizes the generated image back to original dimensions.

The output preview must be meaningful:

- `final`: final composite
- `compare`: before/after comparison
- `mask`: original image with mask overlay
- `before_after`: side-by-side comparison

## Workflow Design

### Existing Simplified Workflow

The current repair chain remains unchanged:

- `Load Image`
- `Load Mask`
- `LLS Simple Checkpoint Loader`
- `LLS Simple Prompt Encode`
- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `VAE Decode`
- `LLS Simple Repair Finish`

### New Professional Workflow

- `Load Image`
- `Load Mask` or `LLS Simple Mask Draw`
- `LLS Simple Checkpoint Loader`
- `LLS Simple Prompt Encode`
- `LLS Pro Image Edit Prepare`
- `LLS Pro KSampler Bridge`
- `VAE Decode`
- `LLS Pro Image Edit Finish`
- `Preview Image`

The professional path does not start from `Empty Latent`. It starts from `image + mask + prompt + backend-aware edit conditioning`.

## Backend Interface

`pro_edit/backends/base.py` should define one focused backend contract. The exact Python shape can be a lightweight base class or protocol, but each backend must expose the same responsibilities:

- normalize and validate capability metadata
- declare whether it supports a given model
- prepare backend-specific latent + conditioning payloads
- declare sampler overrides or recommendations when needed
- describe routing results for debug metadata
- provide finish-time hints when the backend has special composite expectations

Each backend owns only its own semantics. Shared geometric and mask utilities stay in `pro_edit/pro_edit_utils.py`.

## Backend Implementations

### SDXL Backend

The `sdxl` backend is responsible for models whose professional local edit path should follow native SDXL inpaint/edit semantics.

Behavior:

- use masked pixel-space preprocessing
- build `concat_latent_image`
- build `concat_mask`
- encode the original work image latent
- attach `noise_mask` only when it improves or matches native behavior
- preserve crop/region/canvas scope metadata

Auto eligibility:

- `model_family` resolves to `SDXL`
- and either:
  - `model_role in {inpaint, edit, fill}`
  - or `supports_inpaint_native = true`
  - or `preferred_edit_backend = sdxl`

### FLUX Backend

The `flux` backend is responsible for models whose professional local edit path should follow native FLUX image edit / inpaint semantics.

Behavior:

- build edit conditioning using native concat-image + concat-mask semantics, not simplified noise-mask-only behavior
- preserve FLUX guidance handling
- respect FLUX-native sampling defaults when `quality_preset` requests family defaults
- validate that the model exposes real edit capability before proceeding

Auto eligibility:

- `model_family` resolves to `FLUX_DEV` or `FLUX_SCHNELL`
- and either:
  - `model_role in {inpaint, edit, fill}`
  - or `supports_image_edit_native = true`
  - or `preferred_edit_backend = flux`

Professional FLUX support does not mean “all FLUX models work equally well.” The backend contract only guarantees correct routing and conditioning semantics for models declared edit-capable.

## Auto Routing And Manual Override

Every professional node exposes:

- `backend_mode = auto | sdxl | flux`

### Auto Resolution Order

1. `edit_info.backend_name` when already resolved upstream
2. `model_info.preferred_edit_backend`
3. `_lls_*` metadata already tagged on `model`
4. `model_family + model_role + capability flags`
5. explicit runtime error if no professional backend matches

### Manual Override Rules

- `backend_mode = sdxl` forces the SDXL backend
- `backend_mode = flux` forces the FLUX backend
- forced selection still performs capability validation
- if the selected backend does not match the actual model capability, the node must raise a clear runtime error

### No Silent Downgrade

The new professional nodes must never silently downgrade into the old simplified latent-mask path. If the current model is not professional-edit-capable, the user must get an explicit error that says so.

## Capability Metadata

The routing layer should operate on a normalized capability record with these fields:

- `model_family`
- `model_role`
- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

### Capability Meaning

- `model_role`: `base | inpaint | edit | fill | refiner | unknown`
- `supports_inpaint_native`: model can consume native inpaint concat conditions
- `supports_image_edit_native`: model can consume native image edit concat conditions
- `preferred_edit_backend`: routing override for auto mode

### Source Of Truth

The record is built from:

- explicit `model_info` payloads
- `_lls_*` tags on loaded model objects
- family/name inference fallbacks in loader utilities

This requires extending existing model metadata utilities so new models can be added by updating capability detection instead of rewriting node logic.

## Stability Expectations

The new pipeline should be more stable than the current simplified chain because it changes the problem definition:

- from “sample original latent with a mask”
- to “sample with backend-native edit conditioning that explicitly describes the masked edit problem”

That does not guarantee every model will produce the desired object insertion. It does guarantee the pipeline is structurally correct for true image edit / inpaint semantics instead of relying on a simplified approximation.

## Error Handling

The professional nodes must fail early with explicit messages for:

- missing `image`
- missing `mask`
- missing `vae`
- missing `positive` or `negative`
- unsupported `backend_mode`
- `auto` routing with no matching backend
- manual backend mismatch
- edit-capability metadata missing for the requested professional path
- scope-specific invalid geometry such as empty crop mask

## Finish Semantics

`LLS Pro Image Edit Finish` must implement real merge behavior for:

- `region`: blend generated output back onto original image by mask
- `crop`: place generated crop back into original image using `crop_box`, then blend by resized work mask
- `canvas`: merge expanded work image back into final canvas result according to `original_box_in_canvas`

This is intentionally stricter than the current simplified repair finish logic, which still behaves like a placeholder.

## Testing Strategy

Add backend-specific tests instead of only high-level dispatch tests.

Recommended files:

- `tests/test_pro_edit_prepare_sdxl.py`
- `tests/test_pro_edit_prepare_flux.py`
- `tests/test_pro_edit_finish.py`
- `tests/test_pro_edit_registry.py`
- `tests/test_pro_edit_bridge.py`

Required coverage:

- `auto` routes to the expected backend for declared `SDXL` edit-capable models
- `auto` routes to the expected backend for declared `FLUX` edit-capable models
- manual `sdxl` override rejects FLUX-only models
- manual `flux` override rejects SDXL-only models
- prepare writes `backend_name`, `routing_reason`, and capability flags into `edit_info`
- bridge honors `auto_from_edit`
- finish really uses `work_mask` and geometry metadata
- existing `LLSSimple*` repair tests remain unchanged and continue to pass

The new tests must validate actual edit payload behavior rather than only mocking away the core compose path.

## Documentation

README updates should add a new section for the professional edit chain:

- what it is for
- how it differs from the existing `Simple` repair chain
- auto vs manual backend selection
- model capability requirements
- example workflows for `SDXL` and `FLUX`

The key user-facing distinction must be explicit:

- `Simple` = lightweight masked latent resampling
- `Pro` = true image edit / inpaint pipeline

## Compatibility Boundary

The existing simplified chain remains the low-friction default for lightweight experimentation and backward compatibility.

The new professional chain is stricter:

- it requires real backend capability
- it uses native edit conditioning semantics
- it does not silently pretend unsupported models are edit-capable

This boundary is intentional. It prevents “seems to run, but behavior is wrong” failure modes.
