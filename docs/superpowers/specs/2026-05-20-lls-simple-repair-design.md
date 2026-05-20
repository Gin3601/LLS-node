# LLS Simple Repair Design

## Context

`LLS-node` already has a stable minimal `txt2img` path and a stable minimal `img2img` path:

- `LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Simple Empty Latent -> LLS Simple KSampler -> VAE Decode`
- `Load Image -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> VAE Encode -> LLS Simple KSampler -> VAE Decode`

The requested change is to add a general-purpose local repair workflow without breaking either existing path, without introducing multiple task-specific inpaint nodes, and without creating a second sampler node for inpainting.

The user wants one repair feature family with:

- one prepare node
- one finish node
- the existing `LLS Simple KSampler` upgraded to understand repair metadata

The feature must support:

- `region` repair on the original image
- `crop` repair on an automatically derived local work area
- `canvas` repair for expansion and missing-area fill

The implementation must live in its own feature package and not collapse into a single large `nodes.py` file.

## Goals

- Keep the current `txt2img` flow unchanged when no repair inputs are connected.
- Keep the current `img2img` flow unchanged when no repair inputs are connected.
- Add exactly two repair nodes:
  - `LLS Simple Repair Prepare`
  - `LLS Simple Repair Finish`
- Upgrade `LLS Simple KSampler` instead of adding an inpaint-specific sampler node.
- Use one shared `repair_info` payload to coordinate prepare, sample, and finish behavior.
- Support `region`, `crop`, and `canvas` as first-class scopes.
- Support `latent_mask`, `vae_inpaint`, and `native_fill` as repair kernel choices, with clear fallback behavior when a kernel is not available.
- Make the first shipping implementation production-oriented for the requested scope, while explicitly deferring unsupported backend-specific integrations.

## Non-Goals

This change will not implement the following as working end-to-end backend features in this iteration:

- dedicated ControlNet execution
- dedicated IP-Adapter execution
- dedicated Reference Transfer execution
- native sampling adapters for `SD3`, `QWEN`, or `ZIMAGE`
- true backend-specific `native_fill` sampling

These remain explicit interfaces or metadata fields with clear warnings or runtime errors instead of silent fake support.

## Chosen Architecture

Add a new `repair/` feature package and keep node logic split by responsibility:

- `repair/__init__.py`
- `repair/repair_prepare.py`
- `repair/repair_finish.py`
- `repair/repair_utils.py`

Registration remains centralized through the existing top-level package loader by appending `repair` to `_SUBPACKAGES` in the root `__init__.py`.

The upgraded `LLS Simple KSampler` remains in `sampling/nodes.py` and only gains a repair-aware compatibility layer. It does not absorb prepare or finish logic.

This preserves the project’s package-based node organization, keeps the sampler node stable, and makes repair behavior a distinct feature domain instead of a special case scattered across existing files.

## Workflow Design

### Existing txt2img workflow

No changes:

- `LLS Simple Checkpoint Loader`
- `LLS Simple Prompt Encode`
- `LLS Simple Empty Latent`
- `LLS Simple KSampler`
- `VAE Decode`
- `Preview Image`

### Existing img2img workflow

No changes:

- `Load Image`
- `LLS Simple Checkpoint Loader`
- `LLS Simple Prompt Encode`
- `VAE Encode`
- `LLS Simple KSampler`
- `VAE Decode`
- `Preview Image`

### New repair workflow

- `Load Image`
- `Load Mask`
- `LLS Simple Checkpoint Loader`
- `LLS Simple Prompt Encode`
- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `VAE Decode`
- `LLS Simple Repair Finish`
- `Preview Image`

Repair does not start from `Empty Latent`. It starts from `image + mask + vae`, which are converted into a repair-aware latent payload by `LLS Simple Repair Prepare`.

## Types

Two new logical types are introduced for ComfyUI port labeling:

- `LLS_REPAIR_INFO`
- `LLS_GUIDANCE_STACK`

Both are implemented as Python `dict` payloads in practice. Nodes must accept either raw `dict` values or JSON-like string payloads and normalize them before use.

## Node Design

### `LLS Simple Repair Prepare`

#### Inputs

Required:

- `image: IMAGE`
- `mask: MASK`
- `vae: VAE`

Optional:

- `model_info: LLS_MODEL_INFO | dict | STRING`
- `positive: CONDITIONING`
- `negative: CONDITIONING`

#### Parameters

- `repair_scope`: `auto | region | crop | canvas`
- `repair_kernel`: `auto | latent_mask | vae_inpaint | native_fill`
- `task_hint`: `auto | repair | remove | replace | fill | appearance | content | structure | dehaze | deshadow | recolor`
- `mask_grow: INT`
- `mask_blur: FLOAT`
- `mask_threshold: FLOAT`
- `invert_mask: BOOLEAN`
- `crop_context: INT`
- `crop_context_factor: FLOAT`
- `min_size: INT`
- `max_size: INT`
- `resize_mode: keep_aspect | force_square | ranged_size`
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
- `repair_info: LLS_REPAIR_INFO`
- `recommended_denoise: FLOAT`

#### Responsibility

This node prepares repair inputs only. It does not sample and does not composite the final result back into the source image.

### `LLS Simple Repair Finish`

#### Inputs

Required:

- `original_image: IMAGE`
- `generated_image: IMAGE`
- `repair_info: LLS_REPAIR_INFO | dict | STRING`

Optional:

- `work_mask: MASK`
- `sample_info: dict | STRING`

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

This node composites the decoded generation result back into the correct final image space using `repair_info` and the prepared work mask.

### `LLS Simple KSampler`

#### Existing required inputs

The current required inputs remain unchanged:

- `model`
- `positive`
- `negative`
- `latent_image`
- current sampling controls

#### New optional inputs

- `repair_info: LLS_REPAIR_INFO | dict | STRING`
- `guidance_stack: LLS_GUIDANCE_STACK | dict | STRING`
- `model_info: LLS_MODEL_INFO | dict | STRING`

#### New parameters

- `denoise_mode: manual | auto_from_repair`
- `adapter_mode: auto | sd_classic | flux | sd3 | qwen | zimage`

#### Output shape

The node keeps its current outputs:

- `latent`
- `sample_info`

The `sample_info` payload is extended to include repair metadata when repair mode is active.

## Shared `repair_info` Contract

The repair payload is a normalized `dict` and must include at least:

```json
{
  "repair_scope": "region",
  "repair_kernel": "latent_mask",
  "task_hint": "repair",
  "original_size": [1024, 1024],
  "work_size": [1024, 1024],
  "crop_box": null,
  "crop_scale": null,
  "canvas_expand": [0, 0, 0, 0],
  "mask_grow": 8,
  "mask_blur": 8.0,
  "mask_threshold": 0.5,
  "invert_mask": false,
  "recommended_denoise": 0.45,
  "model_family": "SD1.5",
  "model_role": "normal",
  "repair_payload_version": "1.0"
}
```

Additional normalized fields are included when available:

- `has_mask`
- `mask_area_ratio`
- `mask_bbox`
- `warnings`
- `supports_inpaint_native`
- `effective_repair_kernel`
- `original_box_in_canvas`

The implementation treats missing fields as invalid or incomplete repair metadata and either repairs them deterministically or raises a clear error if the payload cannot be trusted.

## Model Metadata Strategy

Repair preparation must not crash if `model_info` is missing or only partially populated.

The normalization rules are:

- missing `model_info` produces `model_family = "UNKNOWN"`
- missing `model_role` produces `model_role = "unknown"`
- unknown `supports_inpaint_native` defaults to `false`
- warnings are appended to the payload instead of silently ignored

`LLS Simple KSampler` resolves its effective family in this order:

1. explicit `model_info`
2. normalized `repair_info`
3. existing model-object family inference

Sampling is only considered implemented for `SD1.5`, `SDXL`, and `FLUX`. Unsupported families must produce a clear runtime error instead of pretending to work.

## Scope Resolution

When `repair_scope=auto`, resolve scope with the following rules:

1. If any canvas expansion value is greater than zero, use `canvas`.
2. Otherwise compute normalized mask metrics and use `crop` if:
   - `mask_area_ratio <= 0.35` and bounding box width and height are each `<= 0.8` of the image size, or
   - `mask_area_ratio <= 0.18`
3. Otherwise use `region`.

If the processed mask is empty:

- in `canvas` mode, continue only if expansion creates a non-empty expansion mask
- otherwise raise a clear empty-mask error

## Kernel Resolution

When `repair_kernel=auto`, resolve kernel with the following rules:

1. If the model role is `inpaint`, `fill`, or `edit`, and `supports_inpaint_native=true`, request `native_fill`.
2. Otherwise if:
   - scope is not `canvas`
   - `mask_area_ratio <= 0.20`
   - `task_hint` is one of `repair`, `appearance`, `dehaze`, `deshadow`, or `recolor`
   then request `latent_mask`.
3. Otherwise request `vae_inpaint`.

If `native_fill` is requested but not actually supported by the runtime path, it is downgraded to `vae_inpaint` and the downgrade is written to `repair_info["warnings"]`.

## Recommended Denoise Rules

Base values by task:

- `repair`: `0.45`
- `appearance`: `0.50`
- `dehaze`: `0.55`
- `deshadow`: `0.55`
- `recolor`: `0.55`
- `structure`: `0.60`
- `content`: `0.65`
- `replace`: `0.72`
- `remove`: `0.88`
- `fill`: `0.90`
- `auto`: `0.55`

Scope adjustments:

- `canvas`: final value is at least `0.90`
- `crop`: final value is clamped into `[0.30, 0.65]`
- `region`: keep the base value unless kernel-specific fallback rules apply

If `auto_recommend=disabled`, use deterministic defaults:

- `region + latent_mask`: `0.45`
- `region + vae_inpaint`: `0.55`
- `crop`: `0.50`
- `canvas`: `0.90`

## Prepare Runtime Behavior

### Common preprocessing

The prepare node must:

- align mask size to the image size
- normalize mask values into `0.0..1.0`
- apply optional inversion
- apply thresholding
- apply grow / dilate behavior
- apply feather / blur behavior
- compute mask statistics and bounding box

### `region`

- keep the original image size
- set `work_image = image`
- set `work_mask = processed_mask`
- encode the full work image with the VAE

For `latent_mask`:

- resize `work_mask` to latent resolution
- store it in `latent["noise_mask"]`
- keep latent samples sourced from the encoded image

For `vae_inpaint`:

- use ComfyUI’s native inpaint encode path when available
- otherwise construct a compatible latent payload from the encoded image plus processed mask metadata

For `native_fill`:

- keep the request in metadata
- downgrade to `vae_inpaint` if no native fill path exists

### `crop`

- derive the initial mask bounding box
- expand it by `crop_context`
- expand again by `crop_context_factor`
- clamp to the original image bounds
- crop `work_image` and `work_mask`
- compute resized work dimensions using `min_size`, `max_size`, and `resize_mode`
- store `crop_box`, `crop_scale`, and `work_size`
- encode the cropped work image

The preferred first implementation path is `vae_inpaint`. Explicit `latent_mask` remains allowed for consistency but is not the primary recommendation.

### `canvas`

- create a new expanded canvas using the requested edge sizes
- place the original image into its offset position
- derive an expansion mask for new canvas area
- if the user also supplied a repair mask, align it into the expanded coordinate space and merge it with the expansion mask
- apply `canvas_fill` to initialize the added area
- encode the expanded work image

The first implementation path uses `vae_inpaint` semantics.

## Finish Runtime Behavior

### Common finish logic

The finish node must:

- normalize and validate `repair_info`
- determine the repair scope
- derive the target output size
- resize `generated_image` when the size can be inferred safely
- build a final alpha mask from `work_mask`, `feather`, and `blend_strength`
- optionally perform color and brightness matching only inside the masked area

`edge_fix` behavior:

- `none`: standard feather only
- `soft`: one extra mild edge smoothing pass on the blend boundary
- `strong`: wider boundary smoothing, still limited to the seam area

### `region`

- resize generated output to original image size when needed
- blend only inside the mask
- if `restore_unmasked_area=true`, force all unmasked pixels back to the original image

### `crop`

- resize generated output to the stored crop-box size
- resize the work mask back into crop-box size
- paste the repaired crop back into the original image at `crop_box`
- keep everything outside the crop box as the original image

### `canvas`

- resize generated output to expanded work size
- if `restore_unmasked_area=true`, preserve the original image inside its original box as much as possible
- use generated pixels in the expanded regions
- feather the seam between original content and expanded content
- output the expanded image size rather than shrinking back to the original size

### Preview output

- `final`: return final image
- `compare`: horizontal before/after comparison
- `mask`: mask visualization
- `before_after`: `before | mask overlay | after`

## Color and Brightness Matching

The first release includes:

- `mean_std`: per-channel mean/standard-deviation alignment inside the mask
- `histogram_simple`: a simplified quantile-style tonal remap
- `brightness_match`: a post-color-match luminance alignment pass

Any failure during these optional adjustments must append a warning and continue instead of failing the whole node.

## KSampler Compatibility Rules

### No `repair_info`

When `repair_info` is absent, `LLS Simple KSampler` must preserve current behavior exactly:

- same family inference path
- same preset behavior
- same seed behavior
- same latent handling
- same `sample_info` fields, except for additive backwards-compatible metadata if needed

### With `repair_info`

When `repair_info` is present:

- mark `repair_mode = true`
- read `repair_scope`
- read `repair_kernel`
- read `recommended_denoise`
- read `model_family`
- read `model_role`

`denoise_mode` rules:

- `manual`: use the UI-provided `denoise`
- `auto_from_repair`: use `repair_info["recommended_denoise"]` when present, otherwise fall back to manual `denoise`

`guidance_stack` rules:

- accept and normalize the input
- do not fail if it is missing or empty
- record whether guidance was provided in `sample_info`
- do not attempt full ControlNet or adapter execution in this change

`adapter_mode` rules:

- `auto` resolves from the normalized family
- `sd_classic` and `flux` map to implemented current sampling paths
- `sd3`, `qwen`, and `zimage` produce explicit runtime errors because their repair-aware sampler adapters are not implemented

The `sample_info` payload must include:

```json
{
  "repair_mode": true,
  "repair_scope": "crop",
  "repair_kernel": "vae_inpaint",
  "denoise": 0.50,
  "model_family": "SDXL",
  "guidance_used": false
}
```

## Error Handling and Boundary Rules

The implementation must explicitly handle:

- image and mask size mismatch
- fully black masks
- fully white masks
- extremely small masks
- out-of-range mask boxes
- out-of-range crop boxes
- crop sizes below `min_size`
- crop sizes above `max_size`
- `canvas` requested with zero expansion
- missing VAE
- missing `model_info`
- incomplete `repair_info`
- generated-image size mismatches
- unsupported kernels
- unsupported model families

Rules:

- auto-correct whenever the correction is deterministic and low-risk
- otherwise raise a clear runtime error
- never fail silently

Notable behaviors:

- empty non-canvas masks are errors
- full-white masks remain valid
- `canvas` with zero expansion but a valid mask degrades to `region` with a warning
- `canvas` with zero expansion and no effective mask is an error
- unknown kernels are errors except for `native_fill`, which downgrades to `vae_inpaint`

## File-Level Responsibilities

- `repair/__init__.py`
  - export `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- `repair/repair_prepare.py`
  - define `LLSSimpleRepairPrepare`
- `repair/repair_finish.py`
  - define `LLSSimpleRepairFinish`
- `repair/repair_utils.py`
  - normalization helpers
  - mask processing helpers
  - crop and canvas geometry helpers
  - color and brightness matching helpers
  - repair payload validation helpers
- `sampling/nodes.py`
  - repair-aware sampler compatibility only
- `__init__.py`
  - register `repair` package
- `README.md`
  - document the repair workflow and compatibility rules

## Testing Strategy

Because the current shell environment does not provide `torch` or ComfyUI runtime modules, validation is split into two layers.

### Automated checks in this environment

- `python3 -m compileall .`
- plugin registration tests for the two new nodes
- node schema tests for prepare, finish, and upgraded sampler
- pure-Python tests for:
  - scope auto-resolution
  - kernel auto-resolution
  - denoise recommendation
  - crop box clamping
  - canvas expansion bookkeeping
  - repair-info normalization
  - sampler behavior with and without `repair_info`

### Manual ComfyUI verification outside this shell

- old `txt2img` workflow still loads and runs
- old `img2img` workflow still loads and runs
- `region` repair workflow runs
- `crop` repair workflow runs
- `canvas` repair workflow runs

The README must document the exact minimum wiring for these checks.

## README Changes

Add a dedicated repair section that documents:

- new `LLS Simple Repair Prepare`
- new `LLS Simple Repair Finish`
- upgraded `LLS Simple KSampler` repair support
- the fact that repair does not start from `Empty Latent`
- the minimal workflow:
  - `image + mask -> Repair Prepare -> KSampler -> VAE Decode -> Repair Finish`
- `repair_scope`
  - `region`
  - `crop`
  - `canvas`
- `repair_kernel`
  - `latent_mask`
  - `vae_inpaint`
  - `native_fill`
- `denoise_mode = auto_from_repair`
- compatibility guarantees for old workflows

## Implementation Order

1. Add `repair/` package and utility functions.
2. Implement `LLS Simple Repair Prepare`.
3. Implement `LLS Simple Repair Finish`.
4. Upgrade `LLS Simple KSampler`.
5. Register new nodes and update README.
6. Add tests and run compile/load checks.

## Acceptance Criteria

The change is complete when:

- the plugin registers `LLS Simple Repair Prepare`
- the plugin registers `LLS Simple Repair Finish`
- `LLS Simple KSampler` accepts `repair_info`, `guidance_stack`, and `model_info` as optional inputs
- existing `txt2img` and `img2img` code paths are preserved when `repair_info` is absent
- repair workflows produce repair-aware latent payloads from images and masks instead of `Empty Latent`
- finish workflows can merge `region`, `crop`, and `canvas` results back into their final image space
- unsupported families and unsupported advanced adapters fail clearly
- docs and tests cover the new workflow and its compatibility boundaries
