# LLS Auto Task Router Design

**Date:** 2026-05-20

**Goal**

Add an `LLSAutoTaskRouter` node that identifies the primary task automatically, applies a manual override when requested, validates model capabilities, and emits a lightweight routing context without moving large ComfyUI objects through custom context fields.

**Non-Goals**

- No `video` task handling in this iteration.
- No removal of `LLSTaskController` in this iteration.
- No migration of large objects like `IMAGE`, `MASK`, `LATENT`, `MODEL`, `CLIP`, `VAE`, or `CONDITIONING` into routing context.
- No full downstream rewrite to make every node `route_key`-driven in this iteration.

## Current State

The plugin currently centralizes workflow intent in `LLS_TASK_CONTEXT`, mainly through `LLSTaskController` and helper functions in `utils/task_context.py`. Downstream nodes consume `task_context` keys such as `task_mode`, `enable_upscale`, `enable_controlnet`, `enable_reference`, `model_family`, and recommendation fields.

This works for explicit workflows, but it forces users to manually select task type up front. It also leaves task recognition, modifier recognition, and model capability validation scattered or absent.

## Design Summary

This change keeps the existing `LLS_TASK_CONTEXT` port type for compatibility, but upgrades the dictionary shape carried through that port.

Two nodes will share the same routing core:

- `LLSAutoTaskRouter`: new node, automatic task identification with manual override fallback
- `LLSTaskController`: existing node, preserved for compatibility, internally normalized onto the same context schema

Shared routing logic will move into `utils/task_context.py` so that task identification, modifier detection, route generation, and capability validation are implemented once.

## Primary Task Detection

`primary_task` is resolved using the following rules:

1. If `task_override != "auto"`, use `task_override`.
2. Otherwise:
   - If `input_image` and `mask_image` are connected and any of `outpaint_left`, `outpaint_right`, `outpaint_top`, `outpaint_bottom` is greater than `0`, use `outpaint`.
   - Else if `input_image` and `mask_image` are connected, use `inpaint`.
   - Else if `input_image` is connected, use `img2img`.
   - Else use `txt2img`.

`video` is intentionally excluded for now.

## Modifier Detection

`modifiers` is a stable ordered list built from lightweight controls:

- Add `controlnet` when `control_image` is connected or `control_type != "none"`.
- Add `reference` when `reference_image` is connected.
- Add `instruction_edit` when `instruction_prompt` is non-empty after trimming whitespace.
- Add `upscale` when `enable_upscale` is true.
- Add `face_fix` when `enable_face_fix` is true.
- Add `detail_refine` when `enable_detail_refine` is true.

Modifier ordering is fixed and deterministic:

1. `controlnet`
2. `reference`
3. `instruction_edit`
4. `upscale`
5. `face_fix`
6. `detail_refine`

## Route Key

`route_key` is generated as follows:

- If `modifiers` is empty: `route_key = primary_task`
- Otherwise: `route_key = primary_task + "." + ".".join(modifiers)`

Examples:

- `txt2img`
- `img2img.controlnet.reference`
- `outpaint.reference.upscale`

## Context Schema

`LLSAutoTaskRouter` continues to output `LLS_TASK_CONTEXT`, but the context payload is expanded to include the new routing fields.

### Required Routing Fields

- `mode`
- `primary_task`
- `modifiers`
- `route_key`
- `model_family`
- `model_name`
- `required_inputs`
- `optional_inputs`
- `warnings`
- `errors`

### Compatibility Fields

To avoid breaking existing nodes, the context also carries or derives:

- `task_mode = primary_task`
- `enable_controlnet = ("controlnet" in modifiers)`
- `enable_reference = ("reference" in modifiers)`
- `enable_upscale = ("upscale" in modifiers)`
- existing recommendation and capability fields already produced by `parse_task_context()`

### Field Semantics

- `mode`
  - `"auto"` when the router used automatic detection
  - `"manual"` when `task_override` forced the task
- `primary_task`
  - one of `txt2img`, `img2img`, `inpaint`, `outpaint`
- `modifiers`
  - ordered list of modifier names
- `required_inputs`
  - list of input names required for the chosen route, never the objects themselves
- `optional_inputs`
  - list of supported but non-required input names for the chosen route
- `warnings`
  - non-fatal issues
- `errors`
  - fatal route validation issues

## Large Object Boundary

The routing context must remain lightweight.

The following objects must not be stored in `LLS_TASK_CONTEXT`:

- `IMAGE`
- `MASK`
- `LATENT`
- `MODEL`
- `CLIP`
- `VAE`
- `CONDITIONING`

These continue through normal ComfyUI ports only.

## Model Capability Validation

The routing layer validates route compatibility against a `MODEL_REGISTRY`.

### Registry Structure

The first iteration uses family-level defaults with optional model-name overrides:

```python
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

MODEL_NAME_OVERRIDES = {
    # "specific_model.safetensors": {...}
}
```

### Validation Rules

Validation happens in three passes:

1. Input completeness
   - `inpaint` requires `input_image` and `mask_image`
   - `outpaint` requires `input_image` and `mask_image`
   - `controlnet` requires `control_image` when the modifier is active
2. Model support
   - `primary_task` must be in the model's supported `primary_tasks`
   - every modifier must be in the model's supported `modifiers`
3. Configuration coherence
   - outpaint padding without required image inputs is invalid
   - forced `task_override` does not bypass support checks

Validation does not silently downgrade the route.

## Error and Warning Behavior

`LLSAutoTaskRouter` always builds internal `warnings` and `errors` lists first.

Behavior:

- If `warnings` is non-empty and `errors` is empty, the node succeeds and includes warnings in context.
- If `errors` is non-empty, the node raises `RuntimeError` with a readable aggregated message and does not silently continue.

Example failure messages:

- `[LLS] Auto task routing failed: model family 'FLUX_SCHNELL' does not support primary task 'outpaint'.`
- `[LLS] Auto task routing failed: modifier 'controlnet' requires control_image.`

## Node Interface

### `LLSAutoTaskRouter`

**Category:** `LLS/Task`

**Function:** `route`

**Return**

- `RETURN_TYPES = (LLS_TASK_CONTEXT_TYPE,)`
- `RETURN_NAMES = ("task_context",)`

**Required inputs**

- `task_override`: `["auto", "txt2img", "img2img", "inpaint", "outpaint"]`, default `"auto"`
- `model_family`: family selection with current compatibility choices
- `model_name`: model name selector
- `control_type`: string-like choice, default `"none"`
- `instruction_prompt`: string, default empty
- `enable_upscale`: boolean
- `enable_face_fix`: boolean
- `enable_detail_refine`: boolean
- `outpaint_left`: int, default `0`
- `outpaint_right`: int, default `0`
- `outpaint_top`: int, default `0`
- `outpaint_bottom`: int, default `0`

**Optional inputs**

- `input_image`: `IMAGE`
- `mask_image`: `MASK`
- `control_image`: `IMAGE`
- `reference_image`: `IMAGE`

This keeps large objects on native ports while allowing route inference from presence or absence.

## Shared Routing Core

The routing behavior should live in reusable helpers inside `utils/task_context.py`.

Recommended helper groups:

- route normalization
- primary task inference
- modifier inference
- required and optional input derivation
- model capability lookup
- route validation
- final context assembly

Both `LLSAutoTaskRouter` and `LLSTaskController` must call this shared logic.

## Compatibility Strategy

### `LLSTaskController`

The node remains registered and usable.

Its behavior changes only internally:

- it still accepts explicit manual task controls
- it now emits the expanded routing context format
- it sets `mode = "manual"`
- it maps `task_mode` onto `primary_task`

This keeps old workflows functional while aligning future routing data.

### `parse_task_context()`

`parse_task_context()` becomes the compatibility bridge between old and new payloads.

Required normalization behavior:

- if `primary_task` exists and `task_mode` is missing, set `task_mode = primary_task`
- if `task_mode` exists and `primary_task` is missing, set `primary_task = task_mode`
- if `mode` is missing, default old payloads to `"manual"`
- if `modifiers` is missing, derive it from legacy booleans:
  - `enable_controlnet`
  - `enable_reference`
  - `enable_upscale`
- if `route_key` is missing, rebuild it from `primary_task` and `modifiers`

This lets old and new nodes exchange context safely during migration.

### Downstream Nodes

This iteration does not rewrite all downstream nodes around `route_key`.

Instead, downstream nodes remain compatible by:

- continuing to accept `task_context`
- continuing to read old fields such as `task_mode`
- optionally preferring `primary_task` over `task_mode` when available
- deriving enable flags from `modifiers` when needed

This minimizes rollout risk.

## Testing Strategy

### New Router Tests

Add a dedicated router test module, for example:

- `tests/test_task_router.py`

Coverage:

- automatic `txt2img`
- automatic `img2img`
- automatic `inpaint`
- automatic `outpaint`
- modifier detection for all supported modifiers
- `task_override` precedence
- `route_key` generation
- model capability errors
- missing required input errors

### Context Compatibility Tests

Add direct tests for `parse_task_context()` normalization:

- old context -> new fields are filled
- new context -> legacy compatibility fields are filled

### Regression Tests

Use `LLSAutoTaskRouter` output as upstream context for existing nodes such as:

- prompt encoding
- sampling
- save image metadata

The purpose is to prove that the new context shape does not break current downstream consumers.

## Implementation Phases

1. Add shared routing helpers and `MODEL_REGISTRY` in `utils/task_context.py`
2. Add `LLSAutoTaskRouter` in `task/nodes.py`
3. Refactor `LLSTaskController` to emit the same expanded schema
4. Add router and compatibility tests
5. Apply minimal downstream compatibility reads where needed

## Risks

- If compatibility keys are not preserved, many existing node tests and workflows will break immediately.
- If registry defaults are too optimistic or too restrictive, users will see false-positive errors or unsupported executions.
- If `parse_task_context()` is overloaded with too much unrelated policy, future refactors will become harder. The new routing helpers should stay factored and focused.

## Acceptance Criteria

- `LLSAutoTaskRouter` exists and outputs `LLS_TASK_CONTEXT`
- automatic task detection works for `txt2img`, `img2img`, `inpaint`, and `outpaint`
- manual override works and still validates support
- `route_key` is deterministic
- large objects are not stored in context
- unsupported model/task or model/modifier combinations fail loudly
- existing `LLSTaskController` remains usable
- existing downstream nodes continue to accept the emitted context without broad breakage
- test coverage includes routing, compatibility normalization, and downstream regression
