# LLS Simple Mask Draw Design

## Context

`LLS-node` already has a local repair workflow centered on:

- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

What is still missing is a user-friendly way to create and refine a repair mask directly on the source image inside the workflow. The current workflow assumes the mask already exists, which makes local repair inconvenient for common cases such as object removal, local cleanup, shadow removal, and selective enhancement.

The requested scope is intentionally narrow:

- add exactly one new node
- make it interactive in the ComfyUI frontend
- do not modify repair prepare/finish behavior in this phase
- do not add polygon, rectangle, ellipse, magic wand, auto-segmentation, or other advanced mask tools

The node must be usable as:

- `Load Image -> LLS Simple Mask Draw -> Preview Image`
- `Load Image -> LLS Simple Mask Draw -> LLS Simple Repair Prepare`

## Goals

- Add one new node: `LLS Simple Mask Draw`
- Let the user draw a white repair mask directly over the input image
- Support `brush` and `erase` modes in the first shipping version
- Support `Clear`, `Undo`, `Redo`, and mask overlay preview
- Output:
  - original `image`
  - final `mask`
  - `preview_image` with red semi-transparent overlay
- Accept an optional `input_mask` and allow continued editing on top of it
- Persist the current mask content across workflow save/load whenever possible
- Keep output `mask` resolution identical to the input `image`
- Fit the existing package-oriented structure of `LLS-node`

## Non-Goals

This iteration will not implement:

- polygon selection
- rectangle selection
- ellipse selection
- magic wand or flood-fill selection
- automatic segmentation
- AI-assisted mask generation
- direct changes to `LLS Simple Repair Prepare`
- direct changes to `LLS Simple Repair Finish`
- direct changes to `LLS Simple KSampler`
- guaranteed pre-execution pixel preview for every possible upstream `IMAGE` producer

Undo/redo history will remain session-only. Workflow reopen restores the current mask result, not the full action timeline.

## Chosen Architecture

Add a new feature package and one frontend extension:

- `mask_draw/__init__.py`
- `mask_draw/node.py`
- `mask_draw/utils.py`
- `web/js/lls_mask_draw.js`

The root package will export:

- `WEB_DIRECTORY = "./web"`

The root package loader will also append `mask_draw` to `_SUBPACKAGES` so registration remains centralized and consistent with the current plugin structure.

This keeps:

- backend node code isolated from `repair/`
- frontend interaction isolated from Python execution code
- registration consistent with the existing package layout

## Node Contract

### Node Name

- registration key: `LLSSimpleMaskDraw`
- display name: `LLS Simple Mask Draw`
- category: `LLS/Image Repair`

### Inputs

Required:

- `image: IMAGE`

Optional:

- `input_mask: MASK`

### Widget Parameters

Visible widgets:

- `draw_mode: brush | erase`
- `brush_size: INT`
- `brush_softness: FLOAT`
- `overlay_alpha: FLOAT`
- `invert_mask: BOOLEAN`

Hidden persistence widget:

- `mask_state_json: STRING`

Hidden runtime widget:

- `node_id: UNIQUE_ID`

### Outputs

- `image: IMAGE`
- `mask: MASK`
- `preview_image: IMAGE`

### Execution Semantics

- `image` is passed through unchanged
- `mask` is the final normalized mask in `[0, 1]`
- `preview_image` is created on the backend by compositing a red semi-transparent overlay over the original image

## State Model

The frontend will persist the current edited mask into `mask_state_json`.

Versioned JSON payload:

```json
{
  "version": 1,
  "mask_png_base64": "iVBORw0K...",
  "touched": true,
  "editor": {
    "draw_mode": "brush",
    "brush_size": 32,
    "brush_softness": 0.5,
    "overlay_alpha": 0.4
  }
}
```

### State Rules

- `touched = false` means the user has not produced a local mask result yet
- `touched = true` means the node owns a concrete local mask result
- `mask_png_base64` stores the current final mask as a grayscale PNG
- `editor` mirrors the last meaningful UI values for restoration convenience

### Persistence Strategy

Persist:

- current mask raster
- `draw_mode`
- `brush_size`
- `brush_softness`
- `overlay_alpha`

Do not persist:

- undo stack
- redo stack
- temporary pointer state
- transient viewport zoom/pan state

This satisfies the requirement to recover the current drawing while keeping workflow size bounded and the format simple.

## Frontend Design

The frontend extension will register with `app.registerExtension(...)` and only target `LLSSimpleMaskDraw`.

### Editor Layout

The node UI will render:

- a scaled image preview area
- a same-size mask editing canvas layered on top
- red semi-transparent overlay for the current visible mask result
- action buttons:
  - `Clear`
  - `Undo`
  - `Redo`
  - `Invert`

Visible parameter widgets remain standard ComfyUI widgets. The canvas editor listens to those widget values rather than replacing them.

### Drawing Model

The frontend maintains:

- one offscreen full-resolution mask canvas as the source of truth
- one scaled display canvas for node rendering

Brush behavior:

- `brush` paints white into the full-resolution mask
- `erase` paints black into the full-resolution mask
- `brush_softness` controls radial falloff from hard edge to soft edge

The display overlay always renders from the current full-resolution mask, resampled to the display canvas.

### Undo/Redo Model

Undo/redo stores snapshots of the raster mask in memory only.

Rules:

- push a snapshot at the end of each stroke
- cap history length at a conservative fixed limit such as `20`
- `Undo` and `Redo` must no-op safely when history is exhausted

### Clear Behavior

`Clear` writes an explicit all-black local mask, sets `touched = true`, clears redo history, and updates persistence.

This is intentionally different from "revert to `input_mask`". The backend must therefore output a black mask after clear.

### Invert Behavior

The `Invert` button will toggle the standard `invert_mask` widget instead of destructively rewriting the stored raster.

This avoids:

- double inversion drift
- mismatches between frontend preview and backend output
- unnecessary rewrites of saved mask data

The frontend overlay preview applies the same invert flag that the backend will apply to the final mask.

## Frontend Initialization Strategy

The frontend initializes in three ordered tiers.

### Tier A: Direct image source recovery

If the upstream image source is a file-backed load node such as `Load Image` or `Load Image Output`, the extension resolves the selected file and loads it through ComfyUI's image view endpoint.

This is the preferred path because it allows immediate editing before execution.

### Tier B: Local persisted mask recovery

If `mask_state_json` already contains a saved raster mask, restore it immediately even if the base image is not yet available.

This ensures workflow reopen does not lose the user's manual mask.

### Tier C: Execution-assisted fallback

If the upstream `image` is produced by an arbitrary intermediate node and no direct source can be resolved, the editor falls back to showing a prompt that the graph should be executed once so the node preview can initialize from executed output.

This is an explicit first-version limitation, not a silent failure.

## `input_mask` Merge Model

The node must support continued editing over an existing `input_mask`.

Behavior is intentionally stateful:

- if the node has no local edited mask yet:
  - use `input_mask` as the editable base when available
- if the node already has a saved local mask:
  - prefer the local mask over `input_mask`

Rationale:

- first open with an existing `input_mask`: user can keep editing from it
- workflow reopen after manual edits: user edits must not be overwritten by a changed or unavailable `input_mask`

This matches the user's priority on preserving manual work.

## Backend Design

`mask_draw/node.py` owns the node class and orchestration.

`mask_draw/utils.py` owns normalization and conversion helpers.

### Responsibilities in `utils.py`

- validate `image` shape
- decode `mask_state_json`
- decode base64 PNG into a standard `MASK`
- create an all-black mask matching image size
- resize `input_mask` to image size when needed
- normalize mask values to `[0, 1]`
- apply `invert_mask`
- build `preview_image`

Where reasonable, backend helpers should reuse the conventions already present in `repair/repair_utils.py` instead of inventing a conflicting mask-processing path.

### Final Mask Resolution Rules

The backend resolves the outgoing mask with this precedence:

1. if `mask_state_json.touched` is true and `mask_png_base64` decodes successfully:
   use the saved local mask
2. else if `input_mask` exists:
   resize and normalize `input_mask`
3. else:
   use an all-black mask

After that:

- if `invert_mask` is true, invert the resolved mask
- clamp to `[0, 1]`
- guarantee output size matches input image size

### Empty and Clear Cases

- untouched node + `input_mask` present => output `input_mask`
- untouched node + no `input_mask` => output all black
- cleared node => output all black

The cleared case is distinguished from untouched by `touched = true`.

### Preview Image Rules

`preview_image` is built on the backend from:

- the original `image`
- the final resolved mask after inversion
- fixed overlay color red
- overlay opacity from `overlay_alpha`

Backend preview generation ensures:

- stable behavior in tests
- stable downstream output
- no dependency on frontend rendering quirks

## Data Flow

### Authoring Flow

1. user connects `image`
2. optional `input_mask` is used as the initial editable base when no local mask exists
3. user paints and erases on the frontend canvas
4. frontend updates `mask_state_json`
5. workflow save serializes `mask_state_json`

### Execution Flow

1. backend receives `image`, optional `input_mask`, visible widget values, and `mask_state_json`
2. backend resolves the final mask using the precedence rules above
3. backend emits:
   - passthrough `image`
   - resolved `mask`
   - generated `preview_image`

## Edge Cases

### Missing `image`

Raise a clear runtime error.

### `input_mask` size mismatch

Automatically resize to the input image size before use.

### No drawing and no `input_mask`

Output a black mask.

### No drawing but `input_mask` exists

Output the resized `input_mask`.

### Clear after prior drawing

Output a black mask, not the original `input_mask`.

### Large images

The frontend display may scale down, but the internal mask source and backend output must remain full resolution.

### Repeated undo/redo

Never throw because history is empty or exhausted.

### Invalid or corrupt persisted mask payload

Fail soft:

- ignore the persisted raster
- fall back to `input_mask` if present
- otherwise fall back to black

This avoids breaking workflow load because of a malformed saved string.

## File-Level Responsibilities

### `mask_draw/__init__.py`

- merge and expose `NODE_CLASS_MAPPINGS`
- merge and expose `NODE_DISPLAY_NAME_MAPPINGS`

### `mask_draw/node.py`

- define `LLSSimpleMaskDraw`
- define the node schema
- coordinate state resolution
- return `image`, `mask`, and `preview_image`

### `mask_draw/utils.py`

- decoding and normalization helpers
- mask/image resize helpers
- preview compositing helpers

### `web/js/lls_mask_draw.js`

- register the frontend extension
- attach the editor UI to the node
- manage stroke input
- manage session-only undo/redo
- persist `mask_state_json`
- restore visible editor state on workflow load

### Root `__init__.py`

- append `mask_draw` to `_SUBPACKAGES`
- export `WEB_DIRECTORY = "./web"`

## Testing Strategy

### Automated Backend Tests

Follow the current repository style:

- `unittest`
- fake tensor/mask helpers where possible

Add tests for:

- plugin registration contains `LLSSimpleMaskDraw`
- display name is `LLS Simple Mask Draw`
- schema matches required and optional ports
- untouched node falls back to `input_mask`
- untouched node without `input_mask` returns black mask
- touched node uses persisted mask state
- clear semantics return black mask
- `invert_mask` flips the final mask
- mismatched `input_mask` is resized
- `preview_image` is produced with the expected shape
- minimal interoperability with `LLS Simple Repair Prepare`

### Manual Frontend Tests

At minimum verify:

- node appears in the add-node menu
- image preview is visible for file-backed sources
- `brush` paints correctly
- `erase` removes mask correctly
- `Clear` resets to black
- `Undo` and `Redo` are stable
- overlay alpha changes the preview
- workflow save and reopen restores the current mask result
- `Load Image -> LLS Simple Mask Draw -> Preview Image`
- `Load Image -> LLS Simple Mask Draw -> LLS Simple Repair Prepare`

## README Changes

Update `README.md` with:

- new node listing: `LLS Simple Mask Draw`
- purpose and positioning in the repair workflow
- input and output descriptions
- how to draw/erase/clear a mask
- how to connect it to `LLS Simple Repair Prepare`
- example uses:
  - manual removal area
  - manual repair area
  - manual shadow removal area
  - manual local enhancement area

Also document first-version limits:

- no polygon/rectangle/ellipse tools
- no magic wand
- no automatic segmentation
- some non-file upstream images may require one execution before the editor can show the base image

## Deferred Work

The following are explicitly deferred:

- polygon tool
- rectangle tool
- ellipse tool
- feather-only edit tools independent of brush softness
- lasso/magic wand selection
- multi-layer mask editing
- direct editing of arbitrary computed upstream `MASK` data before execution
- full persisted undo/redo history
- richer overlay color options

## Acceptance Summary

The feature is complete for this phase when:

- `LLS Simple Mask Draw` registers correctly
- the frontend provides a real interactive mask editor
- brush/erase/clear/undo/redo work in-session
- the node outputs correct `mask` and `preview_image`
- the node can feed `LLS Simple Repair Prepare`
- saved workflows reopen with the current mask result restored
- the README documents usage and current limits clearly
