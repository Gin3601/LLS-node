# LLS Pro Model Profile Routing Design

## Context

`LLS-node` now has a working `pro_edit/` feature family with three nodes:

- `LLS Pro Image Edit Prepare`
- `LLS Pro KSampler Bridge`
- `LLS Pro Image Edit Finish`

The current implementation already separates `SDXL` and `FLUX` professional edit backends, but the routing contract is still too inference-heavy:

- the loader infers a family and lightweight capability tags
- `pro_edit` resolves a backend from `family + role + capability flags`
- `role` often comes from checkpoint-name keyword guessing

This works for a narrow set of obvious names such as `inpaint`, `edit`, `fill`, and now `kontext`, but it does not scale well to real-world model collections:

- many `SD1.5` or `SDXL` derivatives share the same sampling core but differ in edit support
- many `FLUX`-family edit models use different naming conventions
- capability inference is duplicated across loader, routing, and bridge layers
- each node still partially reinterprets the model instead of reading a single authoritative profile

The result is exactly the current user pain:

- a model may clearly belong to `FLUX` or `SDXL`, but still fail the Pro workflow because `model_role` was inferred as `base`
- onboarding a new model often means adding more filename aliases
- routing logic becomes harder to trust as more families and edit variants are added

## Goals

- Keep the existing node names and primary ports:
  - `LLS Pro Image Edit Prepare`
  - `LLS Pro KSampler Bridge`
  - `LLS Pro Image Edit Finish`
- Preserve existing workflow wiring as much as possible.
- Introduce a single explicit internal `ModelProfile` contract that becomes the authoritative source for:
  - loader behavior
  - backend routing
  - sampler routing
- Stop treating checkpoint-name guessing as the main routing truth.
- Support large model families by adapting once per model type, not once per individual checkpoint.
- Keep the current `LLSSimple*` repair chain untouched and separate from the Pro chain.
- Keep auto-routing, but make it profile-driven instead of keyword-driven.
- Preserve manual override capability for debugging and exceptional models.

## Non-Goals

This refactor will not:

- rename the existing Pro nodes
- force users to rebuild current Pro workflows from scratch
- replace the current `LLSSimple*` chain with the Pro chain
- add every future model backend in this iteration
- implement a full external YAML or JSON profile-management UI
- promise that arbitrary unknown edit models can work with zero metadata

## Current Problem Statement

The current Pro path has three architectural weaknesses:

### 1. Inference Is Not Centralized

`model_loader`, `utils/model_info.py`, `pro_edit/backends/registry.py`, and `pro_edit/pro_edit_bridge.py` all participate in deciding what the model is. This creates overlapping heuristics and weak ownership.

### 2. Capability Flags Are Too Low-Level To Be The Main Contract

Flags such as:

- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

are useful, but they are not enough to fully describe execution behavior. They do not explicitly tell the pipeline:

- which backend implementation to use
- which sampler strategy to apply
- which loader assumptions are valid

### 3. Model Family Is Not The Same As Execution Strategy

A family such as `SDXL` or `FLUX_DEV` only tells part of the story.

Examples:

- `SDXL base` and `SDXL inpaint` can share core diffusion family defaults but should not share the same professional edit backend eligibility.
- `FLUX base` and `FLUX Kontext/edit` can share the same broad family while requiring different Pro-routing behavior.
- `SD1.5` belongs in the same unified profile system, but most `SD1.5` models should still route away from the current Pro backend set and remain on Simple workflows.

## Chosen Architecture

Introduce an explicit `ModelProfile` layer and make it the single authoritative routing contract.

The new execution model is:

1. The loader resolves a `ModelProfile`.
2. The loader writes the resolved profile into `_lls_*` metadata on `model`, `clip`, and `vae`.
3. `LLS Pro Image Edit Prepare` reads the resolved profile and dispatches by `backend_type`.
4. `LLS Pro KSampler Bridge` reads the same resolved profile and dispatches by `sampler_strategy`.
5. `LLS Pro Image Edit Finish` continues to composite primarily by `edit_scope`, but can also read `backend_type` for backend-specific finish behavior later.

Checkpoint-name inference remains as a fallback source of evidence, but not as the primary truth source once a profile has been resolved.

## Model Profile Contract

Create a unified internal profile structure with these fields:

- `profile_id`
- `family`
- `role`
- `backend_type`
- `sampler_strategy`
- `loader_strategy`
- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

### Field Meanings

- `profile_id`
  - stable identifier for the resolved profile, such as `sdxl_base`, `sdxl_inpaint`, `flux_edit`
- `family`
  - broad model family, for example `SD1.5`, `SDXL`, `FLUX_DEV`
- `role`
  - semantic model role, such as `base`, `inpaint`, `edit`, `fill`
- `backend_type`
  - Pro backend routing target, for example `none`, `sdxl_native`, `flux_edit`
- `sampler_strategy`
  - bridge-time sampling strategy, for example `standard_k`, `flux_guided`
- `loader_strategy`
  - loader-time resource strategy, for example `sd15_checkpoint`, `sdxl_checkpoint`, `flux_split_or_bundle`
- `supports_inpaint_native`
  - whether the model can consume native inpaint-style conditioning
- `supports_image_edit_native`
  - whether the model can consume native image-edit-style conditioning
- `preferred_edit_backend`
  - compatibility hint for UI, migration, and legacy metadata interop

## Built-In Profiles In This Iteration

The first refactor will ship with built-in profiles for the currently supported scope:

- `sd15_base`
- `sdxl_base`
- `sdxl_inpaint`
- `sdxl_edit`
- `flux_base`
- `flux_edit`

### Intended Behavior

- `sd15_base`
  - stays outside the current Pro backend set
  - keeps using normal/simple workflows unless future SD1.5 native edit support is added
- `sdxl_base`
  - does not enter the current Pro backend set by default
- `sdxl_inpaint`
  - maps to `backend_type = sdxl_native`
  - maps to `sampler_strategy = standard_k`
- `sdxl_edit`
  - maps to `backend_type = sdxl_native`
  - maps to `sampler_strategy = standard_k`
- `flux_base`
  - does not enter the current Pro backend set by default
- `flux_edit`
  - maps to `backend_type = flux_edit`
  - maps to `sampler_strategy = flux_guided`

This structure ensures that many model variants can share one profile instead of requiring one-off code changes per checkpoint.

## Profile Resolution Order

Profile resolution must be deterministic and centralized.

Resolution order:

1. explicit `model_info` overrides
2. existing `_lls_*` profile metadata already attached to the runtime objects
3. built-in registry rules using:
   - family
   - known naming patterns
   - capability hints
4. fallback base profile for the detected family

### Manual Override Behavior

Manual override remains supported through `model_info`, not through new required ports.

The following keys should be accepted when present in `model_info`:

- `profile_id`
- `family`
- `role`
- `backend_type`
- `sampler_strategy`
- `loader_strategy`
- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

This keeps the current node surface stable while still allowing explicit correction when auto-resolution is wrong.

## File Structure Changes

Add a new package:

- `model_profiles/__init__.py`
- `model_profiles/base.py`
- `model_profiles/registry.py`
- `model_profiles/rules.py`

### Responsibilities

- `model_profiles/base.py`
  - define the profile record contract and normalization helpers
- `model_profiles/rules.py`
  - define built-in family and pattern matching rules
- `model_profiles/registry.py`
  - resolve the final `ModelProfile` from runtime inputs and overrides
- `model_profiles/__init__.py`
  - export the public profile-resolution helpers

## Existing File Changes

### `utils/model_info.py`

Keep this file as the low-level family and metadata helper layer, but reduce its authority:

- keep family inference helpers
- keep JSON-like parsing helpers
- keep backwards-compatible accessors
- stop treating capability inference as the final routing truth
- add profile-aware wrappers where needed

This file becomes a supporting utility for profile resolution instead of the main owner of routing semantics.

### `model_loader/nodes.py`

`LLSSimpleCheckpointLoader` becomes the authoritative profile stamping point.

It must:

- resolve family as it does today
- ask the new profile registry for the final `ModelProfile`
- tag `model`, `clip`, and `vae` with:
  - `_lls_profile_id`
  - `_lls_backend_type`
  - `_lls_sampler_strategy`
  - `_lls_loader_strategy`
  - `_lls_model_role`
  - `_lls_supports_inpaint_native`
  - `_lls_supports_image_edit_native`
  - `_lls_preferred_edit_backend`

The loader remains family-aware for text encoder and VAE resolution, but those behaviors should now align with explicit `loader_strategy` values instead of only raw family checks.

### `pro_edit/backends/registry.py`

This file must stop doing broad capability inference.

Its new responsibility is:

- read resolved profile data
- validate `backend_mode` overrides against `backend_type`
- route to the correct backend implementation

New routing logic should be driven by `backend_type`, not by repeated filename or role guessing.

### `pro_edit/pro_edit_prepare.py`

This node should:

- resolve the effective profile
- reject profiles whose `backend_type` is `none`
- dispatch strictly by `backend_type`
- keep current ports and return values

### `pro_edit/pro_edit_bridge.py`

This node should:

- resolve the same effective profile
- dispatch strictly by `sampler_strategy`
- keep current ports and return values
- continue using the shared low-level `_common_ksampler` where valid
- keep FLUX-specific guidance handling behind the `flux_guided` strategy instead of ad-hoc family checks

### `pro_edit/pro_edit_finish.py`

This node can remain primarily `edit_scope`-based in this iteration, but it should:

- normalize and preserve `backend_type` in `edit_info`
- leave explicit extension points for future backend-specific finish behavior

## Runtime Data Flow

### Happy Path

1. User loads a model through `LLS Simple Checkpoint Loader`.
2. Loader resolves family and final profile.
3. Loader writes profile metadata to `model`, `clip`, and `vae`.
4. `LLS Pro Image Edit Prepare` reads profile metadata and chooses the backend via `backend_type`.
5. Backend-specific prepare logic builds native edit conditioning.
6. `LLS Pro KSampler Bridge` reads the same profile and applies the right `sampler_strategy`.
7. `LLS Pro Image Edit Finish` composites the generated output according to `edit_scope`.

### Override Path

1. Auto profile resolution produces an incorrect result or a generic base profile.
2. User supplies `model_info` override fields.
3. Profile registry merges those overrides into the final profile.
4. Pro nodes use the overridden explicit profile instead of fallback inference.

## Simple vs Pro Separation

The separation must become stricter, not weaker.

- `Simple` remains the compatibility-first path for broad masked resampling.
- `Pro` remains the native local edit path for true edit/inpaint-capable profiles.

The Pro chain should not silently downgrade to the Simple chain when a profile is `base` or unsupported. It should fail with a clear error that explains:

- the detected family
- the detected profile
- the missing backend eligibility

This keeps workflow behavior predictable.

## Error Handling

The new profile-driven path should improve the error messages.

### Unsupported Profile For Pro Chain

If `backend_type = none`, raise an error such as:

> `[LLS] Pro image edit is not available for profile '<profile_id>' (family '<family>', role '<role>'). Use a supported native edit/inpaint profile or provide an explicit override.`

### Invalid Manual Override

If the user forces `backend_mode = flux` but the resolved profile maps to `sdxl_native`, raise a clear mismatch error instead of trying to proceed.

### Incomplete Override

If the user provides a partial override that produces an invalid combination, the profile registry should normalize or reject it immediately instead of letting the mismatch leak deeper into the pipeline.

## Compatibility Strategy

The refactor must preserve current workflow usability:

- existing node names stay unchanged
- existing ports remain valid
- `model_info` remains the optional override surface
- old `_lls_*` capability tags continue to be read during migration

During migration, the profile resolver may derive `backend_type` and `sampler_strategy` from old capability fields when a full explicit profile is not yet present.

This prevents a hard break between the current Pro implementation and the new profile-driven contract.

## Testing Strategy

Add focused tests for the new architecture in these categories:

### Profile Resolution Tests

- family base profiles resolve correctly
- explicit `model_info` overrides take precedence
- legacy capability tags still map to the correct profile during migration
- common alias patterns such as `kontext` resolve to the intended edit profile

### Loader Tagging Tests

- loader writes the new `_lls_profile_*` metadata
- loader keeps writing the existing compatibility fields

### Backend Routing Tests

- `backend_type = sdxl_native` routes to the SDXL backend
- `backend_type = flux_edit` routes to the FLUX backend
- `backend_type = none` fails clearly in Pro prepare
- forced backend mismatches are rejected clearly

### Sampler Strategy Tests

- `sampler_strategy = standard_k` uses the shared standard bridge path
- `sampler_strategy = flux_guided` injects FLUX guidance behavior

### Regression Tests

- current Pro SDXL tests still pass
- current Pro FLUX tests still pass
- current `LLSSimple*` repair tests still pass

## Recommended Migration Outcome

After this refactor:

- adding a new model should usually mean adding or adjusting a profile rule
- adding a truly new model type should mean adding a new profile plus backend or sampler strategy
- neither path should require scattering new filename heuristics across loader, registry, and bridge code

This is the core success criterion for the refactor.
