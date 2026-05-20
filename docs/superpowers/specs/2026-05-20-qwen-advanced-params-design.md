# Qwen Advanced Parameters Design

## Context

The current Qwen integration already provides two compressed high-level nodes:

- `LLSQwenTextToImage`
- `LLSQwenImageEdit`

Those nodes intentionally hide `model`, `clip`, `vae`, latent setup, and sampler wiring, and only return `IMAGE`.

That minimal v1 is useful, but it leaves out the official Qwen workflow controls that users commonly adjust in practice, especially:

- negative prompt
- cfg
- sampler
- scheduler
- model shift
- multi-image edit inputs
- edit-specific conditioning controls
- lightning/turbo LoRA toggles

The goal of this design is to add those advanced controls without breaking the compressed single-node UX.

## Goal

Extend the existing Qwen nodes so they expose official/common advanced parameters while still:

- hiding resource loading
- outputting only `IMAGE`
- keeping text-to-image and image-edit as separate nodes
- preserving clear runtime validation

## Scope

### In Scope

- Add advanced prompt/sampling controls to `LLSQwenTextToImage`
- Add advanced prompt/sampling/edit controls to `LLSQwenImageEdit`
- Add optional `image2` and `image3` to the image-edit node
- Add optional ordered user LoRA stack input to both Qwen nodes
- Add one compact helper node for building an ordered Qwen LoRA stack
- Add optional turbo/lightning LoRA mode to both nodes
- Auto-match compatible turbo LoRAs when requested
- Reject incompatible turbo LoRA selections explicitly
- Add test coverage for schema, parameter passthrough, turbo behavior, and failure paths

### Out of Scope

- Exposing `clip_name`, `vae_name`, or `unet_name`
- Returning extra outputs such as `info`, `model`, `vae`, or `latent`
- Merging Qwen into the generic `LLSSimpleCheckpointLoader -> PromptEncode -> KSampler` chain
- Extending Hunyuan, ZImage, or other model families in the same change
- Adding layered Qwen support
- Frontend visual redesign beyond the required input schema changes

## Chosen Approach

Keep the current dedicated `qwen/` feature family and expand it in place.

The nodes stay compressed and high-level, but their input schemas grow to expose the controls that matter most in the official Qwen workflows. Resource loading remains internal. Sampling remains internal. The advanced parameters only steer the internal runtime wrappers.

This keeps the current user experience intact:

- users still drag one node for text-to-image
- users still drag one node for image edit
- users can optionally connect one compact ordered LoRA stack input
- advanced users gain the controls they expect
- the plugin still avoids forcing Qwen through the generic SD/SDXL/FLUX abstractions

## User-Facing Nodes

### `LLSQwenTextToImage`

**Base Inputs**

- `model_name`
- `prompt`
- `width`
- `height`
- `steps`
- `seed`
- `batch_size`

**Advanced Inputs**

- `negative_prompt`
- `cfg`
- `sampler_name`
- `scheduler`
- `shift`
- `lora_stack` (optional input)
- `enable_turbo_mode`
- `turbo_lora_name`
- `turbo_strength`

**Output**

- `IMAGE`

**Behavior**

- Loads the selected Qwen text-to-image model internally
- Resolves the Qwen text encoder and VAE internally
- Encodes `prompt` as positive conditioning
- Encodes `negative_prompt` as negative conditioning
- Passes `cfg`, `sampler_name`, and `scheduler` directly into internal sampling
- Passes `shift` directly into `ModelSamplingAuraFlow`
- Creates the latent internally from `width`, `height`, and `batch_size`
- Applies any connected ordered user LoRA stack to the internal model before turbo handling
- Optionally loads a compatible turbo/lightning LoRA when turbo mode is enabled
- Returns only the final decoded `IMAGE`

### `LLSQwenImageEdit`

**Base Inputs**

- `model_name`
- `image`
- `prompt`
- `steps`
- `seed`

**Advanced Inputs**

- `image2`
- `image3`
- `negative_prompt`
- `cfg`
- `sampler_name`
- `scheduler`
- `shift`
- `cfg_norm_strength`
- `reference_latents_method`
- `lora_stack` (optional input)
- `enable_turbo_mode`
- `turbo_lora_name`
- `turbo_strength`

**Output**

- `IMAGE`

**Behavior**

- Loads the selected Qwen image-edit model internally
- Resolves the Qwen text encoder and VAE internally
- Uses `image` as the required primary edit input
- Optionally passes `image2` and `image3` into the official `TextEncodeQwenImageEditPlus` path
- Encodes `prompt` as positive edit conditioning
- Encodes `negative_prompt` as negative edit conditioning
- Applies the same image reference set to both positive and negative edit conditioning
- Passes `cfg`, `sampler_name`, and `scheduler` directly into internal sampling
- Passes `shift` into `ModelSamplingAuraFlow`
- Passes `cfg_norm_strength` into `CFGNorm`
- Passes `reference_latents_method` into `FluxKontextMultiReferenceLatentMethod`
- Applies any connected ordered user LoRA stack to the internal model before turbo handling
- Optionally loads a compatible turbo/lightning LoRA when turbo mode is enabled
- Returns only the final decoded `IMAGE`

### `LLSQwenLoRAStack`

**Base Inputs**

- `lora_name`
- `strength_model`

**Optional Input**

- `lora_stack`

**Output**

- `LLS_QWEN_LORA_STACK`

**Behavior**

- Builds an ordered model-only LoRA stack for the Qwen nodes
- Supports serial chaining by accepting an existing `lora_stack` and appending one more LoRA entry
- Preserves user-specified order exactly
- Keeps the main Qwen nodes compressed by moving multi-LoRA assembly into one small helper node

## Advanced Parameter Defaults

### Shared Defaults

- `steps = 20`
- `cfg = 4.0`
- `sampler_name = "euler"`
- `scheduler = "simple"`
- `shift = 3.1`
- `enable_turbo_mode = false`
- `turbo_lora_name = "(auto)"`
- `turbo_strength = 1.0`

### Text-to-Image Specific Defaults

- `negative_prompt = ""`
- `batch_size = 1`

### Image Edit Specific Defaults

- `image2 = None`
- `image3 = None`
- `negative_prompt = ""`
- `cfg_norm_strength = 1.0`
- `reference_latents_method = "index_timestep_zero"`

### Default Prompt Policy

The nodes must not silently inject model-specific negative prompts.

Even though some official blueprints ship with built-in negative prompt text, this design keeps the default `negative_prompt` empty. Prompt behavior stays explicit and user-controlled.

## Turbo / Lightning Design

### User Inputs

Both nodes will expose:

- `enable_turbo_mode`
- `turbo_lora_name`
- `turbo_strength`

These inputs control optional lightning/turbo LoRA application while keeping the main resource graph hidden.

### Runtime Rules

When `enable_turbo_mode = false`:

- do not load any turbo/lightning LoRA
- ignore `turbo_lora_name`
- ignore `turbo_strength`
- use the user-provided `steps`, `cfg`, `sampler_name`, `scheduler`, and `shift`

When `enable_turbo_mode = true`:

- find a compatible turbo/lightning LoRA for the selected main model
- if `turbo_lora_name = "(auto)"`, auto-select the matching LoRA
- if `turbo_lora_name` is manually set, validate it against the selected main model
- keep user LoRA stack application order separate from turbo handling
- apply that LoRA to the internal model only
- use `turbo_strength` as the LoRA strength
- override runtime `steps` and `cfg` with turbo presets
- keep `sampler_name`, `scheduler`, and `shift` user-controlled

### Ordered LoRA Rules

When a `lora_stack` input is connected:

- treat it as an ordered serial chain
- apply entries in the exact order they appear in the stack
- use `LoraLoaderModelOnly` semantics because these Qwen nodes keep `clip` hidden
- apply the full user stack before optional turbo/lightning LoRA application

This yields the internal order:

- base Qwen model
- user LoRA #1
- user LoRA #2
- ...
- optional turbo/lightning LoRA

### Compatibility Strategy

#### `LLSQwenTextToImage`

- Accept only Qwen text-to-image turbo/lightning LoRAs
- `qwen_image_2512_*` models prefer `Qwen-Image-2512-Lightning-*`
- non-2512 text-to-image models prefer `Qwen-Image-Lightning-*`

#### `LLSQwenImageEdit`

- Accept only Qwen image-edit turbo/lightning LoRAs
- `qwen_image_edit_2509_*` models prefer `Qwen-Image-Edit-2509-Lightning-*`
- `qwen_image_edit_2511_*` models may use `Qwen-Image-Edit-2511-Lightning-*` when that LoRA is present locally

### Turbo Presets

When turbo mode is enabled, the runtime should replace user-entered `steps` and `cfg` with the following presets:

- `qwen_image_fp8_e4m3fn.safetensors` -> `steps = 8`, `cfg = 1.0`
- `qwen_image_2512_fp8_e4m3fn.safetensors` -> `steps = 4`, `cfg = 1.0`
- `qwen_image_edit_2509_fp8_e4m3fn.safetensors` -> `steps = 4`, `cfg = 1.0`
- `qwen_image_edit_2511_*` -> `steps = 4`, `cfg = 1.0`

If the selected model has no supported turbo profile, turbo mode must fail with a clear error.

### Failure Policy

Do not silently disable turbo mode.

If turbo mode is enabled and any turbo requirement is missing or incompatible, fail explicitly.

Required explicit failures:

- turbo mode enabled but no compatible turbo LoRA exists
- turbo mode enabled but no compatible turbo preset exists
- manually selected turbo LoRA is incompatible with the selected main model
- requested turbo LoRA file is missing

## Runtime Parameter Mapping

### Text-to-Image Mapping

- `prompt` -> positive `CLIPTextEncode`
- `negative_prompt` -> negative `CLIPTextEncode`
- `cfg` -> `KSampler.cfg`
- `sampler_name` -> `KSampler.sampler_name`
- `scheduler` -> `KSampler.scheduler`
- `shift` -> `ModelSamplingAuraFlow.shift`
- `batch_size` -> internal empty latent creation

### Image Edit Mapping

- `image` -> required `image1`
- `image2`, `image3` -> optional secondary references
- `prompt` -> positive `TextEncodeQwenImageEditPlus`
- `negative_prompt` -> negative `TextEncodeQwenImageEditPlus`
- positive and negative conditioning both receive the same image reference set
- `cfg` -> `KSampler.cfg`
- `sampler_name` -> `KSampler.sampler_name`
- `scheduler` -> `KSampler.scheduler`
- `shift` -> `ModelSamplingAuraFlow.shift`
- `cfg_norm_strength` -> `CFGNorm.strength`
- `reference_latents_method` -> `FluxKontextMultiReferenceLatentMethod.reference_latents_method`

## Schema / UX Rules

The nodes should remain compressed and readable.

### Visibility Policy

Keep current base inputs prominent and mark the newly added tuning inputs as advanced where supported by ComfyUI input metadata.

That means:

- the basic experience still looks simple
- users can open advanced controls when needed
- the node is still much smaller than wiring the official subgraph manually

### Output Policy

Do not add extra outputs in this iteration.

Both nodes continue to output only:

- `IMAGE`

## Error Handling

The runtime should continue using fail-fast, explicit errors.

Required error cases:

- selected main model is not compatible with the current Qwen node
- no compatible Qwen main models exist
- required Qwen text encoder is missing
- required Qwen VAE is missing
- primary edit image is missing
- the connected user LoRA stack is malformed
- required ComfyUI runtime component is unavailable
- turbo mode is enabled but no compatible turbo LoRA exists
- turbo mode is enabled but the selected model has no supported turbo preset
- manually selected turbo LoRA is incompatible with the selected main model
- internal Qwen execution fails

The nodes must not silently fall back to:

- generic LLS simple nodes
- non-turbo execution when turbo was explicitly requested
- different model families

## Internal Architecture Changes

Extend the existing `qwen/` package instead of creating a second feature family.

### `qwen/discovery.py`

Add helpers for:

- discovering compatible turbo/lightning LoRAs
- listing user-selectable LoRAs for Qwen stack building
- resolving `(auto)` turbo LoRA matches
- validating manual turbo LoRA selections
- exposing placeholder choices when no compatible turbo LoRAs exist

### `qwen/runtime.py`

Extend runtime wrappers so they accept advanced parameters and:

- pass prompt/sampler controls through to internal Qwen execution
- apply ordered model-only user LoRA stacks
- load optional turbo LoRAs
- apply turbo presets when enabled
- support multi-image edit conditioning
- apply edit-specific conditioning controls

### `qwen/nodes.py`

Expand both node schemas to expose the agreed advanced inputs while preserving:

- only `IMAGE` output
- internal resource encapsulation
- separate text-to-image and image-edit nodes

Also add one compact helper node that outputs `LLS_QWEN_LORA_STACK`.

## Testing

Add or update tests for:

- text-to-image schema includes the new advanced inputs
- image-edit schema includes the new advanced inputs
- LoRA stack helper schema exposes serial chaining
- ordered user LoRA stack entries are applied in order
- negative prompt is routed into the negative conditioning path
- `cfg`, `sampler_name`, `scheduler`, and `shift` are passed through correctly
- `image2` and `image3` are routed into edit conditioning
- `cfg_norm_strength` is passed through correctly
- `reference_latents_method` is passed through correctly
- turbo mode loads the expected LoRA for compatible models
- turbo mode overrides runtime `steps` and `cfg` with the expected preset
- turbo mode disabled leaves user-entered `steps` and `cfg` unchanged
- turbo mode fails clearly when no compatible LoRA exists
- turbo mode fails clearly when a manual LoRA selection is incompatible
- full plugin test suite continues to pass

## Non-Goals For This Iteration

Do not add:

- `clip_name`, `vae_name`, or `unet_name` overrides
- metadata outputs
- latent outputs
- model outputs
- generic Qwen support inside the standard LLS simple chain
- Hunyuan or ZImage advanced-node work

## Acceptance Criteria

This iteration is successful when:

- `LLSQwenTextToImage` exposes official/common advanced prompt and sampling controls
- `LLSQwenImageEdit` exposes official/common advanced prompt, sampling, and edit controls
- both nodes accept an optional ordered model-side LoRA stack input
- both nodes still hide model/clip/vae wiring
- both nodes still output only `IMAGE`
- text-to-image turbo mode works for supported Qwen text models
- image-edit turbo mode works for supported Qwen edit models
- unsupported turbo requests fail explicitly instead of silently degrading
- existing Qwen tests and the full plugin test suite continue to pass
