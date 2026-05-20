# Qwen Minimal Nodes Design

## Context

The current LLS simplified generation chain is designed around `SD1.5`, `SDXL`, and `FLUX` style model families. Qwen image models do not fit that contract cleanly, especially around text encoding and the official image edit workflow.

The goal of this design is to add a minimal, user-facing Qwen experience without forcing Qwen through the existing generic `CheckpointLoader -> PromptEncode -> EmptyLatent -> KSampler` pipeline.

## Goal

Add two high-level Qwen nodes that internally encapsulate model loading, text encoding, VAE handling, and execution:

- `LLSQwenTextToImage`
- `LLSQwenImageEdit`

Both nodes should output only `IMAGE` in v1.

## Scope

### In Scope

- Minimal Qwen text-to-image node
- Minimal Qwen official image edit node
- Model list filtering so each node only shows compatible Qwen models
- Internal runtime validation to reject mismatched model selections
- Clear runtime errors for missing resources or incompatible environments

### Out of Scope

- Extending the current generic LLS simple chain to support Qwen
- Additional metadata outputs such as `model`, `vae`, or `info`
- `Qwen-Image-Layered`
- Traditional KSampler-style `img2img` for Qwen
- Advanced parameter exposure such as sampler, scheduler, cfg, or denoise

## Chosen Approach

Implement Qwen as a dedicated feature family with two specialized high-level nodes instead of trying to fold Qwen into the existing generic family abstraction.

This keeps the node surface small for users while avoiding incorrect assumptions baked into the current SD/SDXL/FLUX-oriented loader and prompt encoding flow.

## User-Facing Nodes

### `LLSQwenTextToImage`

**Inputs**

- `model_name`
- `prompt`
- `width`
- `height`
- `steps`
- `seed`
- `batch_size`

**Outputs**

- `IMAGE`

**Behavior**

- Loads the selected Qwen text-to-image model and the resources it needs internally
- Encodes the prompt using the Qwen-compatible text path
- Resolves latent/image size internally from `width` and `height`
- Runs the official text-to-image generation path
- Returns generated `IMAGE`

### `LLSQwenImageEdit`

**Inputs**

- `model_name`
- `image`
- `prompt`
- `steps`
- `seed`

**Outputs**

- `IMAGE`

**Behavior**

- Loads the selected Qwen image edit model and the resources it needs internally
- Uses the input image as the official edit source, not as traditional denoise-based `img2img`
- Encodes the prompt using the Qwen-compatible edit path
- Handles image preprocessing and internal size alignment automatically
- Runs the official Qwen image edit generation path
- Returns edited `IMAGE`

## Model Compatibility Filtering

Each node will expose only models compatible with its task.

### `LLSQwenTextToImage` model list

- Include Qwen text-to-image models only
- Exclude Qwen image edit models
- Exclude Qwen layered models

### `LLSQwenImageEdit` model list

- Include Qwen image edit models only
- Exclude regular Qwen text-to-image models
- Exclude Qwen layered models

### Validation Strategy

Compatibility checks will happen in two layers:

1. Filter the dropdown choices so users usually only see compatible models
2. Re-validate at execution time so renamed files or stale workflows fail early with a clear message

If no compatible models are found, the node should show a placeholder choice and fail with a clear runtime error when executed.

## Internal Architecture

Create a dedicated `qwen/` feature package for these nodes and their helpers.

Suggested responsibilities:

- `qwen/nodes.py`
  - User-facing node classes
- `qwen/runtime.py`
  - Shared execution helpers for loading resources and running Qwen flows
- `qwen/discovery.py`
  - Compatible model discovery and validation helpers

The internal Qwen runtime should be isolated from the current generic LLS model-family logic rather than forcing new special cases into `LLSSimpleCheckpointLoader` and `LLSSimplePromptEncode`.

## Resource Loading Rules

The nodes should encapsulate the resource graph internally.

That means the user does not manually connect:

- model
- clip/text encoder
- vae
- latent nodes
- sampler nodes

The nodes should auto-resolve all required resources for the selected Qwen model and fail clearly if a required resource is missing.

## Error Handling

The nodes should fail early and explicitly.

Required error cases:

- selected model is not compatible with the current Qwen node
- no compatible Qwen models were found
- required text encoder is missing
- required VAE is missing
- required ComfyUI runtime component is unavailable
- Qwen execution path fails internally

The nodes must not silently fall back to the generic LLS simple nodes.

## Testing

Add test coverage for:

- `LLSQwenTextToImage` model dropdown filtering
- `LLSQwenImageEdit` model dropdown filtering
- placeholder behavior when no compatible models exist
- runtime rejection of incompatible model selection
- minimal happy path for `LLSQwenTextToImage`
- minimal happy path for `LLSQwenImageEdit`
- missing-resource errors

## Non-Goals for V1

Do not implement these in the first Qwen release:

- metadata passthrough outputs
- advanced tuning parameters
- unified “one node with mode switch” UX
- Qwen support inside the generic simple loader/prompt/sampler chain
- layered Qwen workflows

## Acceptance Criteria

V1 is successful when:

- users can generate Qwen images from a single text-to-image node
- users can run official Qwen image edit from a single edit node
- each node only shows compatible Qwen models
- wrong model selections fail immediately with clear messages
- no extra loader/clip/vae/sampler wiring is required from the user
