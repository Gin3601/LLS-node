# LLS Simple Empty Latent Unified Entry Design

## Context

`LLSSimpleEmptyLatent` is currently a `txt2img`-only node. `img2img` requires a separate `LLSSimpleVAEEncode` node. The requested change is to keep the existing node name and outputs, but let it handle both modes:

- no `image` input: behave like the current empty latent generator for `txt2img`
- `image` input connected: encode the image into latent space for `img2img`

The user also wants the existing size controls to remain meaningful in both modes.

## Chosen Approach

Extend `LLSSimpleEmptyLatent` into a unified latent entry node.

- Keep the current outputs: `latent`, `width`, `height`, `latent_info`
- Keep current `txt2img` behavior as the default when no image is connected
- Add optional `image` and `vae` inputs plus a `resize_mode` control
- Reuse the same resize and VAE encode rules already used by `LLSSimpleVAEEncode`
- Continue using the current `size_preset`, `width`, and `height` controls in both modes

This avoids changing downstream nodes while making the node usable as a single entry point.

## Node Interface Changes

`LLSSimpleEmptyLatent` will gain:

- optional `image: IMAGE`
- optional `vae: VAE`
- required `resize_mode` matching `LLSSimpleVAEEncode`

The existing optional `model` input stays in place because it is still useful for family inference when `model_family=Auto`.

No output names or output types change.

## Runtime Behavior

### txt2img path

When `image` is not connected:

- preserve current behavior
- resolve family from `model_family` and optional `model`
- compute width/height from `size_preset`
- allocate zero latent tensor
- return `source=empty_latent` and `task_mode=txt2img`

### img2img path

When `image` is connected:

- require `vae`
- resolve family from `model_family`, `model`, and existing family inference helpers
- compute requested size from `size_preset`
- apply `resize_mode` using the same semantics as `LLSSimpleVAEEncode`
- encode the processed image through `vae.encode`
- return `source=image_encode` and `task_mode=img2img`

In `img2img`, the `batch_size` control is ignored because the batch comes from the input image tensor.

## Size Rules

`size_preset`, `width`, and `height` remain active for both modes:

- `Family Default`: use the resolved family default width and height
- explicit preset like `1024x1024`: use that preset
- `Custom`: use the entered `width` and `height`

Then apply the same latent alignment rules already used in the project.

For `img2img`, the image is resized to the resolved target size according to `resize_mode`.

## Compatibility

- Existing `txt2img` workflows keep working without rewiring
- Existing downstream `KSampler` logic keeps working because it already distinguishes `txt2img` and `img2img` from the latent `source`
- `LLSSimpleVAEEncode` stays available for users who still want a dedicated encode node

## Error Handling

- If `image` is connected but `vae` is missing, raise a clear error
- If `resize_mode` is invalid, keep existing validation behavior
- If VAE encode fails, surface a wrapped `RuntimeError`

## Testing

Add or update tests for:

- schema: new optional `image` and `vae` inputs plus `resize_mode`
- `txt2img` compatibility: existing behavior remains unchanged
- `img2img` path with `Family Default`
- `img2img` path with `Custom` sizing
- `img2img` metadata: `task_mode=img2img`, `latent_source=image_encode`, resolved width/height
- missing `vae` on `img2img` path raises the expected error
