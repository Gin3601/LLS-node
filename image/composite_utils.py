from __future__ import annotations

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None
    _NUMPY_ERR = exc
else:  # pragma: no cover
    _NUMPY_ERR = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_ERR = exc
else:  # pragma: no cover
    _PIL_ERR = None

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_ERR = exc
else:  # pragma: no cover
    _TORCH_ERR = None


ANCHOR_MODE_CHOICES = ["top_left", "center"]
ROTATION_ORIGIN_MODE_CHOICES = ["top_left", "center"]
BLEND_MODE_CHOICES = ["normal"]


def _require_runtime_deps():
    if torch is None:
        raise RuntimeError("[LLS] torch is required for image compositing.") from _TORCH_ERR
    if np is None:
        raise RuntimeError("[LLS] numpy is required for image compositing.") from _NUMPY_ERR
    if Image is None:
        raise RuntimeError("[LLS] Pillow is required for image compositing.") from _PIL_ERR


def _require_image_tensor(name, image):
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        raise RuntimeError(f"[LLS] {name} must have shape [batch, height, width, channels].")
    if int(shape[0]) <= 0 or int(shape[1]) <= 0 or int(shape[2]) <= 0:
        raise RuntimeError(f"[LLS] {name} must have positive batch, width, and height.")
    channels = int(shape[3])
    if channels not in {3, 4}:
        raise RuntimeError(f"[LLS] {name} must have 3 or 4 channels; got {channels}.")


def _tensor_sample_to_rgba_pil(sample):
    rgba = sample.detach().cpu().clamp(0.0, 1.0).numpy()
    if rgba.shape[-1] == 3:
        alpha = np.ones((rgba.shape[0], rgba.shape[1], 1), dtype=np.float32)
        rgba = np.concatenate([rgba, alpha], axis=-1)
    rgba = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _pil_to_tensor(image, *, device, dtype):
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[-1] != 4:
        raise RuntimeError("[LLS] compositing helper expected an RGBA PIL image.")
    rgb = array[..., :3]
    return torch.from_numpy(rgb).to(device=device, dtype=dtype)


def _scaled_overlay_rgba(overlay_rgba, scale):
    width, height = overlay_rgba.size
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    if (scaled_width, scaled_height) == (width, height):
        return overlay_rgba
    return overlay_rgba.resize((scaled_width, scaled_height), Image.BICUBIC)


def _resolve_top_left(width, height, x_offset, y_offset, anchor_mode):
    if anchor_mode == "top_left":
        return int(x_offset), int(y_offset)
    if anchor_mode == "center":
        return int(x_offset - (width / 2.0)), int(y_offset - (height / 2.0))
    raise RuntimeError(f"[LLS] Unsupported anchor_mode '{anchor_mode}'.")


def _resolve_rotation_center(top_left_x, top_left_y, width, height, rotation_origin_mode):
    if rotation_origin_mode == "top_left":
        return float(top_left_x), float(top_left_y)
    if rotation_origin_mode == "center":
        return float(top_left_x) + (width / 2.0), float(top_left_y) + (height / 2.0)
    raise RuntimeError(f"[LLS] Unsupported rotation_origin_mode '{rotation_origin_mode}'.")


def composite_images(
    background_image,
    overlay_image,
    *,
    x_offset,
    y_offset,
    anchor_mode,
    rotation_origin_mode,
    opacity,
    blend_mode,
    scale,
    rotation,
    keep_aspect,
):
    del keep_aspect

    _require_runtime_deps()
    if background_image is None:
        raise RuntimeError("[LLS] background_image is required for image compositing.")
    if overlay_image is None:
        raise RuntimeError("[LLS] overlay_image is required for image compositing.")
    if blend_mode != "normal":
        raise RuntimeError(f"[LLS] Unsupported blend_mode '{blend_mode}'.")

    _require_image_tensor("background_image", background_image)
    _require_image_tensor("overlay_image", overlay_image)
    clamped_scale = max(0.01, float(scale))
    clamped_opacity = max(0.0, min(1.0, float(opacity)))
    if clamped_opacity == 0.0:
        return background_image[..., :3].clone()

    background_batch = int(background_image.shape[0])
    overlay_batch = int(overlay_image.shape[0])
    if overlay_batch not in {1, background_batch}:
        raise RuntimeError(
            "[LLS] overlay_image batch must be 1 or match background_image batch; "
            f"got {overlay_batch} and {background_batch}."
        )

    results = []
    for index in range(background_batch):
        background_sample = background_image[index]
        overlay_sample = overlay_image[0 if overlay_batch == 1 else index]
        background_rgba = _tensor_sample_to_rgba_pil(background_sample)
        overlay_rgba = _scaled_overlay_rgba(_tensor_sample_to_rgba_pil(overlay_sample), clamped_scale)
        top_left_x, top_left_y = _resolve_top_left(
            overlay_rgba.size[0],
            overlay_rgba.size[1],
            int(x_offset),
            int(y_offset),
            anchor_mode,
        )

        overlay_canvas = Image.new("RGBA", background_rgba.size, (0, 0, 0, 0))
        overlay_canvas.paste(overlay_rgba, (top_left_x, top_left_y), overlay_rgba)

        if float(rotation) != 0.0:
            rotation_center = _resolve_rotation_center(
                top_left_x,
                top_left_y,
                overlay_rgba.size[0],
                overlay_rgba.size[1],
                rotation_origin_mode,
            )
            overlay_canvas = overlay_canvas.rotate(
                float(rotation),
                resample=Image.BICUBIC,
                center=rotation_center,
                expand=False,
            )

        if clamped_opacity < 1.0:
            overlay_alpha = overlay_canvas.getchannel("A")
            overlay_alpha = overlay_alpha.point(lambda value: int(round(value * clamped_opacity)))
            overlay_canvas.putalpha(overlay_alpha)

        composed = Image.alpha_composite(background_rgba, overlay_canvas)
        results.append(
            _pil_to_tensor(
                composed,
                device=background_image.device,
                dtype=background_image.dtype,
            )
        )

    return torch.stack(results, dim=0)
