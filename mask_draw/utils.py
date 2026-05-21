from __future__ import annotations

import base64
import io
import json

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional runtime dependency
    np = None
    _NUMPY_ERR = exc
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    _NUMPY_ERR = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - optional runtime dependency
    Image = None
    _PIL_ERR = exc
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    _PIL_ERR = None

try:
    import torch
    import torch.nn.functional as torch_nn_functional
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    torch_nn_functional = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised only in ComfyUI-like runtime
    _TORCH_ERR = None


_DEFAULT_MASK_STATE = {
    "version": 1,
    "mask_png_base64": "",
    "touched": False,
    "editor": {},
}


def parse_mask_state(mask_state_json):
    if not mask_state_json:
        return dict(_DEFAULT_MASK_STATE)

    try:
        payload = json.loads(mask_state_json)
    except (TypeError, ValueError):
        return dict(_DEFAULT_MASK_STATE)

    if not isinstance(payload, dict):
        return dict(_DEFAULT_MASK_STATE)

    state = dict(_DEFAULT_MASK_STATE)
    state["version"] = _coerce_int(payload.get("version"), 1)
    state["mask_png_base64"] = str(payload.get("mask_png_base64") or "")
    state["touched"] = _coerce_bool(payload.get("touched"), False)
    state["editor"] = _coerce_dict(payload.get("editor"))
    return state


def resolve_output_mask(image, input_mask, mask_state_json, invert_mask):
    state = parse_mask_state(mask_state_json)
    resolved = None

    if state["touched"]:
        resolved = _decode_mask_png_base64(state["mask_png_base64"], image)

    if resolved is None and input_mask is not None:
        resolved = _resize_mask_to_image(input_mask, image)

    if resolved is None:
        resolved = _make_black_mask(image)

    if invert_mask:
        resolved = 1.0 - resolved

    return resolved.clamp(0.0, 1.0)


def build_preview_image(image, mask, overlay_alpha):
    _require_torch("[LLS] torch is required for preview image generation.")

    alpha = max(0.0, min(1.0, _coerce_float(overlay_alpha, 0.4)))
    normalized_mask = _resize_mask_to_image(mask, image).unsqueeze(-1)
    overlay_strength = normalized_mask * alpha
    overlay = torch.zeros_like(image)
    overlay[..., 0] = 1.0
    return (image * (1.0 - overlay_strength) + (overlay * overlay_strength)).clamp(0.0, 1.0)


def _get_image_size(image):
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise RuntimeError("[LLS] image must have shape [batch, height, width, channels].")

    _, height, width, _ = tuple(shape)
    width = _coerce_int(width, 0)
    height = _coerce_int(height, 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("[LLS] image width and height must be positive.")
    return width, height


def _get_image_batch_size(image):
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise RuntimeError("[LLS] image must have shape [batch, height, width, channels].")

    batch = _coerce_int(shape[0], 0)
    if batch <= 0:
        raise RuntimeError("[LLS] image batch size must be positive.")
    return batch


def _make_black_mask(image):
    _require_torch("[LLS] torch is required for mask generation.")

    width, height = _get_image_size(image)
    return torch.zeros((image.shape[0], height, width), dtype=image.dtype, device=image.device)


def _resize_mask_to_image(mask, image):
    _require_torch("[LLS] torch is required for mask resizing.")

    shape = getattr(mask, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("[LLS] mask must have shape [batch, height, width].")

    image_batch = _get_image_batch_size(image)
    mask_batch = _coerce_int(shape[0], 0)
    if mask_batch <= 0:
        raise RuntimeError("[LLS] mask batch size must be positive.")

    width, height = _get_image_size(image)
    normalized = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    if mask_batch != image_batch:
        if mask_batch == 1 and image_batch > 1:
            normalized = normalized.repeat(image_batch, 1, 1)
        else:
            raise RuntimeError(
                f"[LLS] mask batch size must match image batch size or be 1; got mask batch {mask_batch} and image batch {image_batch}."
            )
    if tuple(normalized.shape[1:]) == (height, width):
        return normalized

    resized = torch_nn_functional.interpolate(
        normalized.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1).clamp(0.0, 1.0)


def _decode_mask_png_base64(mask_png_base64, image):
    _require_torch("[LLS] torch is required for mask decoding.")
    _require_mask_decode_deps()

    if not mask_png_base64:
        return None

    width, height = _get_image_size(image)
    try:
        payload = base64.b64decode(mask_png_base64)
        decoded = Image.open(io.BytesIO(payload)).convert("L")
    except Exception:
        return None

    if decoded.size != (width, height):
        decoded = decoded.resize((width, height), Image.BILINEAR)

    array = np.asarray(decoded, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).to(device=image.device, dtype=image.dtype)
    tensor = tensor.unsqueeze(0)
    if image.shape[0] > 1:
        tensor = tensor.repeat(image.shape[0], 1, 1)
    return tensor.clamp(0.0, 1.0)


def _require_torch(message):
    if torch is None or torch_nn_functional is None:
        raise RuntimeError(message) from _TORCH_ERR


def _require_mask_decode_deps():
    if np is None:
        raise RuntimeError("[LLS] numpy is required for mask decoding.") from _NUMPY_ERR
    if Image is None:
        raise RuntimeError("[LLS] Pillow is required for mask decoding.") from _PIL_ERR


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    return default


def _coerce_dict(value):
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
