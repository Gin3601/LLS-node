from __future__ import annotations

from collections import deque
import math

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional runtime dependency
    np = None
    _NUMPY_ERR = exc
else:  # pragma: no cover - exercised in runtime/tests
    _NUMPY_ERR = None

try:
    import torch
    import torch.nn.functional as torch_nn_functional
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    torch_nn_functional = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised in runtime/tests
    _TORCH_ERR = None


def _require_torch():
    if torch is None or torch_nn_functional is None:
        raise RuntimeError("[LLS] torch is required for mask processing nodes.") from _TORCH_ERR


def _coerce_non_negative_int(value) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def normalize_mask(mask):
    _require_torch()

    if mask is None:
        raise RuntimeError("[LLS] mask input is required.")
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    ndim = int(mask.ndim)
    if ndim == 2:
        normalized = mask.unsqueeze(0)
    elif ndim == 3:
        normalized = mask
    elif ndim == 4:
        if int(mask.shape[-1]) == 1:
            normalized = mask.squeeze(-1)
        elif int(mask.shape[1]) == 1:
            normalized = mask.squeeze(1)
        else:
            raise RuntimeError(
                "[LLS] mask must have shape [H,W], [B,H,W], [B,H,W,1], or [B,1,H,W]."
            )
    else:
        raise RuntimeError(
            "[LLS] mask must have shape [H,W], [B,H,W], [B,H,W,1], or [B,1,H,W]."
        )

    return normalized.to(dtype=torch.float32).clamp(0.0, 1.0)


def resize_mask_to_hw(mask, height, width):
    normalized = normalize_mask(mask)
    target_height = max(1, int(height))
    target_width = max(1, int(width))
    if tuple(normalized.shape[1:]) == (target_height, target_width):
        return normalized

    padded = normalized.unsqueeze(1)
    resized = torch_nn_functional.interpolate(
        padded,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1).clamp(0.0, 1.0).to(dtype=torch.float32)


def align_mask_batch(mask, target_batch):
    normalized = normalize_mask(mask)
    batch = int(normalized.shape[0])
    target = max(1, int(target_batch))
    if batch == target:
        return normalized
    if batch == 1:
        return normalized.repeat(target, 1, 1)
    if batch < target:
        repeats = int(math.ceil(target / batch))
        return normalized.repeat(repeats, 1, 1)[:target]
    return normalized[:target]


def apply_grow(mask, radius):
    normalized = normalize_mask(mask)
    grow_radius = _coerce_non_negative_int(radius)
    if grow_radius == 0:
        return normalized

    kernel_size = (grow_radius * 2) + 1
    grown = torch_nn_functional.max_pool2d(
        normalized.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=grow_radius,
    )
    return grown.squeeze(1).clamp(0.0, 1.0).to(dtype=torch.float32)


def apply_shrink(mask, radius):
    normalized = normalize_mask(mask)
    shrink_radius = _coerce_non_negative_int(radius)
    if shrink_radius == 0:
        return normalized

    inverted = 1.0 - normalized
    kernel_size = (shrink_radius * 2) + 1
    pooled = torch_nn_functional.max_pool2d(
        inverted.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=shrink_radius,
    )
    return (1.0 - pooled.squeeze(1)).clamp(0.0, 1.0).to(dtype=torch.float32)


def apply_blur(mask, radius):
    normalized = normalize_mask(mask)
    blur_radius = _coerce_non_negative_int(radius)
    if blur_radius == 0:
        return normalized

    kernel_size = (blur_radius * 2) + 1
    padded = torch_nn_functional.pad(
        normalized.unsqueeze(1),
        (blur_radius, blur_radius, blur_radius, blur_radius),
        mode="replicate",
    )
    blurred = torch_nn_functional.avg_pool2d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    )
    return blurred.squeeze(1).clamp(0.0, 1.0).to(dtype=torch.float32)


def fill_mask_holes(mask):
    normalized = normalize_mask(mask)
    if np is None:
        return normalized

    binary = (normalized > 0.5).detach().cpu().numpy().astype(np.uint8)
    filled = np.zeros_like(binary, dtype=np.float32)
    for index, sample in enumerate(binary):
        filled[index] = _fill_holes_binary(sample)

    return torch.from_numpy(filled).to(device=normalized.device, dtype=torch.float32).clamp(0.0, 1.0)


def remove_small_mask_regions(mask, min_area):
    normalized = normalize_mask(mask)
    area_threshold = _coerce_non_negative_int(min_area)
    if area_threshold <= 1 or np is None:
        return normalized

    binary = (normalized > 0.5).detach().cpu().numpy().astype(np.uint8)
    cleaned = np.zeros_like(binary, dtype=np.float32)
    for index, sample in enumerate(binary):
        cleaned[index] = _remove_small_regions_binary(sample, area_threshold)

    return torch.from_numpy(cleaned).to(device=normalized.device, dtype=torch.float32).clamp(0.0, 1.0)


def resolve_image_size(image):
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) != 4:
        return None
    batch = max(1, int(shape[0]))
    height = max(1, int(shape[1]))
    width = max(1, int(shape[2]))
    return batch, height, width


def _fill_holes_binary(sample):
    height, width = sample.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = deque()

    def enqueue(y, x):
        if 0 <= y < height and 0 <= x < width and sample[y, x] == 0 and not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)

    while queue:
        y, x = queue.popleft()
        enqueue(y - 1, x)
        enqueue(y + 1, x)
        enqueue(y, x - 1)
        enqueue(y, x + 1)

    result = sample.copy()
    result[(sample == 0) & (~visited)] = 1
    return result.astype(np.float32)


def _remove_small_regions_binary(sample, min_area):
    height, width = sample.shape
    visited = np.zeros((height, width), dtype=bool)
    result = np.zeros((height, width), dtype=np.float32)

    for start_y in range(height):
        for start_x in range(width):
            if sample[start_y, start_x] == 0 or visited[start_y, start_x]:
                continue

            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            region = []

            while queue:
                y, x = queue.popleft()
                region.append((y, x))
                for next_y, next_x in (
                    (y - 1, x),
                    (y + 1, x),
                    (y, x - 1),
                    (y, x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and sample[next_y, next_x] == 1
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))

            if len(region) >= min_area:
                for y, x in region:
                    result[y, x] = 1.0

    return result
