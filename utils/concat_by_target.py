from __future__ import annotations

try:
    import torch
    import torch.nn.functional as torch_nn_functional
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None
    torch_nn_functional = None
    _TORCH_ERR = exc
else:  # pragma: no cover - exercised in runtime/tests
    _TORCH_ERR = None


DATA_TYPE_CHOICES = ["IMAGE", "MASK"]
TARGET_CHOICES = ["A", "B"]
POSITION_CHOICES = ["top", "bottom", "left", "right"]
RESIZE_MODE_CHOICES = ["keep_proportion", "stretch", "none"]
ALIGN_CHOICES = ["start", "center", "end"]


def _require_torch():
    if torch is None or torch_nn_functional is None:
        raise RuntimeError("[LLS] torch is required for LLS Concat By Target.") from _TORCH_ERR


def _validate_choice(name: str, value: str, choices):
    if value not in choices:
        raise RuntimeError(f"[LLS] Invalid {name} '{value}'. Expected one of: {', '.join(choices)}.")


def parse_hex_color(color_str: str):
    value = str(color_str or "").strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 3:
        value = "".join(part * 2 for part in value)
    if len(value) != 6:
        raise RuntimeError("[LLS] background_color must be a valid HEX color like '#000000' or '#ffffff'.")
    try:
        red = int(value[0:2], 16) / 255.0
        green = int(value[2:4], 16) / 255.0
        blue = int(value[4:6], 16) / 255.0
    except ValueError as exc:
        raise RuntimeError(
            "[LLS] background_color must be a valid HEX color like '#000000' or '#ffffff'."
        ) from exc
    return red, green, blue


def ensure_image_tensor(x):
    _require_torch()
    if x is None:
        raise RuntimeError("[LLS] IMAGE mode requires both image_a and image_b inputs.")
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise RuntimeError("[LLS] IMAGE input must have shape [B,H,W,C] or [H,W,C].")
    batch, height, width, channels = [int(dim) for dim in x.shape]
    if batch <= 0 or height <= 0 or width <= 0 or channels <= 0:
        raise RuntimeError("[LLS] IMAGE input must have positive batch, height, width, and channels.")
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float32)
    return x


def ensure_mask_tensor(x):
    _require_torch()
    if x is None:
        raise RuntimeError("[LLS] MASK mode requires both mask_a and mask_b inputs.")
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        pass
    elif x.ndim == 4 and int(x.shape[-1]) == 1:
        x = x.squeeze(-1)
    elif x.ndim == 4 and int(x.shape[1]) == 1:
        x = x.squeeze(1)
    else:
        raise RuntimeError("[LLS] MASK input must have shape [B,H,W], [H,W], [B,H,W,1], or [B,1,H,W].")
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float32)
    return x.to(dtype=torch.float32).clamp(0.0, 1.0)


def _repeat_batch(tensor, target_batch: int):
    repeat_shape = [1] * tensor.ndim
    repeat_shape[0] = target_batch
    return tensor.repeat(*repeat_shape)


def broadcast_batch(a, b, allow_batch_broadcast: bool):
    batch_a = int(a.shape[0])
    batch_b = int(b.shape[0])
    if batch_a == batch_b:
        return a, b
    if allow_batch_broadcast and batch_a == 1 and batch_b > 1:
        return _repeat_batch(a, batch_b), b
    if allow_batch_broadcast and batch_b == 1 and batch_a > 1:
        return a, _repeat_batch(b, batch_a)
    raise RuntimeError(
        f"[LLS] Batch size mismatch: A has batch {batch_a}, B has batch {batch_b}, and broadcasting is disabled or impossible."
    )


def resize_tensor_to_match(tensor, *, data_type: str, match_axis: str, target_size: int, resize_mode: str):
    _validate_choice("resize_mode", resize_mode, RESIZE_MODE_CHOICES)
    if resize_mode == "none":
        return tensor

    if data_type == "IMAGE":
        current_height = int(tensor.shape[1])
        current_width = int(tensor.shape[2])
    else:
        current_height = int(tensor.shape[1])
        current_width = int(tensor.shape[2])

    if match_axis == "height":
        if current_height == target_size:
            return tensor
        new_height = max(1, int(target_size))
        if resize_mode == "keep_proportion":
            scale = float(new_height) / float(current_height)
            new_width = max(1, int(round(current_width * scale)))
        else:
            new_width = current_width
    elif match_axis == "width":
        if current_width == target_size:
            return tensor
        new_width = max(1, int(target_size))
        if resize_mode == "keep_proportion":
            scale = float(new_width) / float(current_width)
            new_height = max(1, int(round(current_height * scale)))
        else:
            new_height = current_height
    else:
        raise RuntimeError(f"[LLS] Unsupported match axis '{match_axis}'.")

    if data_type == "IMAGE":
        resized = torch_nn_functional.interpolate(
            tensor.movedim(-1, 1),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.movedim(1, -1)

    resized = torch_nn_functional.interpolate(
        tensor.unsqueeze(1),
        size=(new_height, new_width),
        mode="nearest",
    )
    return resized.squeeze(1)


def create_canvas(
    *,
    data_type: str,
    batch: int,
    height: int,
    width: int,
    device,
    dtype,
    channels: int | None = None,
    background_color=None,
    background_value: float = 0.0,
):
    if data_type == "IMAGE":
        if channels is None:
            raise RuntimeError("[LLS] IMAGE canvas creation requires a channels value.")
        canvas = torch.zeros((batch, height, width, channels), device=device, dtype=dtype)
        red, green, blue = background_color
        if channels == 1:
            canvas.fill_((red + green + blue) / 3.0)
            return canvas
        fill_values = [0.0] * channels
        fill_values[0] = red
        if channels > 1:
            fill_values[1] = green
        if channels > 2:
            fill_values[2] = blue
        if channels == 4:
            fill_values[3] = 1.0
        fill_tensor = torch.tensor(fill_values, device=device, dtype=dtype).view(1, 1, 1, channels)
        canvas[:] = fill_tensor
        return canvas

    return torch.full(
        (batch, height, width),
        float(background_value),
        device=device,
        dtype=dtype,
    )


def paste_tensor(canvas, tensor, top: int, left: int):
    if canvas.ndim == 4:
        canvas[:, top : top + tensor.shape[1], left : left + tensor.shape[2], :] = tensor
    else:
        canvas[:, top : top + tensor.shape[1], left : left + tensor.shape[2]] = tensor
    return canvas


def pad_to_multiple(
    tensor,
    *,
    data_type: str,
    multiple_of: int,
    background_color=None,
    background_value: float = 0.0,
):
    if multiple_of <= 0:
        return tensor

    if data_type == "IMAGE":
        batch, height, width, channels = [int(dim) for dim in tensor.shape]
        dtype = tensor.dtype
        device = tensor.device
    else:
        batch, height, width = [int(dim) for dim in tensor.shape]
        channels = None
        dtype = tensor.dtype
        device = tensor.device

    padded_height = ((height + multiple_of - 1) // multiple_of) * multiple_of
    padded_width = ((width + multiple_of - 1) // multiple_of) * multiple_of
    if padded_height == height and padded_width == width:
        return tensor

    canvas = create_canvas(
        data_type=data_type,
        batch=batch,
        height=padded_height,
        width=padded_width,
        channels=channels,
        device=device,
        dtype=dtype,
        background_color=background_color,
        background_value=background_value,
    )
    return paste_tensor(canvas, tensor, 0, 0)


def _alignment_offset(container_size: int, item_size: int, align: str) -> int:
    _validate_choice("align", align, ALIGN_CHOICES)
    if align == "start":
        return 0
    if align == "center":
        return max(0, (container_size - item_size) // 2)
    return max(0, container_size - item_size)


def concat_by_target(
    *,
    data_type: str,
    tensor_a,
    tensor_b,
    target: str,
    position: str,
    match_target_size: bool,
    resize_mode: str,
    align: str,
    gap: int,
    multiple_of: int,
    allow_batch_broadcast: bool,
    background_color=None,
    background_value: float = 0.0,
):
    _validate_choice("data_type", data_type, DATA_TYPE_CHOICES)
    _validate_choice("target", target, TARGET_CHOICES)
    _validate_choice("position", position, POSITION_CHOICES)
    _validate_choice("resize_mode", resize_mode, RESIZE_MODE_CHOICES)
    _validate_choice("align", align, ALIGN_CHOICES)
    if gap < 0:
        raise RuntimeError("[LLS] gap must be >= 0.")
    if multiple_of < 0:
        raise RuntimeError("[LLS] multiple_of must be >= 0.")

    tensor_a, tensor_b = broadcast_batch(tensor_a, tensor_b, allow_batch_broadcast)
    target_tensor = tensor_a if target == "A" else tensor_b
    other_tensor = tensor_b if target == "A" else tensor_a

    if data_type == "IMAGE":
        work_dtype = target_tensor.dtype if torch.is_floating_point(target_tensor) else torch.float32
        target_tensor = target_tensor.to(dtype=work_dtype)
        other_tensor = other_tensor.to(device=target_tensor.device, dtype=work_dtype)
    else:
        target_tensor = target_tensor.to(dtype=torch.float32)
        other_tensor = other_tensor.to(device=target_tensor.device, dtype=torch.float32)

    if match_target_size:
        match_axis = "height" if position in {"left", "right"} else "width"
        target_size = int(target_tensor.shape[1] if match_axis == "height" else target_tensor.shape[2])
        other_tensor = resize_tensor_to_match(
            other_tensor,
            data_type=data_type,
            match_axis=match_axis,
            target_size=target_size,
            resize_mode=resize_mode,
        )

    if data_type == "IMAGE":
        batch = int(target_tensor.shape[0])
        target_height = int(target_tensor.shape[1])
        target_width = int(target_tensor.shape[2])
        other_height = int(other_tensor.shape[1])
        other_width = int(other_tensor.shape[2])
        channels = int(target_tensor.shape[3])
    else:
        batch = int(target_tensor.shape[0])
        target_height = int(target_tensor.shape[1])
        target_width = int(target_tensor.shape[2])
        other_height = int(other_tensor.shape[1])
        other_width = int(other_tensor.shape[2])
        channels = None

    if position in {"left", "right"}:
        canvas_height = max(target_height, other_height)
        canvas_width = target_width + int(gap) + other_width
        target_top = _alignment_offset(canvas_height, target_height, align)
        other_top = _alignment_offset(canvas_height, other_height, align)
        if position == "right":
            target_left = 0
            other_left = target_width + int(gap)
        else:
            other_left = 0
            target_left = other_width + int(gap)
    else:
        canvas_height = target_height + int(gap) + other_height
        canvas_width = max(target_width, other_width)
        target_left = _alignment_offset(canvas_width, target_width, align)
        other_left = _alignment_offset(canvas_width, other_width, align)
        if position == "bottom":
            target_top = 0
            other_top = target_height + int(gap)
        else:
            other_top = 0
            target_top = other_height + int(gap)

    canvas = create_canvas(
        data_type=data_type,
        batch=batch,
        height=canvas_height,
        width=canvas_width,
        channels=channels,
        device=target_tensor.device,
        dtype=target_tensor.dtype,
        background_color=background_color,
        background_value=background_value,
    )
    paste_tensor(canvas, target_tensor, target_top, target_left)
    paste_tensor(canvas, other_tensor, other_top, other_left)
    canvas = pad_to_multiple(
        canvas,
        data_type=data_type,
        multiple_of=int(multiple_of),
        background_color=background_color,
        background_value=background_value,
    )

    if data_type == "IMAGE":
        final_height = int(canvas.shape[1])
        final_width = int(canvas.shape[2])
    else:
        final_height = int(canvas.shape[1])
        final_width = int(canvas.shape[2])
    return canvas, final_width, final_height


def mask_to_preview_image(mask):
    normalized = ensure_mask_tensor(mask)
    return normalized.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0)


class LLSConcatByTarget:
    CATEGORY = "LLS/Utils"
    FUNCTION = "concat"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    DESCRIPTION = "Concatenate IMAGE or MASK inputs around a chosen target canvas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_type": (DATA_TYPE_CHOICES, {"default": "IMAGE"}),
                "target": (TARGET_CHOICES, {"default": "A"}),
                "position": (POSITION_CHOICES, {"default": "right"}),
                "match_target_size": ("BOOLEAN", {"default": True}),
                "resize_mode": (RESIZE_MODE_CHOICES, {"default": "keep_proportion"}),
                "align": (ALIGN_CHOICES, {"default": "center"}),
                "gap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "background_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multiple_of": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "allow_batch_broadcast": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            },
        }

    def concat(
        self,
        data_type,
        target,
        position,
        match_target_size,
        resize_mode,
        align,
        gap,
        background_color,
        background_value,
        multiple_of,
        allow_batch_broadcast,
        image_a=None,
        image_b=None,
        mask_a=None,
        mask_b=None,
    ):
        _require_torch()
        data_type = str(data_type or "IMAGE")
        background_value = max(0.0, min(1.0, float(background_value)))

        if data_type == "IMAGE":
            if image_a is None or image_b is None:
                raise RuntimeError("[LLS] IMAGE mode requires both image_a and image_b inputs.")
            color = parse_hex_color(background_color)
            tensor_a = ensure_image_tensor(image_a)
            tensor_b = ensure_image_tensor(image_b)
            output_image, width, height = concat_by_target(
                data_type="IMAGE",
                tensor_a=tensor_a,
                tensor_b=tensor_b,
                target=str(target),
                position=str(position),
                match_target_size=bool(match_target_size),
                resize_mode=str(resize_mode),
                align=str(align),
                gap=int(gap),
                multiple_of=int(multiple_of),
                allow_batch_broadcast=bool(allow_batch_broadcast),
                background_color=color,
                background_value=background_value,
            )
            output_image = output_image.clamp(0.0, 1.0)
            output_mask = torch.zeros(
                (int(output_image.shape[0]), int(output_image.shape[1]), int(output_image.shape[2])),
                device=output_image.device,
                dtype=torch.float32,
            )
            return output_image, output_mask, int(width), int(height)

        if data_type == "MASK":
            if mask_a is None or mask_b is None:
                raise RuntimeError("[LLS] MASK mode requires both mask_a and mask_b inputs.")
            tensor_a = ensure_mask_tensor(mask_a)
            tensor_b = ensure_mask_tensor(mask_b)
            output_mask, width, height = concat_by_target(
                data_type="MASK",
                tensor_a=tensor_a,
                tensor_b=tensor_b,
                target=str(target),
                position=str(position),
                match_target_size=bool(match_target_size),
                resize_mode=str(resize_mode),
                align=str(align),
                gap=int(gap),
                multiple_of=int(multiple_of),
                allow_batch_broadcast=bool(allow_batch_broadcast),
                background_color=None,
                background_value=background_value,
            )
            output_mask = output_mask.clamp(0.0, 1.0).to(dtype=torch.float32)
            output_image = mask_to_preview_image(output_mask)
            return output_image, output_mask, int(width), int(height)

        raise RuntimeError(f"[LLS] Invalid data_type '{data_type}'. Expected IMAGE or MASK.")


NODE_CLASS_MAPPINGS = {"LLSConcatByTarget": LLSConcatByTarget}
NODE_DISPLAY_NAME_MAPPINGS = {"LLSConcatByTarget": "LLS Concat By Target"}
