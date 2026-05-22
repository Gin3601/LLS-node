import importlib
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_repair"
_UNSET = object()


class FakeTensor:
    def __init__(self, shape, label="image", crop_box=None, fill_mode=None, original_box=None):
        self.shape = tuple(shape)
        self.label = label
        self.crop_box = crop_box
        self.fill_mode = fill_mode
        self.original_box = original_box

    def cropped(self, x1, y1, x2, y2):
        batch, _, _, channels = self.shape
        crop_box = (x1, y1, x2, y2)
        return FakeTensor(
            (batch, max(0, y2 - y1), max(0, x2 - x1), channels),
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=crop_box,
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )

    def resized(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:resized",
            crop_box=self.crop_box,
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )

    def canvas_expanded(self, width, height, fill_mode="edge", original_box=None):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:canvas[{fill_mode}]",
            crop_box=self.crop_box,
            fill_mode=fill_mode,
            original_box=original_box,
        )

    def masked_fill(self, mask, fill_value):
        del mask
        return FakeTensor(
            self.shape,
            label=f"{self.label}:masked[{fill_value}]",
            crop_box=self.crop_box,
            fill_mode=self.fill_mode,
            original_box=self.original_box,
        )


class FakeMask:
    def __init__(self, shape, mask_bbox=None, mask_area_ratio=0.0, label="mask", crop_box=None):
        self.shape = tuple(shape)
        self.mask_bbox = mask_bbox
        self.mask_area_ratio = float(mask_area_ratio)
        self.label = label
        self.crop_box = crop_box

    def _clone(self, shape=None, mask_bbox=_UNSET, mask_area_ratio=None, label=None, crop_box=None):
        return FakeMask(
            shape or self.shape,
            mask_bbox=self.mask_bbox if mask_bbox is _UNSET else mask_bbox,
            mask_area_ratio=self.mask_area_ratio if mask_area_ratio is None else mask_area_ratio,
            label=self.label if label is None else label,
            crop_box=self.crop_box if crop_box is None else crop_box,
        )

    def _current_size(self):
        return self.shape[2], self.shape[1]

    @staticmethod
    def _clamp_box(box, width, height):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(x1, min(int(x2), width))
        y2 = max(y1, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_area_ratio(box, width, height):
        if box is None or width <= 0 or height <= 0:
            return 0.0
        return ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)

    def cropped(self, x1, y1, x2, y2):
        batch, _, _ = self.shape
        crop_box = (x1, y1, x2, y2)
        new_width = max(0, x2 - x1)
        new_height = max(0, y2 - y1)
        new_bbox = None
        if self.mask_bbox is not None:
            new_bbox = self._clamp_box(
                (
                    self.mask_bbox[0] - x1,
                    self.mask_bbox[1] - y1,
                    self.mask_bbox[2] - x1,
                    self.mask_bbox[3] - y1,
                ),
                new_width,
                new_height,
            )
        return FakeMask(
            (batch, new_height, new_width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, new_width, new_height),
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=crop_box,
        )

    def resized(self, width, height):
        batch, _, _ = self.shape
        source_width, source_height = self._current_size()
        new_bbox = None
        if self.mask_bbox is not None and source_width > 0 and source_height > 0:
            scale_x = width / float(source_width)
            scale_y = height / float(source_height)
            new_bbox = self._clamp_box(
                (
                    round(self.mask_bbox[0] * scale_x),
                    round(self.mask_bbox[1] * scale_y),
                    round(self.mask_bbox[2] * scale_x),
                    round(self.mask_bbox[3] * scale_y),
                ),
                width,
                height,
            )
        return FakeMask(
            (batch, height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:resized",
            crop_box=self.crop_box,
        )

    def canvas_expanded(self, width, height, original_box=None):
        batch, _, _ = self.shape
        if original_box is None:
            original_box = (0, 0, self.shape[2], self.shape[1])
        offset_x, offset_y = original_box[0], original_box[1]
        new_bbox = None
        if self.mask_bbox is not None:
            new_bbox = self._clamp_box(
                (
                    self.mask_bbox[0] + offset_x,
                    self.mask_bbox[1] + offset_y,
                    self.mask_bbox[2] + offset_x,
                    self.mask_bbox[3] + offset_y,
                ),
                width,
                height,
            )
        return FakeMask(
            (batch, height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:canvas",
            crop_box=self.crop_box,
        )

    def normalized(self):
        return self._clone(label=f"{self.label}:normalize")

    def inverted(self, image_size):
        width, height = image_size
        inverted_ratio = max(0.0, min(1.0, 1.0 - self.mask_area_ratio))
        inverted_bbox = None if inverted_ratio <= 0.0 else (0, 0, width, height)
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=inverted_bbox,
            mask_area_ratio=inverted_ratio,
            label=f"{self.label}:invert",
        )

    def thresholded(self, threshold):
        width, height = self._current_size()
        if self.mask_bbox is None or self.mask_area_ratio <= 0.0:
            return self._clone(label=f"{self.label}:threshold[{threshold}]")
        shrink_ratio = max(0.0, min(1.0, 1.0 - max(0.0, float(threshold) - 0.5)))
        if shrink_ratio <= 0.0:
            return self._clone(mask_bbox=None, mask_area_ratio=0.0, label=f"{self.label}:threshold[{threshold}]")
        x1, y1, x2, y2 = self.mask_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_width = max(1.0, (x2 - x1) * shrink_ratio / 2.0)
        half_height = max(1.0, (y2 - y1) * shrink_ratio / 2.0)
        new_bbox = self._clamp_box(
            (
                round(cx - half_width),
                round(cy - half_height),
                round(cx + half_width),
                round(cy + half_height),
            ),
            width,
            height,
        )
        return self._clone(
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:threshold[{threshold}]",
        )

    def grown(self, amount, image_size):
        width, height = image_size
        if self.mask_bbox is None:
            return self._clone(shape=(self.shape[0], height, width), label=f"{self.label}:grow[{amount}]")
        x1, y1, x2, y2 = self.mask_bbox
        new_bbox = self._clamp_box((x1 - amount, y1 - amount, x2 + amount, y2 + amount), width, height)
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:grow[{amount}]",
        )

    def blurred(self, radius, image_size):
        blur_expand = max(1, int(round(float(radius) / 4.0)))
        return self.grown(blur_expand, image_size)._clone(label=f"{self.label}:blur[{radius}]")

    def merged_with(self, other, image_size):
        width, height = image_size
        if self.mask_bbox is None:
            return other._clone(shape=(other.shape[0], height, width), label=f"{self.label}:merge:{other.label}")
        if other.mask_bbox is None:
            return self._clone(shape=(self.shape[0], height, width), label=f"{self.label}:merge:{other.label}")
        new_bbox = self._clamp_box(
            (
                min(self.mask_bbox[0], other.mask_bbox[0]),
                min(self.mask_bbox[1], other.mask_bbox[1]),
                max(self.mask_bbox[2], other.mask_bbox[2]),
                max(self.mask_bbox[3], other.mask_bbox[3]),
            ),
            width,
            height,
        )
        return self._clone(
            shape=(self.shape[0], height, width),
            mask_bbox=new_bbox,
            mask_area_ratio=self._bbox_area_ratio(new_bbox, width, height),
            label=f"{self.label}:merge:{other.label}",
        )

    @classmethod
    def canvas_region(cls, shape, original_box, label="canvas-region"):
        batch, height, width = shape
        original_width = max(0, original_box[2] - original_box[0])
        original_height = max(0, original_box[3] - original_box[1])
        total_area = max(1, width * height)
        original_area = max(0, original_width * original_height)
        area_ratio = max(0.0, min(1.0, (total_area - original_area) / float(total_area)))
        bbox = None if area_ratio <= 0.0 else (0, 0, width, height)
        return cls(shape, mask_bbox=bbox, mask_area_ratio=area_ratio, label=label)


class FakeLatentTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVAE:
    def __init__(self, latent_channels=4, downscale_ratio=8):
        self.latent_channels = int(latent_channels)
        self.downscale_ratio = int(downscale_ratio)

    def encode(self, image):
        batch, height, width, _ = image.shape
        latent_height = max(1, height // self.downscale_ratio)
        latent_width = max(1, width // self.downscale_ratio)
        return FakeLatentTensor((batch, self.latent_channels, latent_height, latent_width))


class FakeModel:
    def __init__(
        self,
        family="SDXL",
        model_role="base",
        supports_inpaint_native=False,
        supports_image_edit_native=False,
        preferred_edit_backend=None,
        model_name="demo-model.safetensors",
        profile_id="",
        backend_type="",
        sampler_strategy="",
        loader_strategy="",
    ):
        self._lls_family = family
        self._lls_model_role = model_role
        self._lls_supports_inpaint_native = supports_inpaint_native
        self._lls_supports_image_edit_native = supports_image_edit_native
        self._lls_preferred_edit_backend = preferred_edit_backend
        self._lls_model_name = model_name
        self._lls_checkpoint_name = model_name
        self._lls_profile_id = profile_id
        self._lls_backend_type = backend_type
        self._lls_sampler_strategy = sampler_strategy
        self._lls_loader_strategy = loader_strategy


def make_conditioning(label):
    return [[label, {"origin": label}]]


def load_plugin_package():
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(f"{MODULE_NAME}."):
            sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def import_plugin_submodule(plugin, dotted_name: str):
    return importlib.import_module(f"{plugin.__name__}.{dotted_name}")
