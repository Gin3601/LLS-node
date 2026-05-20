import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_repair"


class FakeTensor:
    def __init__(self, shape, label="image", crop_box=None):
        self.shape = tuple(shape)
        self.label = label
        self.crop_box = crop_box

    def cropped(self, x1, y1, x2, y2):
        batch, _, _, channels = self.shape
        crop_box = (x1, y1, x2, y2)
        return FakeTensor(
            (batch, max(0, y2 - y1), max(0, x2 - x1), channels),
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=crop_box,
        )

    def resized(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:resized",
            crop_box=self.crop_box,
        )

    def canvas_expanded(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor(
            (batch, height, width, channels),
            label=f"{self.label}:canvas",
            crop_box=self.crop_box,
        )


class FakeMask:
    def __init__(self, shape, mask_bbox=None, mask_area_ratio=0.0, label="mask", crop_box=None):
        self.shape = tuple(shape)
        self.mask_bbox = mask_bbox
        self.mask_area_ratio = float(mask_area_ratio)
        self.label = label
        self.crop_box = crop_box

    def cropped(self, x1, y1, x2, y2):
        batch, _, _ = self.shape
        crop_box = (x1, y1, x2, y2)
        return FakeMask(
            (batch, max(0, y2 - y1), max(0, x2 - x1)),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:crop[{x1},{y1},{x2},{y2}]",
            crop_box=crop_box,
        )

    def resized(self, width, height):
        batch, _, _ = self.shape
        return FakeMask(
            (batch, height, width),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:resized",
            crop_box=self.crop_box,
        )

    def canvas_expanded(self, width, height):
        batch, _, _ = self.shape
        return FakeMask(
            (batch, height, width),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:canvas",
            crop_box=self.crop_box,
        )


class FakeLatentTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return FakeLatentTensor((batch, 4, height // 8, width // 8))


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
