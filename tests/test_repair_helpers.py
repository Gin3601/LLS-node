import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_repair"


class FakeTensor:
    def __init__(self, shape, label="image"):
        self.shape = tuple(shape)
        self.label = label

    def resized(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor((batch, height, width, channels), label=f"{self.label}:resized")

    def canvas_expanded(self, width, height):
        batch, _, _, channels = self.shape
        return FakeTensor((batch, height, width, channels), label=f"{self.label}:canvas")


class FakeMask:
    def __init__(self, shape, mask_bbox=None, mask_area_ratio=0.0, label="mask"):
        self.shape = tuple(shape)
        self.mask_bbox = mask_bbox
        self.mask_area_ratio = float(mask_area_ratio)
        self.label = label

    def resized(self, width, height):
        batch, _, _ = self.shape
        return FakeMask(
            (batch, height, width),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:resized",
        )

    def canvas_expanded(self, width, height):
        batch, _, _ = self.shape
        return FakeMask(
            (batch, height, width),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:canvas",
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
