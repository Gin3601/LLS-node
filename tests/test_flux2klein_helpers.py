import importlib
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_flux2klein"


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


class FakeTensor:
    def __init__(self, shape, label="image"):
        self.shape = tuple(shape)
        self.label = label

    def resized(self, width, height):
        batch, _old_h, _old_w, channels = self.shape
        return FakeTensor((batch, height, width, channels), label=f"{self.label}:resized[{width}x{height}]")


class FakeMask:
    def __init__(self, shape, mask_bbox=None, mask_area_ratio=0.0, label="mask"):
        self.shape = tuple(shape)
        self.mask_bbox = mask_bbox
        self.mask_area_ratio = float(mask_area_ratio)
        self.label = label

    def resized(self, width, height):
        batch, _old_h, _old_w = self.shape
        return FakeMask(
            (batch, height, width),
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:resized[{width}x{height}]",
        )

    def normalized(self):
        return FakeMask(
            self.shape,
            mask_bbox=self.mask_bbox,
            mask_area_ratio=self.mask_area_ratio,
            label=f"{self.label}:normalized",
        )

    def inverted(self, image_size):
        width, height = image_size
        ratio = max(0.0, min(1.0, 1.0 - self.mask_area_ratio))
        bbox = None if ratio <= 0.0 else (0, 0, width, height)
        return FakeMask(
            (self.shape[0], height, width),
            mask_bbox=bbox,
            mask_area_ratio=ratio,
            label=f"{self.label}:inverted",
        )


class FakeLatentTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVAE:
    def __init__(self, latent_channels=4, downscale_ratio=8):
        self.latent_channels = int(latent_channels)
        self.downscale_ratio = int(downscale_ratio)
        self.encoded_shapes = []

    def encode(self, image):
        self.encoded_shapes.append(tuple(image.shape))
        batch, height, width, _channels = image.shape
        latent_height = max(1, height // self.downscale_ratio)
        latent_width = max(1, width // self.downscale_ratio)
        return FakeLatentTensor((batch, self.latent_channels, latent_height, latent_width))


class StandardClipStub:
    def __init__(self):
        self.tokenize_calls = []
        self.encode_calls = []

    def tokenize(self, text, images=None, **kwargs):
        self.tokenize_calls.append(
            {
                "text": text,
                "images": list(images or []),
                "kwargs": dict(kwargs),
            }
        )
        return {
            "text": text,
            "image_count": len(images or []),
        }

    def encode_from_tokens_scheduled(self, tokens, add_dict=None):
        self.encode_calls.append({"tokens": dict(tokens), "add_dict": dict(add_dict or {})})
        return [[f"cond::{tokens['text']}", {"pooled_output": dict(add_dict or {})}]]


class BrokenClipStub:
    pass


class NodeOutputStub:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


class NativeQwenStub:
    last_call = None

    class TextEncodeQwenImageEditPlus:
        @staticmethod
        def execute(clip, prompt, vae=None, image1=None, image2=None, image3=None):
            NativeQwenStub.last_call = {
                "clip": clip,
                "prompt": prompt,
                "vae": vae,
                "image1": image1,
                "image2": image2,
                "image3": image3,
            }
            return NodeOutputStub([["native-conditioning", {"source": "native-qwen"}]])


class NativeCoreNodesStub:
    last_text_call = None
    last_vae_calls = []

    @classmethod
    def reset(cls):
        cls.last_text_call = None
        cls.last_vae_calls = []

    class CLIPTextEncode:
        @staticmethod
        def execute(clip, prompt):
            NativeCoreNodesStub.last_text_call = {
                "clip": clip,
                "prompt": prompt,
            }
            return NodeOutputStub([[f"native-text-conditioning::{prompt}", {"pooled_output": {}}]])

    class VAEEncode:
        @staticmethod
        def execute(vae, image):
            NativeCoreNodesStub.last_vae_calls.append(
                {
                    "vae": vae,
                    "image": image,
                }
            )
            return NodeOutputStub({"samples": vae.encode(image)})


class NativeEditModelStub:
    last_calls = []

    @classmethod
    def reset(cls):
        cls.last_calls = []

    class ReferenceLatent:
        @staticmethod
        def execute(conditioning, latent=None):
            NativeEditModelStub.last_calls.append(
                {
                    "conditioning": conditioning,
                    "latent": latent,
                }
            )
            updated = []
            for cond, meta in conditioning:
                copied = dict(meta)
                refs = list(copied.get("reference_latents", []))
                refs.append(latent["samples"])
                copied["reference_latents"] = refs
                updated.append([cond, copied])
            return NodeOutputStub(updated)


class NativeFluxStub:
    last_empty_call = None

    @classmethod
    def reset(cls):
        cls.last_empty_call = None

    class EmptyFlux2LatentImage:
        @staticmethod
        def execute(width, height, batch_size=1):
            NativeFluxStub.last_empty_call = {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            }
            latent_height = max(1, int(height) // 16)
            latent_width = max(1, int(width) // 16)
            return NodeOutputStub(
                {
                    "samples": FakeLatentTensor((int(batch_size), 128, latent_height, latent_width)),
                }
            )
