import importlib.util
import json
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "lls_node_test_refactor",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test_refactor"] = module
    spec.loader.exec_module(module)
    return module


class TaggedValue:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return self.label


class FolderPathsStub:
    def __init__(self):
        self._files = {
            "checkpoints": [
                "sd15.safetensors",
                "sdxl_turbo.safetensors",
                "flux-dev-checkpoint.safetensors",
            ],
            "diffusion_models": [
                "flux1-schnell.safetensors",
            ],
            "vae": [
                "ae.safetensors",
            ],
            "text_encoders": [
                "clip_l.safetensors",
                "clip_g.safetensors",
                "t5xxl_fp16.safetensors",
            ],
            "upscale_models": [],
        }

    def get_filename_list(self, category):
        return list(self._files.get(category, []))

    def get_full_path_or_raise(self, category, name):
        if name in self._files.get(category, []):
            return f"/fake/{category}/{name}"
        raise FileNotFoundError(name)

    def get_full_path(self, category, name):
        if name in self._files.get(category, []):
            return f"/fake/{category}/{name}"
        return None

    def get_folder_paths(self, category):
        return [f"/fake/{category}"]


class ComfySDStub:
    def load_checkpoint_guess_config(
        self,
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=None,
    ):
        if ckpt_path.endswith("sd15.safetensors"):
            return (TaggedValue("MODEL::SD15"), TaggedValue("CLIP::SD15"), TaggedValue("VAE::SD15"))
        if ckpt_path.endswith("sdxl_turbo.safetensors"):
            return (TaggedValue("MODEL::SDXL_TURBO"), None, None)
        if ckpt_path.endswith("flux-dev-checkpoint.safetensors"):
            return (TaggedValue("MODEL::FLUX_DEV"), None, None)
        raise AssertionError(f"unexpected checkpoint path: {ckpt_path}")

    def load_diffusion_model(self, unet_path, model_options=None, disable_dynamic=False):
        return TaggedValue(f"MODEL::{pathlib.Path(unet_path).name}")


class CoreNodesStub:
    save_calls = []
    preview_calls = []

    class VAELoader:
        @staticmethod
        def vae_list(_self):
            return ["ae.safetensors"]

        def load_vae(self, vae_name):
            return (TaggedValue(f"VAE::{vae_name}"),)

    class CLIPLoader:
        def load_clip(self, clip_name, type="stable_diffusion", device="default"):
            return (TaggedValue(f"CLIP::{type}::{clip_name}"),)

    class DualCLIPLoader:
        def load_clip(self, clip_name1, clip_name2, type, device="default"):
            return (TaggedValue(f"CLIP::{type}::{clip_name1}+{clip_name2}"),)

    class SaveImage:
        def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
            CoreNodesStub.save_calls.append(
                {
                    "images": images,
                    "filename_prefix": filename_prefix,
                    "prompt": prompt,
                    "extra_pnginfo": extra_pnginfo,
                }
            )
            return {
                "ui": {
                    "images": [
                        {
                            "filename": f"{filename_prefix}.png",
                            "subfolder": "",
                            "type": "output",
                            "extra_pnginfo": extra_pnginfo,
                            "prompt": prompt,
                        }
                    ]
                }
            }

    class PreviewImage:
        def save_images(self, images, prompt=None, extra_pnginfo=None):
            CoreNodesStub.preview_calls.append(
                {
                    "images": images,
                    "prompt": prompt,
                    "extra_pnginfo": extra_pnginfo,
                }
            )
            return {
                "ui": {
                    "images": [
                        {
                            "filename": "preview.png",
                            "subfolder": "",
                            "type": "temp",
                            "extra_pnginfo": extra_pnginfo,
                            "prompt": prompt,
                        }
                    ]
                }
            }


class RecordingClip:
    def __init__(self, token_keys, calls=None):
        self.token_keys = tuple(token_keys)
        self.calls = [] if calls is None else calls
        self.layer = None

    def clone(self):
        cloned = RecordingClip(self.token_keys, self.calls)
        cloned.layer = self.layer
        return cloned

    def clip_layer(self, layer):
        self.layer = layer

    def tokenize(self, text):
        tokens = {"text": text}
        for key in self.token_keys:
            tokens[key] = [f"{key}::{text}"]
        return tokens

    def encode_from_tokens_scheduled(self, tokens, add_dict=None):
        self.calls.append(
            {
                "tokens": dict(tokens),
                "add_dict": dict(add_dict or {}),
                "layer": self.layer,
            }
        )
        return [[f"cond::{tokens.get('text', '')}", {"pooled_output": dict(add_dict or {})}]]


class FakeVAE:
    def __init__(self, encode_channels=4, encode_ratio=8, vae_name="vae.safetensors"):
        self.encode_channels = encode_channels
        self.encode_ratio = encode_ratio
        self.encoded_shapes = []
        self._lls_vae_name = vae_name

    def encode(self, pixels):
        self.encoded_shapes.append(tuple(pixels.shape))
        batch, height, width, _channels = pixels.shape
        return FakeTensor(
            (
                batch,
                self.encode_channels,
                max(1, height // self.encode_ratio),
                max(1, width // self.encode_ratio),
            )
        )

    def decode(self, latent_tensor):
        batch, _channels, height, width = latent_tensor.shape
        return FakeTensor((batch, height, width, 3))


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.dtype = "float32"
        self.layout = "strided"
        self.device = "cpu"
        self.is_nested = False

    def size(self):
        return self.shape

    def movedim(self, source, destination):
        rank = len(self.shape)
        source = source if source >= 0 else rank + source
        destination = destination if destination >= 0 else rank + destination
        order = list(range(rank))
        moved_axis = order.pop(source)
        order.insert(destination, moved_axis)
        return FakeTensor(tuple(self.shape[index] for index in order))

    def reshape(self, *shape):
        return FakeTensor(shape)


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def zeros(shape, device=None, dtype=None, layout=None):
        return FakeTensor(shape)


class FakeComfyUtils:
    def __init__(self):
        self.calls = []

    def common_upscale(self, samples, width, height, upscale_method, crop):
        self.calls.append(
            {
                "input_shape": tuple(samples.shape),
                "width": width,
                "height": height,
                "upscale_method": upscale_method,
                "crop": crop,
            }
        )
        batch, channels, _source_height, _source_width = samples.shape
        return FakeTensor((batch, channels, height, width))


class TestLoaderPromptRefactor(unittest.TestCase):
    def test_task_nodes_are_not_registered_in_contextless_mode(self):
        plugin = load_plugin_package()

        self.assertNotIn("LLSTaskController", plugin.NODE_CLASS_MAPPINGS)
        self.assertNotIn("LLSTaskInspector", plugin.NODE_CLASS_MAPPINGS)

    def test_loader_schema_uses_standard_ports_only(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleCheckpointLoader"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("MODEL", "CLIP", "VAE", "CLIP"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("model", "clip", "vae", "text_encoder"),
        )
        self.assertEqual(
            tuple(required.keys()),
            (
                "ckpt_name",
                "model_family",
                "load_mode",
                "vae_source",
                "text_encoder_source",
                "external_vae_name",
                "external_text_encoder_1",
                "external_text_encoder_2",
            ),
        )
        self.assertNotIn("optional", node_cls.INPUT_TYPES())

    def test_universal_loader_schema_exposes_single_text_encoder_output(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSUniversalModelLoader"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(
            node_cls.RETURN_TYPES,
            ("MODEL", "CLIP", "VAE", "STRING"),
        )
        self.assertEqual(
            node_cls.RETURN_NAMES,
            ("model", "text_encoder", "vae", "model_info"),
        )
        self.assertEqual(
            tuple(required.keys()),
            (
                "model_name",
                "model_family",
                "load_mode",
                "vae_source",
                "text_encoder_source",
                "text_encoder_1",
                "text_encoder_2",
                "vae_name",
            ),
        )
        self.assertEqual(required["model_family"][1]["default"], "Auto")
        self.assertEqual(required["text_encoder_source"][1]["default"], "auto")

    def test_loader_tags_sd15_resources_without_context_output(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, clip, vae, text_encoder = node.load_checkpoint(
                "sd15.safetensors",
                "SD1.5",
                "simple",
                "auto",
                "auto",
                "(auto)",
                "(auto)",
                "(auto)",
            )

        self.assertEqual(model.label, "MODEL::SD15")
        self.assertEqual(clip.label, "CLIP::SD15")
        self.assertIs(text_encoder, clip)
        self.assertEqual(vae.label, "VAE::SD15")
        self.assertEqual(model._lls_family, "SD1.5")
        self.assertEqual(model._lls_model_name, "sd15.safetensors")
        self.assertEqual(clip._lls_family, "SD1.5")
        self.assertEqual(clip._lls_text_encoder_type, "clip")
        self.assertEqual(vae._lls_family, "SD1.5")

    def test_universal_loader_loads_sd15_into_single_text_encoder_port(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSUniversalModelLoader()
            model, text_encoder, vae, model_info = node.load(
                "sd15.safetensors",
                "SD1.5",
                "simple",
                "auto",
                "auto",
                "(auto)",
                "(auto)",
                "(auto)",
            )

        payload = json.loads(model_info)
        self.assertEqual(model.label, "MODEL::SD15")
        self.assertEqual(text_encoder.label, "CLIP::SD15")
        self.assertEqual(vae.label, "VAE::SD15")
        self.assertEqual(payload["model_family"], "SD1.5")
        self.assertEqual(payload["checkpoint_name"], "sd15.safetensors")
        self.assertEqual(payload["text_encoder_source"], "embedded")
        self.assertEqual(payload["vae_source"], "embedded")

    def test_loader_tags_flux_resources_and_keeps_standard_text_encoder_alias(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, clip, vae, text_encoder = node.load_checkpoint(
                "diffusion_models/flux1-schnell.safetensors",
                "FLUX_SCHNELL",
                "advanced",
                "external",
                "external",
                "ae.safetensors",
                "clip_l.safetensors",
                "t5xxl_fp16.safetensors",
            )

        self.assertEqual(model.label, "MODEL::flux1-schnell.safetensors")
        self.assertEqual(clip.label, "CLIP::flux::clip_l.safetensors+t5xxl_fp16.safetensors")
        self.assertIs(text_encoder, clip)
        self.assertEqual(vae.label, "VAE::ae.safetensors")
        self.assertEqual(model._lls_family, "FLUX_SCHNELL")
        self.assertEqual(clip._lls_text_encoder_type, "flux_clip_l_t5xxl")
        self.assertEqual(clip._lls_model_name, "diffusion_models/flux1-schnell.safetensors")
        self.assertEqual(vae._lls_vae_name, "ae.safetensors")

    def test_universal_loader_loads_flux_dual_encoders_into_single_text_encoder_port(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSUniversalModelLoader()
            model, text_encoder, vae, model_info = node.load(
                "diffusion_models/flux1-schnell.safetensors",
                "FLUX_SCHNELL",
                "advanced",
                "external",
                "external",
                "clip_l.safetensors",
                "t5xxl_fp16.safetensors",
                "ae.safetensors",
            )

        payload = json.loads(model_info)
        self.assertEqual(model.label, "MODEL::flux1-schnell.safetensors")
        self.assertEqual(text_encoder.label, "CLIP::flux::clip_l.safetensors+t5xxl_fp16.safetensors")
        self.assertEqual(vae.label, "VAE::ae.safetensors")
        self.assertEqual(payload["model_family"], "FLUX_SCHNELL")
        self.assertEqual(payload["text_encoder_name_1"], "clip_l.safetensors")
        self.assertEqual(payload["text_encoder_name_2"], "t5xxl_fp16.safetensors")
        self.assertEqual(payload["text_encoder_name"], "clip_l.safetensors, t5xxl_fp16.safetensors")
        self.assertEqual(payload["vae_name"], "ae.safetensors")

    def test_universal_loader_loads_flux2_klein_with_qwen_and_flux2_vae(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        folder_paths = FolderPathsStub()
        folder_paths._files["diffusion_models"].append("flux-2-klein-9b.safetensors")
        folder_paths._files["text_encoders"].append("qwen_3_8b.safetensors")
        folder_paths._files["vae"].append("flux2-vae.safetensors")

        with mock.patch.object(loader_nodes, "folder_paths", folder_paths), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSUniversalModelLoader()
            model, text_encoder, vae, model_info = node.load(
                "diffusion_models/flux-2-klein-9b.safetensors",
                "Auto",
                "advanced",
                "external",
                "external",
                "qwen_3_8b.safetensors",
                "(auto)",
                "flux2-vae.safetensors",
            )

        payload = json.loads(model_info)
        self.assertEqual(model.label, "MODEL::flux-2-klein-9b.safetensors")
        self.assertEqual(text_encoder.label, "CLIP::flux2::qwen_3_8b.safetensors")
        self.assertEqual(vae.label, "VAE::flux2-vae.safetensors")
        self.assertEqual(payload["model_family"], "FLUX2_KLEIN")
        self.assertEqual(payload["text_encoder_name_1"], "qwen_3_8b.safetensors")
        self.assertEqual(payload["text_encoder_name_2"], "")
        self.assertEqual(payload["text_encoder_name"], "qwen_3_8b.safetensors")
        self.assertEqual(payload["vae_name"], "flux2-vae.safetensors")
        self.assertEqual(text_encoder._lls_text_encoder_type, "flux2_qwen")

    def test_loader_raises_clear_error_when_flux_text_encoder_is_missing(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        folder_paths = FolderPathsStub()
        folder_paths._files["text_encoders"] = ["clip_l.safetensors"]
        with mock.patch.object(loader_nodes, "folder_paths", folder_paths), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            with self.assertRaisesRegex(RuntimeError, "t5xxl_fp8_e4m3fn.safetensors|t5xxl_fp16.safetensors"):
                node.load_checkpoint(
                    "diffusion_models/flux1-schnell.safetensors",
                    "FLUX_DEV",
                    "simple",
                    "external",
                    "external",
                    "ae.safetensors",
                    "clip_l.safetensors",
                    "(auto)",
                )

    def test_prompt_encode_schema_uses_standard_clip_ports_and_prompt_info(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("CONDITIONING", "CONDITIONING", "STRING"))
        self.assertEqual(required["model_family"][1]["default"], "Auto")
        self.assertEqual(optional["text_encoder"][0], "CLIP")
        self.assertEqual(optional["clip"][0], "CLIP")
        self.assertNotIn("task_context", optional)

    def test_universal_prompt_encode_schema_requires_single_text_encoder_input(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSUniversalPromptEncode"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("CONDITIONING", "CONDITIONING", "STRING"))
        self.assertEqual(required["text_encoder"][0], "CLIP")
        self.assertEqual(required["clip_skip"][1]["default"], -1)
        self.assertNotIn("clip", required)
        self.assertEqual(optional["model_info"][0], "STRING")

    def test_prompt_encode_dispatches_sdxl_using_inferred_family_defaults(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip(("g", "l"))

        positive, negative, prompt_info = node.encode(
            clip=clip,
            positive_prompt="a castle",
            negative_prompt="low quality",
            clip_skip=-1,
            model_family="Auto",
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(
            clip.calls[0]["add_dict"],
            {
                "width": 1024,
                "height": 1024,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": 1024,
                "target_height": 1024,
            },
        )
        payload = json.loads(prompt_info)
        self.assertEqual(payload["model_family"], "SDXL")
        self.assertEqual(payload["prompt_mode"], "sdxl")
        self.assertEqual(payload["clip_skip"], -1)
        self.assertEqual(positive[0][0], "cond::a castle")
        self.assertEqual(negative[0][0], "cond::low quality")

    def test_prompt_encode_dispatches_flux_and_neutralizes_negative_prompt(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip(("t5xxl",))

        positive, negative, prompt_info = node.encode(
            clip=clip,
            positive_prompt="a robot in a forest",
            negative_prompt="ugly, blurry",
            clip_skip=-1,
            model_family="Auto",
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(clip.calls[0]["add_dict"]["guidance"], 3.5)
        self.assertEqual(clip.calls[1]["tokens"]["text"], "")
        payload = json.loads(prompt_info)
        self.assertEqual(payload["model_family"], "FLUX_DEV")
        self.assertEqual(payload["negative_mode"], "ignored_for_flux")
        self.assertEqual(payload["prompt_mode"], "flux")
        self.assertEqual(positive[0][0], "cond::a robot in a forest")
        self.assertEqual(negative[0][0], "cond::")

    def test_universal_prompt_encode_dispatches_sdxl_from_model_info(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSUniversalPromptEncode"]
        node = node_cls()
        clip = RecordingClip(("g", "l"))

        positive, negative, prompt_info = node.encode(
            text_encoder=clip,
            positive_prompt="a castle",
            negative_prompt="low quality",
            clip_skip=-1,
            model_info=json.dumps({"model_family": "SDXL", "checkpoint_name": "sdxl_turbo.safetensors"}),
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(clip.calls[0]["add_dict"]["width"], 1024)
        payload = json.loads(prompt_info)
        self.assertEqual(payload["model_family"], "SDXL")
        self.assertEqual(payload["prompt_mode"], "sdxl")
        self.assertEqual(payload["checkpoint_name"], "sdxl_turbo.safetensors")
        self.assertEqual(positive[0][0], "cond::a castle")
        self.assertEqual(negative[0][0], "cond::low quality")

    def test_universal_prompt_encode_dispatches_flux_from_model_info(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSUniversalPromptEncode"]
        node = node_cls()
        clip = RecordingClip(("t5xxl",))

        positive, negative, prompt_info = node.encode(
            text_encoder=clip,
            positive_prompt="a robot in a forest",
            negative_prompt="ugly, blurry",
            clip_skip=-1,
            model_info=json.dumps(
                {
                    "model_family": "FLUX_DEV",
                    "checkpoint_name": "diffusion_models/flux1-schnell.safetensors",
                    "text_encoder_name": "clip_l.safetensors, t5xxl_fp16.safetensors",
                }
            ),
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(clip.calls[0]["add_dict"]["guidance"], 3.5)
        self.assertEqual(clip.calls[1]["tokens"]["text"], "")
        payload = json.loads(prompt_info)
        self.assertEqual(payload["model_family"], "FLUX_DEV")
        self.assertEqual(payload["prompt_mode"], "flux")
        self.assertEqual(payload["checkpoint_name"], "diffusion_models/flux1-schnell.safetensors")
        self.assertEqual(payload["text_encoder_name"], "clip_l.safetensors, t5xxl_fp16.safetensors")
        self.assertEqual(positive[0][0], "cond::a robot in a forest")
        self.assertEqual(negative[0][0], "cond::")

    def test_empty_latent_uses_model_family_inference_for_flux_shape(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()
        model = TaggedValue("MODEL::flux1-schnell.safetensors")
        model._lls_family = "FLUX_SCHNELL"

        with mock.patch.object(latent_nodes, "torch", FakeTorch()):
            latent, width, height, latent_info = node.create_empty_latent(
                "Family Default",
                512,
                512,
                2,
                model_family="Auto",
                model=model,
            )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 64, 64))
        self.assertEqual(latent["downscale_ratio_spacial"], 16)
        self.assertEqual(payload["model_family"], "FLUX_SCHNELL")
        self.assertEqual(payload["task_mode"], "txt2img")
        self.assertEqual(payload["batch_size"], 2)

    def test_empty_latent_schema_accepts_optional_img2img_inputs(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleEmptyLatent"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(
            tuple(required.keys()),
            (
                "size_preset",
                "width",
                "height",
                "batch_size",
                "model_family",
                "resize_mode",
            ),
        )
        self.assertEqual(required["resize_mode"][0], ["keep_aspect", "crop_center", "stretch", "none"])
        self.assertEqual(optional["image"][0], "IMAGE")
        self.assertEqual(optional["vae"][0], "VAE")
        self.assertEqual(optional["model"][0], "MODEL")

    def test_empty_latent_can_encode_image_using_family_default_size(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes
        from lls_node_test_refactor.image import nodes as image_nodes

        comfy_utils = FakeComfyUtils()
        model = TaggedValue("MODEL::SDXL")
        model._lls_family = "SDXL"
        vae = FakeVAE(vae_name="embedded")

        with mock.patch.object(image_nodes, "comfy_utils", comfy_utils):
            node = latent_nodes.LLSSimpleEmptyLatent()
            latent, width, height, latent_info = node.create_empty_latent(
                "Family Default",
                512,
                512,
                3,
                model_family="Auto",
                resize_mode="keep_aspect",
                model=model,
                image=FakeTensor((1, 768, 1536, 3)),
                vae=vae,
            )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (1024, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 128))
        self.assertEqual(latent["downscale_ratio_spacial"], 8)
        self.assertEqual(vae.encoded_shapes, [(1, 512, 1024, 3)])
        self.assertEqual(payload["task_mode"], "img2img")
        self.assertEqual(payload["latent_source"], "image_encode")
        self.assertEqual(payload["model_family"], "SDXL")
        self.assertEqual(payload["size_preset"], "Family Default")
        self.assertEqual(payload["batch_size"], 1)

    def test_empty_latent_can_encode_image_using_custom_size(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes
        from lls_node_test_refactor.image import nodes as image_nodes

        comfy_utils = FakeComfyUtils()
        vae = FakeVAE()

        with mock.patch.object(image_nodes, "comfy_utils", comfy_utils):
            node = latent_nodes.LLSSimpleEmptyLatent()
            latent, width, height, latent_info = node.create_empty_latent(
                "Custom",
                640,
                768,
                4,
                model_family="SD1.5",
                resize_mode="stretch",
                image=FakeTensor((1, 512, 512, 3)),
                vae=vae,
            )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (640, 768))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 96, 80))
        self.assertEqual(vae.encoded_shapes, [(1, 768, 640, 3)])
        self.assertEqual(payload["task_mode"], "img2img")
        self.assertEqual(payload["size_preset"], "Custom")
        self.assertEqual(payload["resize_mode"], "stretch")

    def test_empty_latent_requires_vae_when_image_is_connected(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()

        with self.assertRaisesRegex(RuntimeError, "Missing VAE"):
            node.create_empty_latent(
                "Custom",
                640,
                640,
                1,
                model_family="SD1.5",
                resize_mode="stretch",
                image=FakeTensor((1, 512, 512, 3)),
            )

    def test_ksampler_reads_model_and_latent_inference_for_img2img(self):
        load_plugin_package()
        from lls_node_test_refactor.sampling import nodes as sampling_nodes

        recorded = {}
        model = TaggedValue("MODEL::SDXL_TURBO")
        model._lls_family = "SDXL_TURBO"

        def fake_common_ksampler(**kwargs):
            recorded.update(kwargs)
            return kwargs["latent"]

        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            object(),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            side_effect=fake_common_ksampler,
        ):
            node = sampling_nodes.LLSSimpleKSampler()
            latent, sample_info = node.sample(
                model=model,
                positive=[["pos", {}]],
                negative=[["neg", {}]],
                latent_image={"samples": FakeTensor((1, 4, 64, 64)), "source": "image_encode"},
                model_family="Auto",
                quality_preset="Family Default",
                seed=123,
                steps=30,
                cfg=7.0,
                sampler_name="euler_ancestral",
                scheduler="karras",
                denoise=1.0,
                flux_guidance=9.0,
            )

        sample_data = json.loads(sample_info)
        self.assertEqual(recorded["steps"], 4)
        self.assertEqual(recorded["cfg"], 1.0)
        self.assertEqual(recorded["denoise"], 1.0)
        self.assertEqual(sample_data["family"], "SDXL_TURBO")
        self.assertEqual(sample_data["task_mode"], "img2img")
        self.assertEqual(sample_data["sampler_name"], "euler")
        self.assertEqual(latent["samples"].shape, (1, 4, 64, 64))

    def test_ksampler_schema_keeps_legacy_widget_order_for_workflow_compatibility(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleKSampler"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(
            tuple(required.keys()),
            (
                "model",
                "positive",
                "negative",
                "latent_image",
                "quality_preset",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "denoise_mode",
                "adapter_mode",
                "flux_guidance",
                "model_family",
            ),
        )
        self.assertEqual(required["model_family"][1]["default"], "Auto")

    def test_ksampler_accepts_blank_flux_guidance_for_legacy_workflows(self):
        load_plugin_package()
        from lls_node_test_refactor.sampling import nodes as sampling_nodes

        required = sampling_nodes.LLSSimpleKSampler.INPUT_TYPES()["required"]
        flux_input_type, flux_input_opts = required["flux_guidance"]
        self.assertEqual(flux_input_type, "STRING,FLOAT,INT")
        self.assertEqual(flux_input_opts["widgetType"], "FLOAT")
        self.assertTrue(sampling_nodes.LLSSimpleKSampler.VALIDATE_INPUTS(flux_guidance=""))

        recorded = {}
        model = TaggedValue("MODEL::FLUX_DEV")
        model._lls_family = "FLUX_DEV"

        def fake_common_ksampler(**kwargs):
            recorded.update(kwargs)
            return kwargs["latent"]

        fake_node_helpers = types.SimpleNamespace(
            conditioning_set_values=lambda conditioning, values: [{"conditioning": conditioning, **values}]
        )
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            object(),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            side_effect=fake_common_ksampler,
        ), mock.patch.object(
            sampling_nodes,
            "node_helpers",
            fake_node_helpers,
        ):
            node = sampling_nodes.LLSSimpleKSampler()
            latent, sample_info = node.sample(
                model=model,
                positive=[["pos", {}]],
                negative=[["neg", {}]],
                latent_image={"samples": FakeTensor((1, 128, 64, 64)), "source": "empty_latent"},
                model_family="Auto",
                quality_preset="Manual",
                seed=123,
                steps=20,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                flux_guidance="",
            )

        sample_data = json.loads(sample_info)
        self.assertEqual(recorded["positive"][0]["guidance"], 3.5)
        self.assertEqual(recorded["negative"][0]["guidance"], 3.5)
        self.assertEqual(sample_data["guidance"], 3.5)
        self.assertEqual(latent["samples"].shape, (1, 128, 64, 64))

    def test_vae_decode_returns_decode_info_for_flux(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEDecode"]
        node = node_cls()

        image, decode_info = node.decode(
            {"samples": FakeTensor((1, 128, 64, 64)), "downscale_ratio_spacial": 16},
            FakeVAE(vae_name="ae.safetensors"),
        )

        payload = json.loads(decode_info)
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertEqual(payload["width"], 1024)
        self.assertEqual(payload["height"], 1024)
        self.assertEqual(payload["vae_name"], "ae.safetensors")
        self.assertEqual(payload["decode_stage"], "vae_decode")

    def test_vae_encode_schema_exposes_img2img_inputs_without_context_port(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEEncode"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.CATEGORY, "LLS/Image")
        self.assertEqual(node_cls.RETURN_TYPES, ("LATENT", "INT", "INT", "STRING"))
        self.assertEqual(required["image"][0], "IMAGE")
        self.assertEqual(required["vae"][0], "VAE")
        self.assertEqual(required["resize_mode"][0], ["keep_aspect", "crop_center", "stretch", "none"])
        self.assertEqual(required["size_source"][0], ["input_image", "custom", "model_recommended"])
        self.assertEqual(required["model_family"][1]["default"], "Auto")
        self.assertEqual(optional["model"][0], "MODEL")
        self.assertEqual(optional["clip"][0], "CLIP")
        self.assertNotIn("task_context", optional)

    def test_vae_encode_uses_inferred_model_recommended_size_for_img2img(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        comfy_utils = FakeComfyUtils()
        vae = FakeVAE()
        node = image_nodes.LLSSimpleVAEEncode()
        model = TaggedValue("MODEL::SDXL")
        model._lls_family = "SDXL"
        vae._lls_vae_name = "embedded"

        with mock.patch.object(image_nodes, "comfy_utils", comfy_utils):
            latent, width, height, latent_info = node.encode(
                image=FakeTensor((1, 768, 1536, 3)),
                vae=vae,
                resize_mode="keep_aspect",
                size_source="model_recommended",
                width=512,
                height=512,
                model_family="Auto",
                model=model,
            )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (1024, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 128))
        self.assertEqual(latent["downscale_ratio_spacial"], 8)
        self.assertEqual(vae.encoded_shapes, [(1, 512, 1024, 3)])
        self.assertEqual(payload["task_mode"], "img2img")
        self.assertEqual(payload["latent_source"], "image_encode")
        self.assertEqual(payload["model_family"], "SDXL")
        self.assertEqual(payload["vae_name"], "embedded")
        self.assertEqual(
            comfy_utils.calls,
            [
                {
                    "input_shape": (1, 3, 768, 1536),
                    "width": 1024,
                    "height": 512,
                    "upscale_method": "bilinear",
                    "crop": "disabled",
                }
            ],
        )

    def test_vae_encode_input_image_mode_can_passthrough_aligned_size(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        vae = FakeVAE()
        node = image_nodes.LLSSimpleVAEEncode()
        latent, width, height, latent_info = node.encode(
            image=FakeTensor((1, 512, 768, 3)),
            vae=vae,
            resize_mode="none",
            size_source="input_image",
            width=1024,
            height=1024,
            model_family="SD1.5",
        )

        payload = json.loads(latent_info)
        self.assertEqual((width, height), (768, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 96))
        self.assertEqual(latent["downscale_ratio_spacial"], 8)
        self.assertEqual(vae.encoded_shapes, [(1, 512, 768, 3)])
        self.assertEqual(payload["latent_source"], "image_encode")
        self.assertEqual(payload["size_source"], "input_image")
        self.assertEqual(payload["resize_mode"], "none")

    def test_save_image_schema_uses_info_strings_only(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSaveImage"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(required["filename_prefix"][0], "STRING")
        self.assertEqual(required["output_mode"][0], ["save", "preview_only"])
        self.assertEqual(required["save_metadata"][0], "BOOLEAN")
        self.assertNotIn("image", required)
        self.assertNotIn("model", optional)
        self.assertNotIn("clip", optional)
        self.assertNotIn("vae", optional)
        self.assertEqual(optional["image"], ("IMAGE",))
        self.assertEqual(optional["mask"], ("MASK",))
        self.assertEqual(optional["prompt_info"], ("STRING", {"forceInput": True}))
        self.assertEqual(optional["latent_info"], ("STRING", {"forceInput": True}))
        self.assertEqual(optional["sample_info"], ("STRING", {"forceInput": True}))
        self.assertEqual(optional["decode_info"], ("STRING", {"forceInput": True}))
        self.assertEqual(optional["upscale_info"], ("STRING", {"forceInput": True}))

    def test_save_image_merges_metadata_from_info_strings(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 512, 512, 3)),
                filename_prefix="LLS",
                save_metadata=True,
                prompt_info=json.dumps(
                    {
                        "positive_prompt": "cat",
                        "negative_prompt": "bad",
                        "model_family": "SD1.5",
                        "checkpoint_name": "sd15.safetensors",
                        "text_encoder_name": "clip_l.safetensors",
                    }
                ),
                sample_info=json.dumps(
                    {
                        "seed": 42,
                        "steps": 20,
                        "cfg": 7.0,
                        "guidance": None,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                    }
                ),
                decode_info=json.dumps(
                    {
                        "vae_name": "VAE::SD15",
                        "width": 512,
                        "height": 512,
                        "batch_size": 1,
                    }
                ),
                upscale_info=json.dumps({"mode": "none", "scale": 1.0}),
            )

        metadata = result["ui"]["images"][0]["extra_pnginfo"]["lls_metadata"]
        self.assertEqual(metadata["positive_prompt"], "cat")
        self.assertEqual(metadata["checkpoint_name"], "sd15.safetensors")
        self.assertEqual(metadata["text_encoder_name"], "clip_l.safetensors")
        self.assertEqual(metadata["vae_name"], "VAE::SD15")
        self.assertEqual(metadata["steps"], 20)
        self.assertEqual(metadata["upscale_mode"], "none")

    def test_save_image_preview_only_uses_native_preview_node_without_saving(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 512, 512, 3)),
                filename_prefix="LLS",
                save_metadata=True,
                output_mode="preview_only",
            )

        self.assertEqual(CoreNodesStub.save_calls, [])
        self.assertEqual(len(CoreNodesStub.preview_calls), 1)
        self.assertEqual(result["ui"]["images"][0]["type"], "temp")
        self.assertIsNone(result["ui"]["images"][0]["extra_pnginfo"])

    def test_save_image_preview_only_emits_image_and_mask_previews(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            with mock.patch.object(
                image_nodes,
                "mask_to_image",
                side_effect=lambda mask: FakeTensor((mask.shape[0], mask.shape[1], mask.shape[2], 3)),
                create=True,
            ):
                node = image_nodes.LLSSaveImage()
                result = node.save(
                    image=FakeTensor((1, 4, 4, 3)),
                    mask=FakeTensor((1, 4, 4)),
                    filename_prefix="LLS",
                    save_metadata=True,
                    output_mode="preview_only",
                )

        self.assertEqual(CoreNodesStub.save_calls, [])
        self.assertEqual(len(CoreNodesStub.preview_calls), 2)
        self.assertEqual(tuple(CoreNodesStub.preview_calls[0]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(tuple(CoreNodesStub.preview_calls[1]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(result["ui"]["images"][0]["type"], "temp")
        self.assertEqual(len(result["ui"]["images"]), 2)
        self.assertIsNone(result["ui"]["images"][1]["extra_pnginfo"])

    def test_save_image_preview_only_supports_mask_without_image(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            with mock.patch.object(
                image_nodes,
                "mask_to_image",
                side_effect=lambda mask: FakeTensor((mask.shape[0], mask.shape[1], mask.shape[2], 3)),
                create=True,
            ):
                node = image_nodes.LLSSaveImage()
                result = node.save(
                    image=None,
                    mask=FakeTensor((1, 4, 4)),
                    filename_prefix="LLS",
                    save_metadata=True,
                    output_mode="preview_only",
                )

        self.assertEqual(CoreNodesStub.save_calls, [])
        self.assertEqual(len(CoreNodesStub.preview_calls), 1)
        self.assertEqual(tuple(CoreNodesStub.preview_calls[0]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(len(result["ui"]["images"]), 1)
        self.assertEqual(result["ui"]["images"][0]["type"], "temp")

    def test_save_image_save_mode_emits_separate_mask_file_without_lls_metadata(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            with mock.patch.object(
                image_nodes,
                "mask_to_image",
                side_effect=lambda mask: FakeTensor((mask.shape[0], mask.shape[1], mask.shape[2], 3)),
                create=True,
            ):
                node = image_nodes.LLSSaveImage()
                result = node.save(
                    image=FakeTensor((1, 4, 4, 3)),
                    mask=FakeTensor((1, 4, 4)),
                    filename_prefix="LLS",
                    save_metadata=True,
                    prompt_info=json.dumps({"positive_prompt": "cat"}),
                )

        self.assertEqual(len(CoreNodesStub.preview_calls), 0)
        self.assertEqual(len(CoreNodesStub.save_calls), 2)
        self.assertEqual(CoreNodesStub.save_calls[0]["filename_prefix"], "LLS")
        self.assertEqual(CoreNodesStub.save_calls[1]["filename_prefix"], "LLS_mask")
        self.assertIn("lls_metadata", CoreNodesStub.save_calls[0]["extra_pnginfo"])
        self.assertEqual(CoreNodesStub.save_calls[1]["extra_pnginfo"], {})
        self.assertEqual(tuple(CoreNodesStub.save_calls[1]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(len(result["ui"]["images"]), 2)

    def test_save_image_save_mode_supports_mask_without_image(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        CoreNodesStub.save_calls = []
        CoreNodesStub.preview_calls = []

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            with mock.patch.object(
                image_nodes,
                "mask_to_image",
                side_effect=lambda mask: FakeTensor((mask.shape[0], mask.shape[1], mask.shape[2], 3)),
                create=True,
            ):
                node = image_nodes.LLSSaveImage()
                result = node.save(
                    image=None,
                    mask=FakeTensor((1, 4, 4)),
                    filename_prefix="LLS",
                    save_metadata=True,
                )

        self.assertEqual(len(CoreNodesStub.preview_calls), 0)
        self.assertEqual(len(CoreNodesStub.save_calls), 1)
        self.assertEqual(CoreNodesStub.save_calls[0]["filename_prefix"], "LLS")
        self.assertEqual(CoreNodesStub.save_calls[0]["extra_pnginfo"], {})
        self.assertEqual(tuple(CoreNodesStub.save_calls[0]["images"].shape), (1, 4, 4, 3))
        self.assertEqual(len(result["ui"]["images"]), 1)

    def test_save_image_rejects_missing_image_and_mask(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()

            with self.assertRaisesRegex(RuntimeError, "At least one of IMAGE or MASK"):
                node.save(
                    image=None,
                    mask=None,
                    filename_prefix="LLS",
                    save_metadata=True,
                    output_mode="preview_only",
                )

    def test_generation_config_uses_model_inference_without_context(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSGenerationConfig"]
        node = node_cls()
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]
        model = TaggedValue("MODEL::FLUX_DEV")
        model._lls_family = "FLUX_DEV"

        self.assertEqual(required["model_family"][1]["default"], "Auto")
        self.assertEqual(optional["model"][0], "MODEL")
        self.assertEqual(optional["clip"][0], "CLIP")
        self.assertEqual(
            tuple(required.keys()),
            ("quality_preset", "size_preset", "model_family"),
        )

        width, height, steps, cfg, guidance, sampler_name, scheduler, denoise, config_info = node.execute(
            quality_preset="Family Default",
            size_preset="Family Default",
            model_family="Auto",
            model=model,
        )

        payload = json.loads(config_info)
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual((steps, cfg, guidance), (20, 1.0, 3.5))
        self.assertEqual(sampler_name, "euler")
        self.assertEqual(scheduler, "simple")
        self.assertEqual(denoise, 1.0)
        self.assertEqual(payload["family"], "FLUX_DEV")


if __name__ == "__main__":
    unittest.main()
