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
            return ("MODEL::SD15", "CLIP::SD15", "VAE::SD15")
        if ckpt_path.endswith("sdxl_turbo.safetensors"):
            return ("MODEL::SDXL_TURBO", None, None)
        if ckpt_path.endswith("flux-dev-checkpoint.safetensors"):
            return ("MODEL::FLUX_DEV", None, None)
        raise AssertionError(f"unexpected checkpoint path: {ckpt_path}")

    def load_diffusion_model(self, unet_path, model_options=None, disable_dynamic=False):
        return f"MODEL::{pathlib.Path(unet_path).name}"


class CoreNodesStub:
    class VAELoader:
        @staticmethod
        def vae_list(_self):
            return ["ae.safetensors"]

        def load_vae(self, vae_name):
            return (f"VAE::{vae_name}",)

    class CLIPLoader:
        def load_clip(self, clip_name, type="stable_diffusion", device="default"):
            return (f"CLIP::{type}::{clip_name}",)

    class DualCLIPLoader:
        def load_clip(self, clip_name1, clip_name2, type, device="default"):
            return (f"CLIP::{type}::{clip_name1}+{clip_name2}",)

    class SaveImage:
        def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
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


class RecordingClip:
    def __init__(self, calls=None):
        self.calls = [] if calls is None else calls
        self.layer = None

    def clone(self):
        cloned = RecordingClip(self.calls)
        cloned.layer = self.layer
        return cloned

    def clip_layer(self, layer):
        self.layer = layer

    def tokenize(self, text):
        return {
            "text": text,
            "g": [f"g::{text}"],
            "l": [f"l::{text}"],
            "t5xxl": [f"t5::{text}"],
        }

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
    def __init__(self, encode_channels=4, encode_ratio=8):
        self.encode_channels = encode_channels
        self.encode_ratio = encode_ratio
        self.encoded_shapes = []

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
    def test_task_controller_generates_context_with_recommended_defaults(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSTaskController"]
        node = node_cls()
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(node_cls.CATEGORY, "LLS-node")
        self.assertEqual(node_cls.RETURN_TYPES, ("LLS_TASK_CONTEXT",))
        self.assertEqual(
            tuple(required.keys()),
            (
                "task_mode",
                "model_family",
                "workflow_preset",
                "quality_preset",
                "enable_upscale",
                "enable_controlnet",
                "enable_reference",
                "use_external_vae",
                "use_external_text_encoder",
            ),
        )

        (task_context,) = node.execute(
            "img2img",
            "SDXL",
            "standard",
            "balanced",
            True,
            False,
            False,
            False,
            False,
        )

        self.assertEqual(task_context["task_mode"], "img2img")
        self.assertEqual(task_context["model_family"], "SDXL")
        self.assertEqual(task_context["recommended_width"], 1024)
        self.assertEqual(task_context["recommended_height"], 1024)
        self.assertEqual(task_context["recommended_cfg"], 7.0)
        self.assertEqual(task_context["recommended_denoise"], 0.5)
        self.assertEqual(task_context["source"], "LLS Task Controller")

    def test_loader_schema_exposes_task_context_and_source_controls(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleCheckpointLoader"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("MODEL", "LLS_TEXT_ENCODER", "VAE", "LLS_TASK_CONTEXT"))
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
        family_choices = required["model_family"][0]
        self.assertIn("SD1.5", family_choices)
        self.assertIn("SDXL_TURBO", family_choices)
        self.assertIn("FLUX_SCHNELL", family_choices)
        self.assertIn("FLUX_DEV", family_choices)
        self.assertEqual(optional["task_context"][0], "LLS_TASK_CONTEXT")

    def test_loader_returns_updated_task_context_for_sd15_embedded_resources(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, text_encoder, vae, task_context = node.load_checkpoint(
                "sd15.safetensors",
                "SD1.5",
                "simple",
                "auto",
                "auto",
                "(auto)",
                "(auto)",
                "(auto)",
                task_context={"task_mode": "txt2img", "model_family": "SD1.5"},
            )

        self.assertEqual(model, "MODEL::SD15")
        self.assertEqual(text_encoder, "CLIP::SD15")
        self.assertEqual(vae, "VAE::SD15")
        self.assertEqual(task_context["resolved_model_family"], "SD1.5")
        self.assertEqual(task_context["checkpoint_name"], "sd15.safetensors")
        self.assertEqual(task_context["text_encoder_type"], "clip")
        self.assertEqual(task_context["vae_source"], "embedded")
        self.assertEqual(task_context["latent_channels"], 4)
        self.assertTrue(task_context["supports_img2img"])
        self.assertTrue(task_context["supports_clip_skip"])
        self.assertFalse(task_context["supports_flux_guidance"])
        self.assertEqual(task_context["recommended_width"], 512)

    def test_loader_builds_flux_resources_and_updates_task_context(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, text_encoder, vae, task_context = node.load_checkpoint(
                "diffusion_models/flux1-schnell.safetensors",
                "FLUX_SCHNELL",
                "advanced",
                "external",
                "external",
                "ae.safetensors",
                "clip_l.safetensors",
                "t5xxl_fp16.safetensors",
                task_context={"task_mode": "txt2img", "model_family": "FLUX"},
            )

        self.assertEqual(model, "MODEL::flux1-schnell.safetensors")
        self.assertEqual(text_encoder, "CLIP::flux::clip_l.safetensors+t5xxl_fp16.safetensors")
        self.assertEqual(vae, "VAE::ae.safetensors")
        self.assertEqual(task_context["resolved_model_family"], "FLUX_SCHNELL")
        self.assertEqual(task_context["text_encoder_type"], "flux_clip_l_t5xxl")
        self.assertEqual(task_context["vae_source"], "external")
        self.assertEqual(task_context["text_encoder_source"], "external")
        self.assertEqual(task_context["latent_channels"], 128)
        self.assertTrue(task_context["supports_flux_guidance"])
        self.assertFalse(task_context["supports_clip_skip"])
        self.assertEqual(task_context["recommended_cfg"], 1.0)

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
                    task_context={"task_mode": "txt2img", "model_family": "FLUX"},
                )

    def test_prompt_encode_schema_uses_task_context_port(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(optional["text_encoder"][0], "LLS_TEXT_ENCODER")
        self.assertEqual(optional["task_context"][0], "LLS_TASK_CONTEXT")

    def test_prompt_encode_dispatches_sdxl_using_task_context_dimensions(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        task_context = {
            "model_family": "SDXL",
            "resolved_model_family": "SDXL",
            "text_encoder_type": "sdxl_dual_clip",
            "recommended_width": 1216,
            "recommended_height": 832,
            "supports_clip_skip": True,
        }

        positive, negative, next_context = node.encode(
            clip,
            "a castle",
            "low quality",
            -1,
            task_context=task_context,
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertIn("g", clip.calls[0]["tokens"])
        self.assertIn("l", clip.calls[0]["tokens"])
        self.assertEqual(
            clip.calls[0]["add_dict"],
            {
                "width": 1216,
                "height": 832,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": 1216,
                "target_height": 832,
            },
        )
        self.assertEqual(positive[0][0], "cond::a castle")
        self.assertEqual(negative[0][0], "cond::low quality")
        self.assertEqual(next_context["resolved_model_family"], "SDXL")
        self.assertEqual(next_context["prompt_mode"], "sdxl")
        self.assertEqual(next_context["positive_prompt_length"], len("a castle"))
        self.assertEqual(next_context["negative_prompt_length"], len("low quality"))

    def test_prompt_encode_dispatches_flux_and_neutralizes_negative_prompt(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        task_context = {
            "model_family": "FLUX",
            "resolved_model_family": "FLUX_DEV",
            "guidance": 3.5,
            "supports_flux_guidance": True,
            "text_encoder_type": "flux_clip_l_t5xxl",
        }

        positive, negative, next_context = node.encode(
            clip,
            "a robot in a forest",
            "ugly, blurry",
            -1,
            task_context=task_context,
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertIn("t5xxl", clip.calls[0]["tokens"])
        self.assertEqual(clip.calls[0]["add_dict"]["guidance"], 3.5)
        self.assertEqual(clip.calls[1]["tokens"]["text"], "")
        self.assertEqual(positive[0][0], "cond::a robot in a forest")
        self.assertEqual(negative[0][0], "cond::")
        self.assertEqual(next_context["negative_mode"], "ignored_for_flux")
        self.assertEqual(next_context["prompt_mode"], "flux")

    def test_empty_latent_uses_task_context_recommended_size_and_flux_shape(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()
        task_context = {
            "model_family": "FLUX",
            "resolved_model_family": "FLUX_SCHNELL",
            "latent_channels": 128,
            "downscale_ratio": 16,
            "recommended_width": 1024,
            "recommended_height": 1024,
        }

        with mock.patch.object(latent_nodes, "torch", FakeTorch()):
            latent, width, height, next_context = node.create_empty_latent(
                "Family Default",
                512,
                512,
                2,
                task_context=task_context,
            )

        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 64, 64))
        self.assertEqual(latent["downscale_ratio_spacial"], 16)
        self.assertEqual(next_context["resolved_model_family"], "FLUX_SCHNELL")
        self.assertEqual(next_context["latent_source"], "empty_latent")
        self.assertEqual(next_context["final_width"], 1024)
        self.assertEqual(next_context["batch_size"], 2)

    def test_ksampler_reads_task_context_and_updates_it_for_img2img(self):
        load_plugin_package()
        from lls_node_test_refactor.sampling import nodes as sampling_nodes

        recorded = {}

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
            latent, sample_info, task_context = node.sample(
                model="model",
                positive=[["pos", {}]],
                negative=[["neg", {}]],
                latent_image={"samples": FakeTensor((1, 4, 64, 64))},
                quality_preset="Family Default",
                seed=123,
                steps=30,
                cfg=7.0,
                sampler_name="euler_ancestral",
                scheduler="karras",
                denoise=1.0,
                flux_guidance=9.0,
                task_context={
                    "model_family": "SDXL",
                    "resolved_model_family": "SDXL_TURBO",
                    "task_mode": "img2img",
                    "latent_source": "image_encode",
                    "recommended_steps": 4,
                    "recommended_cfg": 1.0,
                    "recommended_denoise": 0.5,
                    "recommended_sampler": "euler",
                    "recommended_scheduler": "normal",
                    "supports_flux_guidance": False,
                },
            )

        sample_data = json.loads(sample_info)
        self.assertEqual(recorded["steps"], 4)
        self.assertEqual(recorded["cfg"], 1.0)
        self.assertEqual(recorded["denoise"], 0.5)
        self.assertEqual(sample_data["family"], "SDXL_TURBO")
        self.assertEqual(sample_data["sampler_name"], "euler")
        self.assertEqual(latent["samples"].shape, (1, 4, 64, 64))
        self.assertTrue(task_context["sampled"])
        self.assertEqual(task_context["task_mode"], "img2img")
        self.assertEqual(task_context["denoise"], 0.5)

    def test_vae_decode_updates_task_context_for_flux(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEDecode"]
        node = node_cls()

        image, task_context = node.decode(
            {"samples": FakeTensor((1, 128, 64, 64))},
            FakeVAE(),
            task_context={"resolved_model_family": "FLUX_DEV", "downscale_ratio": 16, "vae_name": "ae.safetensors"},
        )

        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertEqual(task_context["final_width"], 1024)
        self.assertEqual(task_context["vae_name"], "ae.safetensors")
        self.assertTrue(task_context["decoded"])
        self.assertEqual(task_context["decode_stage"], "vae_decode")

    def test_vae_encode_schema_exposes_img2img_inputs_and_task_context_port(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEEncode"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("LATENT", "INT", "INT", "LLS_TASK_CONTEXT"))
        self.assertEqual(required["image"][0], "IMAGE")
        self.assertEqual(required["vae"][0], "VAE")
        self.assertEqual(required["resize_mode"][0], ["keep_aspect", "crop_center", "stretch", "none"])
        self.assertEqual(required["size_source"][0], ["input_image", "custom", "model_recommended"])
        self.assertEqual(optional["task_context"][0], "LLS_TASK_CONTEXT")

    def test_vae_encode_uses_task_context_recommended_size_for_img2img(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        comfy_utils = FakeComfyUtils()
        vae = FakeVAE()
        node = image_nodes.LLSSimpleVAEEncode()

        with mock.patch.object(image_nodes, "comfy_utils", comfy_utils):
            latent, width, height, task_context = node.encode(
                image=FakeTensor((1, 768, 1536, 3)),
                vae=vae,
                resize_mode="keep_aspect",
                size_source="model_recommended",
                width=512,
                height=512,
                task_context={
                    "model_family": "SDXL",
                    "resolved_model_family": "SDXL",
                    "recommended_width": 1024,
                    "recommended_height": 1024,
                    "vae_source": "embedded",
                },
            )

        self.assertEqual((width, height), (1024, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 128))
        self.assertEqual(latent["downscale_ratio_spacial"], 8)
        self.assertEqual(vae.encoded_shapes, [(1, 512, 1024, 3)])
        self.assertEqual(task_context["latent_source"], "image_encode")
        self.assertEqual(task_context["final_width"], 1024)
        self.assertEqual(task_context["final_height"], 512)
        self.assertEqual(task_context["resize_mode"], "keep_aspect")
        self.assertEqual(task_context["vae_source"], "embedded")
        self.assertEqual(task_context["input_image_width"], 1536)
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
        latent, width, height, task_context = node.encode(
            image=FakeTensor((1, 512, 768, 3)),
            vae=vae,
            resize_mode="none",
            size_source="input_image",
            width=1024,
            height=1024,
            task_context={"model_family": "SD1.5", "resolved_model_family": "SD1.5", "vae_source": "embedded"},
        )

        self.assertEqual((width, height), (768, 512))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 64, 96))
        self.assertEqual(latent["downscale_ratio_spacial"], 8)
        self.assertEqual(vae.encoded_shapes, [(1, 512, 768, 3)])
        self.assertEqual(task_context["latent_source"], "image_encode")
        self.assertEqual(task_context["size_source"], "input_image")
        self.assertEqual(task_context["resize_mode"], "none")

    def test_upscale_switcher_passthroughs_when_task_context_disables_upscale(self):
        load_plugin_package()
        from lls_node_test_refactor.upscale import nodes as upscale_nodes

        recorder_calls = []

        def fake_pytorch(self, image, scale, interpolation):
            recorder_calls.append(("pytorch", scale, interpolation))
            return ("INTERP_RESULT",)

        with mock.patch.object(
            upscale_nodes.LLSUpscaleSwitcher,
            "_upscale_with_pytorch",
            new=fake_pytorch,
        ):
            node = upscale_nodes.LLSUpscaleSwitcher()
            image, task_context = node.upscale(
                image="image",
                mode="upscale_model",
                scale=2.0,
                interpolation="bilinear",
                model_name=upscale_nodes.NO_UPSCALE_MODEL_PLACEHOLDER,
                tile=512,
                overlap=32,
                task_context={"resolved_model_family": "SDXL", "enable_upscale": False},
            )

        self.assertEqual(image, "image")
        self.assertEqual(task_context["upscaled"], False)
        self.assertEqual(task_context["upscale_mode"], "disabled_by_task_context")
        self.assertEqual(recorder_calls, [])

    def test_save_image_merges_metadata_from_task_context(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 512, 512, 3)),
                filename_prefix="LLS",
                save_metadata=True,
                task_context={
                    "model_family": "SD1.5",
                    "resolved_model_family": "SD1.5",
                    "checkpoint_name": "sd15.safetensors",
                    "vae_name": "VAE::SD15",
                    "text_encoder_name_1": "clip_l.safetensors",
                    "positive_prompt": "cat",
                    "negative_prompt": "bad",
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "guidance": None,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "final_width": 512,
                    "final_height": 512,
                    "batch_size": 1,
                    "upscale_mode": "none",
                    "upscale_scale": 1.0,
                },
            )

        metadata = result["ui"]["images"][0]["extra_pnginfo"]["lls_metadata"]
        self.assertEqual(metadata["positive_prompt"], "cat")
        self.assertEqual(metadata["checkpoint_name"], "sd15.safetensors")
        self.assertEqual(metadata["steps"], 20)
        self.assertEqual(metadata["upscale_mode"], "none")


if __name__ == "__main__":
    unittest.main()
