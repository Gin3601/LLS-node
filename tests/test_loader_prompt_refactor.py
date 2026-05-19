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

    def reshape(self, *shape):
        return FakeTensor(shape)


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def zeros(shape, device=None, dtype=None, layout=None):
        return FakeTensor(shape)


class TestLoaderPromptRefactor(unittest.TestCase):
    def test_loader_schema_exposes_new_family_and_source_controls(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleCheckpointLoader"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(node_cls.RETURN_TYPES, ("MODEL", "LLS_TEXT_ENCODER", "VAE", "LLS_MODEL_INFO"))
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

    def test_loader_returns_structured_model_info_for_sd15_embedded_resources(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, text_encoder, vae, model_info = node.load_checkpoint(
                "sd15.safetensors",
                "SD1.5",
                "simple",
                "auto",
                "auto",
                "(auto)",
                "(auto)",
                "(auto)",
            )

        self.assertEqual(model, "MODEL::SD15")
        self.assertEqual(text_encoder, "CLIP::SD15")
        self.assertEqual(vae, "VAE::SD15")
        self.assertEqual(model_info["family"], "SD1.5")
        self.assertEqual(model_info["text_encoder_type"], "clip")
        self.assertTrue(model_info["has_embedded_vae"])
        self.assertEqual(model_info["required_text_encoders"], ["clip"])
        self.assertEqual(model_info["required_vae"], "optional")
        self.assertEqual(model_info["default_width"], 512)

    def test_loader_builds_flux_resources_from_external_models(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            model, text_encoder, vae, model_info = node.load_checkpoint(
                "diffusion_models/flux1-schnell.safetensors",
                "FLUX_SCHNELL",
                "advanced",
                "external",
                "external",
                "ae.safetensors",
                "clip_l.safetensors",
                "t5xxl_fp16.safetensors",
            )

        self.assertEqual(model, "MODEL::flux1-schnell.safetensors")
        self.assertEqual(text_encoder, "CLIP::flux::clip_l.safetensors+t5xxl_fp16.safetensors")
        self.assertEqual(vae, "VAE::ae.safetensors")
        self.assertEqual(model_info["family"], "FLUX_SCHNELL")
        self.assertEqual(model_info["text_encoder_type"], "flux_clip_l_t5xxl")
        self.assertEqual(model_info["required_text_encoders"], ["clip_l", "t5xxl"])
        self.assertEqual(model_info["required_vae"], "required")
        self.assertTrue(model_info["is_flux"])
        self.assertEqual(model_info["default_guidance"], 3.5)

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

    def test_prompt_encode_schema_uses_custom_model_info_port(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(optional["text_encoder"][0], "LLS_TEXT_ENCODER")
        self.assertEqual(optional["model_info"][0], "LLS_MODEL_INFO")

    def test_prompt_encode_dispatches_sdxl_using_model_info_dimensions(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        model_info = {"family": "SDXL", "default_width": 1216, "default_height": 832}

        positive, negative, prompt_info = node.encode(
            clip,
            "a castle",
            "low quality",
            -1,
            model_info=model_info,
        )

        prompt_data = json.loads(prompt_info)
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
        self.assertEqual(prompt_data["family"], "SDXL")

    def test_prompt_encode_dispatches_flux_and_neutralizes_negative_prompt(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        model_info = {"family": "FLUX_DEV", "guidance": 3.5, "guidance_embed": True}

        positive, negative, prompt_info = node.encode(
            clip,
            "a robot in a forest",
            "ugly, blurry",
            -1,
            model_info=model_info,
        )

        prompt_data = json.loads(prompt_info)
        self.assertEqual(len(clip.calls), 2)
        self.assertIn("t5xxl", clip.calls[0]["tokens"])
        self.assertEqual(clip.calls[0]["add_dict"]["guidance"], 3.5)
        self.assertEqual(clip.calls[1]["tokens"]["text"], "")
        self.assertEqual(positive[0][0], "cond::a robot in a forest")
        self.assertEqual(negative[0][0], "cond::")
        self.assertEqual(prompt_data["negative_mode"], "ignored_for_flux")

    def test_empty_latent_uses_family_default_size_and_flux_shape(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()
        model_info = {"family": "FLUX_SCHNELL", "latent_channels": 128, "downscale_ratio": 16}

        with mock.patch.object(latent_nodes, "torch", FakeTorch()):
            latent, width, height, latent_info = node.create_empty_latent(
                "Family Default",
                512,
                512,
                2,
                model_info=model_info,
            )

        latent_data = json.loads(latent_info)
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 64, 64))
        self.assertEqual(latent["downscale_ratio_spacial"], 16)
        self.assertEqual(latent_data["family"], "FLUX_SCHNELL")

    def test_ksampler_uses_turbo_family_defaults_and_guidance(self):
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
            latent, sample_info = node.sample(
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
                model_info={"family": "SDXL_TURBO", "is_turbo": True},
            )

        sample_data = json.loads(sample_info)
        self.assertEqual(recorded["steps"], 4)
        self.assertEqual(recorded["cfg"], 1.0)
        self.assertEqual(sample_data["family"], "SDXL_TURBO")
        self.assertEqual(sample_data["sampler_name"], "euler")
        self.assertEqual(latent["samples"].shape, (1, 4, 64, 64))

    def test_vae_decode_uses_model_info_downscale_ratio_for_flux(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEDecode"]
        node = node_cls()

        image, decode_info = node.decode(
            {"samples": FakeTensor((1, 128, 64, 64))},
            FakeVAE(),
            model_info={"family": "FLUX_DEV", "downscale_ratio": 16, "vae_name": "ae.safetensors"},
        )

        decode_data = json.loads(decode_info)
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertEqual(decode_data["width"], 1024)
        self.assertEqual(decode_data["vae_name"], "ae.safetensors")

    def test_upscale_switcher_falls_back_to_interpolation_with_info(self):
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
            image, upscale_info = node.upscale(
                image="image",
                mode="upscale_model",
                scale=2.0,
                interpolation="bilinear",
                model_name=upscale_nodes.NO_UPSCALE_MODEL_PLACEHOLDER,
                tile=512,
                overlap=32,
                model_info={"family": "SDXL"},
            )

        info = json.loads(upscale_info)
        self.assertEqual(image, "INTERP_RESULT")
        self.assertEqual(info["mode"], "interpolation")
        self.assertEqual(info["warning"], "no_upscale_model_found_fallback_to_interpolation")
        self.assertEqual(recorder_calls, [("pytorch", 2.0, "bilinear")])

    def test_save_image_merges_metadata_from_pipeline_info(self):
        load_plugin_package()
        from lls_node_test_refactor.image import nodes as image_nodes

        with mock.patch.object(image_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = image_nodes.LLSSaveImage()
            result = node.save(
                image=FakeTensor((1, 512, 512, 3)),
                filename_prefix="LLS",
                save_metadata=True,
                model_info={
                    "family": "SD1.5",
                    "checkpoint_name": "sd15.safetensors",
                    "vae_name": "VAE::SD15",
                    "text_encoder_name_1": "clip_l.safetensors",
                },
                prompt_info=json.dumps({"positive_prompt": "cat", "negative_prompt": "bad"}),
                latent_info=json.dumps({"width": 512, "height": 512, "batch_size": 1}),
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
                decode_info=json.dumps({"vae_name": "VAE::SD15", "width": 512, "height": 512, "batch_size": 1}),
                upscale_info=json.dumps({"mode": "none", "scale": 1.0}),
            )

        metadata = result["ui"]["images"][0]["extra_pnginfo"]["lls_metadata"]
        self.assertEqual(metadata["positive_prompt"], "cat")
        self.assertEqual(metadata["checkpoint_name"], "sd15.safetensors")
        self.assertEqual(metadata["steps"], 20)
        self.assertEqual(metadata["upscale_mode"], "none")


if __name__ == "__main__":
    unittest.main()
