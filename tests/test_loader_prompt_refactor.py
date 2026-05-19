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
                "flux-schnell.safetensors",
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
    class CLIPType:
        STABLE_DIFFUSION = "stable_diffusion"
        FLUX = "flux"

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

    def load_clip(
        self,
        ckpt_paths,
        embedding_directory=None,
        clip_type=None,
        model_options=None,
        disable_dynamic=False,
    ):
        joined = "+".join(pathlib.Path(path).name for path in ckpt_paths)
        return f"CLIP::{clip_type}::{joined}"


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
        self.assertIn("SD15", family_choices)
        self.assertIn("SDXL_TURBO", family_choices)
        self.assertIn("FLUX_SCHNELL", family_choices)
        self.assertIn("FLUX_DEV", family_choices)

    def test_loader_returns_json_model_info_for_sd15_embedded_resources(self):
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
                "SD15",
                "simple",
                "auto",
                "auto",
                "(auto)",
                "(auto)",
                "(auto)",
            )

        info = json.loads(model_info)
        self.assertEqual(model, "MODEL::SD15")
        self.assertEqual(text_encoder, "CLIP::SD15")
        self.assertEqual(vae, "VAE::SD15")
        self.assertEqual(info["family"], "SD15")
        self.assertEqual(info["text_encoder_type"], "clip")
        self.assertTrue(info["has_embedded_vae"])
        self.assertEqual(info["required_text_encoders"], ["clip"])
        self.assertEqual(info["required_vae"], "optional")
        self.assertFalse(info["is_turbo"])

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
                "diffusion_models/flux-schnell.safetensors",
                "FLUX_SCHNELL",
                "advanced",
                "external",
                "external",
                "ae.safetensors",
                "clip_l.safetensors",
                "t5xxl_fp16.safetensors",
            )

        info = json.loads(model_info)
        self.assertEqual(model, "MODEL::flux-schnell.safetensors")
        self.assertEqual(text_encoder, "CLIP::flux::clip_l.safetensors+t5xxl_fp16.safetensors")
        self.assertEqual(vae, "VAE::ae.safetensors")
        self.assertEqual(info["family"], "FLUX_SCHNELL")
        self.assertEqual(info["text_encoder_type"], "flux_clip_l_t5xxl")
        self.assertEqual(info["required_text_encoders"], ["clip_l", "t5xxl"])
        self.assertEqual(info["required_vae"], "required")

    def test_loader_raises_clear_error_when_flux_resources_are_missing(self):
        load_plugin_package()
        from lls_node_test_refactor.model_loader import nodes as loader_nodes

        with mock.patch.object(loader_nodes, "folder_paths", FolderPathsStub()), mock.patch.object(
            loader_nodes,
            "comfy_sd",
            ComfySDStub(),
        ), mock.patch.object(loader_nodes, "comfy_core_nodes", CoreNodesStub()):
            node = loader_nodes.LLSSimpleCheckpointLoader()
            with self.assertRaisesRegex(RuntimeError, "clip_l.*t5xxl"):
                node.load_checkpoint(
                    "diffusion_models/flux-schnell.safetensors",
                    "FLUX_DEV",
                    "simple",
                    "external",
                    "external",
                    "ae.safetensors",
                    "(auto)",
                    "(auto)",
                )

    def test_prompt_encode_dispatches_sdxl_using_model_info_dimensions(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        model_info = json.dumps({"family": "SDXL", "base_width": 1216, "base_height": 832})

        positive, negative, prompt_info = node.encode(
            clip,
            "a castle",
            "low quality",
            -1,
            model_info=model_info,
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
        self.assertIn("family=SDXL", prompt_info)

    def test_prompt_encode_dispatches_flux_and_neutralizes_negative_prompt(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimplePromptEncode"]
        node = node_cls()
        clip = RecordingClip()
        model_info = json.dumps({"family": "FLUX_DEV", "guidance": 3.5, "guidance_embed": True})

        positive, negative, prompt_info = node.encode(
            clip,
            "a robot in a forest",
            "ugly, blurry",
            -1,
            model_info=model_info,
        )

        self.assertEqual(len(clip.calls), 2)
        self.assertIn("t5xxl", clip.calls[0]["tokens"])
        self.assertEqual(clip.calls[0]["add_dict"]["guidance"], 3.5)
        self.assertEqual(clip.calls[1]["tokens"]["text"], "")
        self.assertEqual(positive[0][0], "cond::a robot in a forest")
        self.assertEqual(negative[0][0], "cond::")
        self.assertIn("negative_ignored_for_flux", prompt_info)

    def test_empty_latent_uses_flux_shape_when_model_info_requests_it(self):
        load_plugin_package()
        from lls_node_test_refactor.latent import nodes as latent_nodes

        node = latent_nodes.LLSSimpleEmptyLatent()
        model_info = json.dumps({"family": "FLUX_SCHNELL", "latent_channels": 128, "downscale_ratio": 16})

        with mock.patch.object(latent_nodes, "torch", FakeTorch()):
            latent, width, height, latent_info = node.create_empty_latent(
                "Custom",
                1000,
                1000,
                2,
                model_info=model_info,
            )

        self.assertEqual((width, height), (1008, 1008))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 63, 63))
        self.assertEqual(latent["downscale_ratio_spacial"], 16)
        self.assertIn("family=FLUX_SCHNELL", latent_info)

    def test_ksampler_uses_turbo_preset_defaults_when_model_info_requests_it(self):
        load_plugin_package()
        from lls_node_test_refactor.sampling import nodes as sampling_nodes

        recorded = {}

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
                model="model",
                positive="positive",
                negative="negative",
                latent_image={"samples": FakeTensor((1, 4, 64, 64))},
                quality_preset="Balanced",
                seed=123,
                steps=30,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                model_info=json.dumps({"family": "SDXL_TURBO", "is_turbo": True}),
            )

        self.assertEqual(recorded["steps"], 6)
        self.assertEqual(recorded["cfg"], 1.5)
        self.assertEqual(latent["samples"].shape, (1, 4, 64, 64))
        self.assertIn("family=SDXL_TURBO", sample_info)

    def test_vae_decode_uses_model_info_downscale_ratio_for_flux(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleVAEDecode"]
        node = node_cls()

        image, decode_info = node.decode(
            {"samples": FakeTensor((1, 128, 64, 64))},
            FakeVAE(),
            model_info=json.dumps({"family": "FLUX_DEV", "downscale_ratio": 16}),
        )

        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertIn("decoded=1024x1024", decode_info)


if __name__ == "__main__":
    unittest.main()
