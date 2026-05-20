import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin_package():
    for name in list(sys.modules):
        if name == "lls_node_test_qwen" or name.startswith("lls_node_test_qwen."):
            sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(
        "lls_node_test_qwen",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lls_node_test_qwen"] = module
    spec.loader.exec_module(module)
    return module


class NodeOutputStub:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args

    def __getitem__(self, index):
        return self.args[index]


class QwenFolderPathsStub:
    def __init__(self):
        self._files = {
            "diffusion_models": [
                "qwen_image_fp8_e4m3fn.safetensors",
                "qwen_image_2512_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2511_bf16.safetensors",
                "qwen_image_layered_bf16.safetensors",
                "flux1-dev.safetensors",
            ],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
        }

    def get_filename_list(self, category):
        return list(self._files.get(category, []))


class QwenClipStub:
    def __init__(self, clip_name, clip_type):
        self.clip_name = clip_name
        self.clip_type = clip_type


class QwenVAEStub:
    def __init__(self, vae_name):
        self.vae_name = vae_name


class CoreQwenNodesStub:
    last_unet_call = None
    last_clip_call = None
    last_text_encode_calls = []
    last_ksampler_call = None
    last_vae_encode_call = None
    last_vae_decode_call = None
    last_vae_load_call = None

    @classmethod
    def reset(cls):
        cls.last_unet_call = None
        cls.last_clip_call = None
        cls.last_text_encode_calls = []
        cls.last_ksampler_call = None
        cls.last_vae_encode_call = None
        cls.last_vae_decode_call = None
        cls.last_vae_load_call = None

    class UNETLoader:
        def load_unet(self, unet_name, weight_dtype):
            CoreQwenNodesStub.last_unet_call = {
                "unet_name": unet_name,
                "weight_dtype": weight_dtype,
            }
            return (f"MODEL::{unet_name}",)

    class CLIPLoader:
        def load_clip(self, clip_name, type="stable_diffusion", device="default"):
            CoreQwenNodesStub.last_clip_call = {
                "clip_name": clip_name,
                "type": type,
                "device": device,
            }
            return (QwenClipStub(clip_name, type),)

    class VAELoader:
        def load_vae(self, vae_name):
            CoreQwenNodesStub.last_vae_load_call = {"vae_name": vae_name}
            return (QwenVAEStub(vae_name),)

    class CLIPTextEncode:
        def encode(self, clip, text):
            CoreQwenNodesStub.last_text_encode_calls.append(
                {
                    "clip_name": clip.clip_name,
                    "clip_type": clip.clip_type,
                    "text": text,
                }
            )
            return (f"COND::{text}",)

    class KSampler:
        def sample(
            self,
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=1.0,
        ):
            CoreQwenNodesStub.last_ksampler_call = {
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent_image": latent_image,
                "denoise": denoise,
            }
            return ({"samples": f"SAMPLED::{seed}::{steps}"},)

    class VAEDecode:
        def decode(self, vae, samples):
            CoreQwenNodesStub.last_vae_decode_call = {
                "vae_name": vae.vae_name,
                "samples": samples,
            }
            return (f"IMAGE::{samples['samples']}",)

    class VAEEncode:
        def encode(self, vae, pixels):
            CoreQwenNodesStub.last_vae_encode_call = {
                "vae_name": vae.vae_name,
                "pixels": pixels,
            }
            return ({"samples": f"ENCODED::{pixels}"},)


class SD3NodesStub:
    last_generate_call = None

    @classmethod
    def reset(cls):
        cls.last_generate_call = None

    class EmptySD3LatentImage:
        def generate(self, width, height, batch_size=1):
            SD3NodesStub.last_generate_call = {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            }
            return (
                {
                    "samples": f"EMPTY::{width}x{height}::{batch_size}",
                    "downscale_ratio_spacial": 8,
                },
            )


class ModelAdvancedStub:
    last_call = None

    @classmethod
    def reset(cls):
        cls.last_call = None

    class ModelSamplingAuraFlow:
        def patch_aura(self, model, shift):
            ModelAdvancedStub.last_call = {"model": model, "shift": shift}
            return (f"AURA::{model}::{shift}",)


class FluxNodesStub:
    last_scale_call = None
    last_reference_calls = []

    @classmethod
    def reset(cls):
        cls.last_scale_call = None
        cls.last_reference_calls = []

    class FluxKontextImageScale:
        @classmethod
        def execute(cls, image):
            FluxNodesStub.last_scale_call = {"image": image}
            return NodeOutputStub(f"SCALED::{image}")

    class FluxKontextMultiReferenceLatentMethod:
        @classmethod
        def execute(cls, conditioning, reference_latents_method):
            FluxNodesStub.last_reference_calls.append(
                {
                    "conditioning": conditioning,
                    "reference_latents_method": reference_latents_method,
                }
            )
            return NodeOutputStub(f"{conditioning}::{reference_latents_method}")


class QwenExtraNodesStub:
    last_calls = []

    @classmethod
    def reset(cls):
        cls.last_calls = []

    class TextEncodeQwenImageEditPlus:
        @classmethod
        def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None):
            QwenExtraNodesStub.last_calls.append(
                {
                    "clip_name": clip.clip_name,
                    "prompt": prompt,
                    "vae_name": getattr(vae, "vae_name", None),
                    "image1": image1,
                    "image2": image2,
                    "image3": image3,
                }
            )
            return NodeOutputStub(f"EDIT_COND::{prompt}::{image1}")


class CFGNodesStub:
    last_call = None

    @classmethod
    def reset(cls):
        cls.last_call = None

    class CFGNorm:
        @classmethod
        def execute(cls, model, strength):
            CFGNodesStub.last_call = {"model": model, "strength": strength}
            return NodeOutputStub(f"CFGNORM::{model}::{strength}")


class TestQwenNodes(unittest.TestCase):
    def setUp(self):
        CoreQwenNodesStub.reset()
        SD3NodesStub.reset()
        ModelAdvancedStub.reset()
        FluxNodesStub.reset()
        QwenExtraNodesStub.reset()
        CFGNodesStub.reset()

    def test_plugin_registers_qwen_nodes(self):
        plugin = load_plugin_package()

        self.assertIn("LLSQwenTextToImage", plugin.NODE_CLASS_MAPPINGS)
        self.assertIn("LLSQwenImageEdit", plugin.NODE_CLASS_MAPPINGS)

    def test_qwen_text_node_filters_only_text_models(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenTextToImage.INPUT_TYPES()["required"]["model_name"][0]

        self.assertEqual(
            choices,
            [
                "qwen_image_2512_fp8_e4m3fn.safetensors",
                "qwen_image_fp8_e4m3fn.safetensors",
            ],
        )

    def test_qwen_edit_node_filters_only_edit_models(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenImageEdit.INPUT_TYPES()["required"]["model_name"][0]

        self.assertEqual(choices, ["qwen_image_edit_2511_bf16.safetensors"])

    def test_qwen_text_node_uses_placeholder_when_no_compatible_models_exist(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        stub = QwenFolderPathsStub()
        stub._files["diffusion_models"] = ["flux1-dev.safetensors", "sdxl.safetensors"]
        with mock.patch.object(qwen_discovery, "folder_paths", stub):
            choices = qwen_nodes.LLSQwenTextToImage.INPUT_TYPES()["required"]["model_name"][0]

        self.assertEqual(choices, ["(no qwen text-to-image models found)"])

    def test_qwen_text_runtime_rejects_incompatible_model_name(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "not compatible with LLSQwenTextToImage"):
                qwen_runtime.run_qwen_text_to_image(
                    model_name="qwen_image_edit_2511_bf16.safetensors",
                    prompt="a cat",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                )

    def test_qwen_edit_runtime_rejects_incompatible_model_name(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "not compatible with LLSQwenImageEdit"):
                qwen_runtime.run_qwen_image_edit(
                    model_name="qwen_image_fp8_e4m3fn.safetensors",
                    image="IMAGE::input",
                    prompt="edit the cat",
                    steps=20,
                    seed=1,
                )

    def test_qwen_text_runtime_raises_when_qwen_text_encoder_is_missing(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        stub._files["text_encoders"] = []
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing Qwen text encoder"):
                qwen_runtime.run_qwen_text_to_image(
                    model_name="qwen_image_fp8_e4m3fn.safetensors",
                    prompt="a cat",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                )

    def test_qwen_text_runtime_raises_when_qwen_vae_is_missing(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        stub._files["vae"] = []
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing Qwen VAE"):
                qwen_runtime.run_qwen_text_to_image(
                    model_name="qwen_image_fp8_e4m3fn.safetensors",
                    prompt="a cat",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                )

    def test_qwen_text_runtime_executes_minimal_official_pipeline(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ), mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), mock.patch.object(
            qwen_runtime,
            "nodes_sd3",
            SD3NodesStub,
        ), mock.patch.object(qwen_runtime, "nodes_model_advanced", ModelAdvancedStub):
            image = qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                width=1024,
                height=1152,
                steps=20,
                seed=99,
                batch_size=2,
            )

        self.assertEqual(image, "IMAGE::SAMPLED::99::20")
        self.assertEqual(CoreQwenNodesStub.last_clip_call["type"], "qwen_image")
        self.assertEqual(SD3NodesStub.last_generate_call, {"width": 1024, "height": 1152, "batch_size": 2})
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 4.0)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["sampler_name"], "euler")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["scheduler"], "simple")
        self.assertEqual(CoreQwenNodesStub.last_text_encode_calls[0]["text"], "a cat")
        self.assertEqual(CoreQwenNodesStub.last_text_encode_calls[1]["text"], "")
        self.assertEqual(ModelAdvancedStub.last_call["shift"], 3.1)

    def test_qwen_edit_runtime_executes_minimal_official_pipeline(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ), mock.patch.object(qwen_runtime, "comfy_core_nodes", CoreQwenNodesStub), mock.patch.object(
            qwen_runtime,
            "nodes_model_advanced",
            ModelAdvancedStub,
        ), mock.patch.object(qwen_runtime, "nodes_qwen", QwenExtraNodesStub), mock.patch.object(
            qwen_runtime,
            "nodes_flux",
            FluxNodesStub,
        ), mock.patch.object(qwen_runtime, "nodes_cfg", CFGNodesStub):
            image = qwen_runtime.run_qwen_image_edit(
                model_name="qwen_image_edit_2511_bf16.safetensors",
                image="IMAGE::input",
                prompt="turn the cat blue",
                steps=30,
                seed=123,
            )

        self.assertEqual(image, "IMAGE::SAMPLED::123::30")
        self.assertEqual(FluxNodesStub.last_scale_call, {"image": "IMAGE::input"})
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["prompt"], "turn the cat blue")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image1"], "SCALED::IMAGE::input")
        self.assertEqual(QwenExtraNodesStub.last_calls[1]["prompt"], "")
        self.assertEqual(len(FluxNodesStub.last_reference_calls), 2)
        self.assertEqual(FluxNodesStub.last_reference_calls[0]["reference_latents_method"], "index_timestep_zero")
        self.assertEqual(CFGNodesStub.last_call["strength"], 1.0)
        self.assertEqual(CoreQwenNodesStub.last_vae_encode_call["pixels"], "SCALED::IMAGE::input")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 4.0)

    def test_qwen_text_node_returns_image_only(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenTextToImage"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("image",))
        self.assertEqual(
            tuple(required.keys()),
            ("model_name", "prompt", "width", "height", "steps", "seed", "batch_size"),
        )

    def test_qwen_edit_node_returns_image_only(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenImageEdit"]
        required = node_cls.INPUT_TYPES()["required"]

        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("image",))
        self.assertEqual(tuple(required.keys()), ("model_name", "image", "prompt", "steps", "seed"))

    def test_qwen_text_node_executes_runtime(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import nodes as qwen_nodes
        node = qwen_nodes.LLSQwenTextToImage()

        with mock.patch.object(qwen_nodes.runtime, "run_qwen_text_to_image", return_value="IMAGE::node"):
            result = node.generate(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                width=1024,
                height=1024,
                steps=20,
                seed=1,
                batch_size=1,
            )

        self.assertEqual(result, ("IMAGE::node",))

    def test_qwen_edit_node_executes_runtime(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import nodes as qwen_nodes
        node = qwen_nodes.LLSQwenImageEdit()

        with mock.patch.object(qwen_nodes.runtime, "run_qwen_image_edit", return_value="IMAGE::node-edit"):
            result = node.generate(
                model_name="qwen_image_edit_2511_bf16.safetensors",
                image="IMAGE::input",
                prompt="edit the cat",
                steps=20,
                seed=1,
            )

        self.assertEqual(result, ("IMAGE::node-edit",))


if __name__ == "__main__":
    unittest.main()
