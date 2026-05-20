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
                "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2511_bf16.safetensors",
                "qwen_image_layered_bf16.safetensors",
                "flux1-dev.safetensors",
            ],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
            "loras": [
                "Qwen-Image-Lightning-8steps-V1.0.safetensors",
                "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
                "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                "Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors",
                "flux-dev-style.safetensors",
            ],
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
    last_lora_calls = []

    @classmethod
    def reset(cls):
        cls.last_unet_call = None
        cls.last_clip_call = None
        cls.last_text_encode_calls = []
        cls.last_ksampler_call = None
        cls.last_vae_encode_call = None
        cls.last_vae_decode_call = None
        cls.last_vae_load_call = None
        cls.last_lora_calls = []

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

    class LoraLoaderModelOnly:
        def load_lora_model_only(self, model, lora_name, strength_model):
            CoreQwenNodesStub.last_lora_calls.append(
                {
                    "model": model,
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                }
            )
            return (f"LORA::{model}::{lora_name}::{strength_model}",)

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
            return NodeOutputStub(f"EDIT_COND::{prompt}::{image1}::{image2}::{image3}")


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
        self.assertNotIn("LLSQwenLoRAStack", plugin.NODE_CLASS_MAPPINGS)

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

        self.assertEqual(
            choices,
            [
                "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "qwen_image_edit_2511_bf16.safetensors",
            ],
        )

    def test_qwen_text_node_schema_exposes_advanced_inputs_and_optional_model(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenTextToImage"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("image",))
        self.assertEqual(
            tuple(required.keys()),
            (
                "model_name",
                "prompt",
                "width",
                "height",
                "steps",
                "seed",
                "batch_size",
                "negative_prompt",
                "cfg",
                "sampler_name",
                "scheduler",
                "shift",
                "enable_turbo_mode",
                "turbo_lora_name",
                "turbo_strength",
            ),
        )
        self.assertEqual(required["negative_prompt"][1]["default"], "")
        self.assertEqual(required["cfg"][1]["default"], 4.0)
        self.assertIn("euler", required["sampler_name"][0])
        self.assertIn("simple", required["scheduler"][0])
        self.assertEqual(required["shift"][1]["default"], 3.1)
        self.assertEqual(required["enable_turbo_mode"][1]["default"], False)
        self.assertEqual(optional["model"][0], "MODEL")

    def test_qwen_edit_node_schema_exposes_advanced_inputs_optional_images_and_model(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSQwenImageEdit"]
        required = node_cls.INPUT_TYPES()["required"]
        optional = node_cls.INPUT_TYPES()["optional"]

        self.assertEqual(node_cls.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node_cls.RETURN_NAMES, ("image",))
        self.assertEqual(
            tuple(required.keys()),
            (
                "model_name",
                "image",
                "prompt",
                "steps",
                "seed",
                "negative_prompt",
                "cfg",
                "sampler_name",
                "scheduler",
                "shift",
                "cfg_norm_strength",
                "reference_latents_method",
                "enable_turbo_mode",
                "turbo_lora_name",
                "turbo_strength",
            ),
        )
        self.assertEqual(optional["image2"][0], "IMAGE")
        self.assertEqual(optional["image3"][0], "IMAGE")
        self.assertEqual(optional["model"][0], "MODEL")
        self.assertEqual(
            required["reference_latents_method"][0],
            ["offset", "index", "uxo/uno", "index_timestep_zero"],
        )

    def test_qwen_text_node_turbo_lora_choices_filter_only_text_loras(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenTextToImage.INPUT_TYPES()["required"]["turbo_lora_name"][0]

        self.assertEqual(
            choices,
            [
                "(auto)",
                "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
                "Qwen-Image-Lightning-8steps-V1.0.safetensors",
            ],
        )

    def test_qwen_edit_node_turbo_lora_choices_filter_only_edit_loras(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        with mock.patch.object(qwen_discovery, "folder_paths", QwenFolderPathsStub()):
            choices = qwen_nodes.LLSQwenImageEdit.INPUT_TYPES()["required"]["turbo_lora_name"][0]

        self.assertEqual(
            choices,
            [
                "(auto)",
                "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                "Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors",
            ],
        )

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
                    negative_prompt="",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    enable_turbo_mode=False,
                    turbo_lora_name="(auto)",
                    turbo_strength=1.0,
                    model=None,
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
                    image2=None,
                    image3=None,
                    prompt="edit the cat",
                    negative_prompt="",
                    steps=20,
                    seed=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    cfg_norm_strength=1.0,
                    reference_latents_method="index_timestep_zero",
                    enable_turbo_mode=False,
                    turbo_lora_name="(auto)",
                    turbo_strength=1.0,
                    model=None,
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
                    negative_prompt="",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    enable_turbo_mode=False,
                    turbo_lora_name="(auto)",
                    turbo_strength=1.0,
                    model=None,
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
                    negative_prompt="",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    enable_turbo_mode=False,
                    turbo_lora_name="(auto)",
                    turbo_strength=1.0,
                    model=None,
                )

    def test_qwen_text_runtime_executes_internal_pipeline_when_model_not_connected(self):
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
                negative_prompt="low quality",
                width=1024,
                height=1152,
                steps=33,
                seed=99,
                batch_size=2,
                cfg=7.5,
                sampler_name="heun",
                scheduler="normal",
                shift=2.25,
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
                model=None,
            )

        self.assertEqual(image, "IMAGE::SAMPLED::99::33")
        self.assertEqual(CoreQwenNodesStub.last_unet_call["unet_name"], "qwen_image_fp8_e4m3fn.safetensors")
        self.assertEqual(CoreQwenNodesStub.last_text_encode_calls[0]["text"], "a cat")
        self.assertEqual(CoreQwenNodesStub.last_text_encode_calls[1]["text"], "low quality")
        self.assertEqual(SD3NodesStub.last_generate_call, {"width": 1024, "height": 1152, "batch_size": 2})
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["steps"], 33)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 7.5)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["sampler_name"], "heun")
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["scheduler"], "normal")
        self.assertEqual(ModelAdvancedStub.last_call["shift"], 2.25)

    def test_qwen_text_runtime_uses_external_model_without_internal_unet_load(self):
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
                negative_prompt="low quality",
                width=1024,
                height=1024,
                steps=20,
                seed=5,
                batch_size=1,
                cfg=4.5,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
                model="MODEL::external",
            )

        self.assertEqual(image, "IMAGE::SAMPLED::5::20")
        self.assertIsNone(CoreQwenNodesStub.last_unet_call)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 4.5)
        self.assertEqual(ModelAdvancedStub.last_call["model"], "MODEL::external")

    def test_qwen_text_runtime_applies_turbo_lora_on_external_model(self):
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
            qwen_runtime.run_qwen_text_to_image(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                negative_prompt="",
                width=1024,
                height=1024,
                steps=50,
                seed=99,
                batch_size=1,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                enable_turbo_mode=True,
                turbo_lora_name="(auto)",
                turbo_strength=0.75,
                model="MODEL::external",
            )

        self.assertIsNone(CoreQwenNodesStub.last_unet_call)
        self.assertEqual(CoreQwenNodesStub.last_lora_calls[0]["model"], "MODEL::external")
        self.assertEqual(
            CoreQwenNodesStub.last_lora_calls[0]["lora_name"],
            "Qwen-Image-Lightning-8steps-V1.0.safetensors",
        )
        self.assertEqual(CoreQwenNodesStub.last_lora_calls[0]["strength_model"], 0.75)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["steps"], 8)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 1.0)

    def test_qwen_text_runtime_rejects_incompatible_manual_turbo_lora(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import discovery as qwen_discovery
        from lls_node_test_qwen.qwen import runtime as qwen_runtime

        stub = QwenFolderPathsStub()
        with mock.patch.object(qwen_discovery, "folder_paths", stub), mock.patch.object(
            qwen_runtime,
            "folder_paths",
            stub,
        ):
            with self.assertRaisesRegex(RuntimeError, "Turbo LoRA .* is not compatible"):
                qwen_runtime.run_qwen_text_to_image(
                    model_name="qwen_image_fp8_e4m3fn.safetensors",
                    prompt="a cat",
                    negative_prompt="",
                    width=1024,
                    height=1024,
                    steps=20,
                    seed=1,
                    batch_size=1,
                    cfg=4.0,
                    sampler_name="euler",
                    scheduler="simple",
                    shift=3.1,
                    enable_turbo_mode=True,
                    turbo_lora_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                    turbo_strength=1.0,
                    model=None,
                )

    def test_qwen_edit_runtime_uses_external_model_without_internal_unet_load(self):
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
                model_name="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                image="IMAGE::input",
                image2="IMAGE::ref2",
                image3="IMAGE::ref3",
                prompt="turn the cat blue",
                negative_prompt="low quality",
                steps=28,
                seed=123,
                cfg=6.5,
                sampler_name="heun",
                scheduler="normal",
                shift=2.0,
                cfg_norm_strength=0.8,
                reference_latents_method="index",
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
                model="MODEL::external-edit",
            )

        self.assertEqual(image, "IMAGE::SAMPLED::123::28")
        self.assertIsNone(CoreQwenNodesStub.last_unet_call)
        self.assertEqual(FluxNodesStub.last_scale_call, {"image": "IMAGE::input"})
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["prompt"], "turn the cat blue")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image1"], "SCALED::IMAGE::input")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image2"], "IMAGE::ref2")
        self.assertEqual(QwenExtraNodesStub.last_calls[0]["image3"], "IMAGE::ref3")
        self.assertEqual(QwenExtraNodesStub.last_calls[1]["prompt"], "low quality")
        self.assertEqual(FluxNodesStub.last_reference_calls[0]["reference_latents_method"], "index")
        self.assertEqual(CFGNodesStub.last_call["strength"], 0.8)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 6.5)
        self.assertEqual(ModelAdvancedStub.last_call["model"], "MODEL::external-edit")

    def test_qwen_edit_runtime_supports_2511_turbo_when_matching_lora_exists(self):
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
            qwen_runtime.run_qwen_image_edit(
                model_name="qwen_image_edit_2511_bf16.safetensors",
                image="IMAGE::input",
                image2=None,
                image3=None,
                prompt="turn the cat blue",
                negative_prompt="",
                steps=20,
                seed=123,
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                cfg_norm_strength=1.0,
                reference_latents_method="index_timestep_zero",
                enable_turbo_mode=True,
                turbo_lora_name="(auto)",
                turbo_strength=0.7,
                model="MODEL::external-edit",
            )

        self.assertIsNone(CoreQwenNodesStub.last_unet_call)
        self.assertEqual(
            CoreQwenNodesStub.last_lora_calls[-1]["lora_name"],
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors",
        )
        self.assertEqual(CoreQwenNodesStub.last_lora_calls[-1]["strength_model"], 0.7)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["steps"], 4)
        self.assertEqual(CoreQwenNodesStub.last_ksampler_call["cfg"], 1.0)

    def test_qwen_text_node_executes_runtime(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        node = qwen_nodes.LLSQwenTextToImage()

        with mock.patch.object(qwen_nodes.runtime, "run_qwen_text_to_image", return_value="IMAGE::node") as runtime_mock:
            result = node.generate(
                model_name="qwen_image_fp8_e4m3fn.safetensors",
                prompt="a cat",
                width=1024,
                height=1024,
                steps=20,
                seed=1,
                batch_size=1,
                negative_prompt="",
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
                model="MODEL::external",
            )

        self.assertEqual(result, ("IMAGE::node",))
        self.assertEqual(runtime_mock.call_args.kwargs["model"], "MODEL::external")

    def test_qwen_edit_node_executes_runtime(self):
        load_plugin_package()
        from lls_node_test_qwen.qwen import nodes as qwen_nodes

        node = qwen_nodes.LLSQwenImageEdit()

        with mock.patch.object(qwen_nodes.runtime, "run_qwen_image_edit", return_value="IMAGE::node-edit") as runtime_mock:
            result = node.generate(
                model_name="qwen_image_edit_2511_bf16.safetensors",
                image="IMAGE::input",
                prompt="edit the cat",
                steps=20,
                seed=1,
                image2=None,
                image3=None,
                negative_prompt="",
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                shift=3.1,
                cfg_norm_strength=1.0,
                reference_latents_method="index_timestep_zero",
                enable_turbo_mode=False,
                turbo_lora_name="(auto)",
                turbo_strength=1.0,
                model="MODEL::external",
            )

        self.assertEqual(result, ("IMAGE::node-edit",))
        self.assertEqual(runtime_mock.call_args.kwargs["model"], "MODEL::external")


if __name__ == "__main__":
    unittest.main()
