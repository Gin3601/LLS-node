import json
import types
import unittest
from unittest import mock

try:
    from .test_repair_helpers import FakeLatentTensor, load_plugin_package
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeLatentTensor, load_plugin_package


class TestRepairSampler(unittest.TestCase):
    def test_sampler_schema_exposes_repair_inputs(self):
        plugin = load_plugin_package()
        node_cls = plugin.NODE_CLASS_MAPPINGS["LLSSimpleKSampler"]
        schema = node_cls.INPUT_TYPES()

        self.assertEqual(schema["required"]["denoise_mode"][0], ["manual", "auto_from_repair"])
        self.assertEqual(schema["required"]["adapter_mode"][0], ["auto", "sd_classic", "flux", "sd3", "qwen", "zimage"])
        self.assertEqual(schema["optional"]["repair_info"][0], "LLS_REPAIR_INFO")
        self.assertEqual(schema["optional"]["guidance_stack"][0], "LLS_GUIDANCE_STACK")
        self.assertEqual(schema["optional"]["model_info"][0], "STRING")

    def test_sampler_keeps_manual_denoise_without_repair_info(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "empty_latent"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            return_value={"samples": "done"},
        ) as common:
            result_latent, sample_info = node.sample(
                model=types.SimpleNamespace(_lls_family="SD1.5"),
                positive="positive",
                negative="negative",
                latent_image=latent,
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="karras",
                denoise=0.33,
                denoise_mode="manual",
                adapter_mode="auto",
                flux_guidance=3.5,
                model_family="Auto",
                repair_info=None,
                guidance_stack=None,
                model_info=None,
            )

        self.assertEqual(result_latent["samples"], "done")
        self.assertEqual(common.call_args.kwargs["denoise"], 0.33)
        payload = json.loads(sample_info)
        self.assertFalse(payload["repair_mode"])
        self.assertFalse(payload["guidance_used"])

    def test_sampler_uses_repair_denoise_when_requested(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "repair_prepare_region"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ), mock.patch.object(
            sampling_nodes,
            "_common_ksampler",
            return_value={"samples": "done"},
        ) as common:
            _result_latent, sample_info = node.sample(
                model=types.SimpleNamespace(_lls_family="SDXL"),
                positive="positive",
                negative="negative",
                latent_image=latent,
                quality_preset="Manual",
                seed=7,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="karras",
                denoise=0.33,
                denoise_mode="auto_from_repair",
                adapter_mode="auto",
                flux_guidance=3.5,
                model_family="Auto",
                repair_info={
                    "repair_scope": "region",
                    "repair_kernel": "latent_mask",
                    "recommended_denoise": 0.61,
                    "model_family": "SDXL",
                    "model_role": "normal",
                },
                guidance_stack={"kind": "placeholder"},
                model_info=None,
            )

        self.assertEqual(common.call_args.kwargs["denoise"], 0.61)
        payload = json.loads(sample_info)
        self.assertTrue(payload["repair_mode"])
        self.assertTrue(payload["guidance_used"])
        self.assertEqual(payload["repair_scope"], "region")
        self.assertEqual(payload["repair_kernel"], "latent_mask")

    def test_sampler_rejects_unsupported_qwen_adapter(self):
        load_plugin_package()
        from lls_node_test_repair.sampling import nodes as sampling_nodes

        node = sampling_nodes.LLSSimpleKSampler()
        latent = {"samples": FakeLatentTensor((1, 4, 64, 64)), "source": "repair_prepare_region"}
        with mock.patch.object(sampling_nodes, "comfy_sample", object()), mock.patch.object(
            sampling_nodes,
            "comfy_samplers",
            types.SimpleNamespace(KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["karras"])),
        ):
            with self.assertRaisesRegex(RuntimeError, "QWEN"):
                node.sample(
                    model=types.SimpleNamespace(_lls_family="SD1.5"),
                    positive="positive",
                    negative="negative",
                    latent_image=latent,
                    quality_preset="Manual",
                    seed=7,
                    steps=20,
                    cfg=7.0,
                    sampler_name="euler",
                    scheduler="karras",
                    denoise=0.33,
                    denoise_mode="auto_from_repair",
                    adapter_mode="auto",
                    flux_guidance=3.5,
                    model_family="Auto",
                    repair_info={
                        "repair_scope": "region",
                        "repair_kernel": "vae_inpaint",
                        "recommended_denoise": 0.55,
                        "model_family": "QWEN",
                        "model_role": "unknown",
                    },
                    guidance_stack=None,
                    model_info=None,
                )


if __name__ == "__main__":
    unittest.main()
